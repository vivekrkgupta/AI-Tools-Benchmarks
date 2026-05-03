#!/usr/bin/env python3
"""
Memory-fabric contention across CPU, GPU, and Apple Neural Engine.

This is the canonical edge receipt for the article. It approximates a real
agentic-edge workload running concurrently on a single SoC:
  - GPU: sustained matmul via MLX (analog of LLM token generation)
  - ANE: CNN inference via Core ML (analog of perception / multimodal heads)
  - CPU: configurable streaming-kernel threads on 64 MB working sets
         (analog of an orchestration loop manipulating agent state)

Thesis: compute units are independent, but the memory fabric is shared.
GPU and ANE never touch each other's compute pipelines, but as soon as they
run concurrently, they compete for memory bandwidth on the SoC. CPU
orchestration traffic compounds the contention. The article's argument that
on-device AI hits the data center memory-fabric problem earlier in the
lifecycle than hyperscalers do is grounded in this experiment.

Why the ANE matters specifically: most edge devices ship NPUs, not high-end
discrete GPUs. NPUs in Copilot+ PCs, in-vehicle compute (NXP, Qualcomm,
MediaTek auto), and smartphones all share a memory fabric with the CPU.
Demonstrating the contention on the ANE generalizes the thesis to the
entire on-device AI category, not just Apple Silicon.

A simpler GPU-only variant of this experiment is in
`cpu_gpu_memory_contention.py`.

Usage:
    python3 memory_fabric_contention.py             # ANE + CPU only
    python3 memory_fabric_contention.py --mixed     # GPU + ANE + CPU (canonical)
    python3 memory_fabric_contention.py --duration 5.0
"""

import argparse
import os
import sys
import tempfile
import threading
import time

import numpy as np

CPU_PAYLOAD_ELEMS = 16 * 1024 * 1024  # 64 MB float32 per CPU thread
DEFAULT_DURATION = 3.0
CPU_THREAD_SWEEP = [0, 1, 2, 4, 6, 8]


def build_ane_model(cache_path: str) -> str:
    """Trace a small CNN, convert to Core ML mlprogram, save and return path."""
    import torch
    import torch.nn as nn
    import coremltools as ct

    # Larger model so ANE inference is memory-bandwidth-bound.
    # Tiny CNNs fit in cache and resist CPU contention; the agentic-edge
    # workloads we care about (1B+ parameter LLMs streamed through every
    # token) are not cache-resident. This stack approximates that regime
    # with a CNN large enough that weights + activations don't fit cache.
    class HeavyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(3, 128, 3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1, stride=2),
                nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 512, 3, padding=1, stride=2),
                nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 1024, 3, padding=1, stride=2),
                nn.BatchNorm2d(1024), nn.ReLU(),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(1024, 1000),
            )

        def forward(self, x):
            return self.layers(x)

    print("Building Core ML model (one-time)...", file=sys.stderr)
    model = HeavyCNN().eval()
    example_input = torch.randn(1, 3, 384, 384)
    traced = torch.jit.trace(model, example_input)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)],
        convert_to="mlprogram",
    )
    mlmodel.save(cache_path)
    return cache_path


def load_ane_model(path: str):
    import coremltools as ct
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


def measure_ane_throughput(model, duration_s: float, warmup: int = 5) -> tuple[int, float]:
    """Run ANE inference until duration elapsed; return (count, infer_per_s)."""
    input_arr = np.random.randn(1, 3, 384, 384).astype(np.float32)

    for _ in range(warmup):
        model.predict({"input": input_arr})

    n = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        model.predict({"input": input_arr})
        n += 1
    elapsed = time.perf_counter() - start
    return n, n / elapsed


def measure_gpu_throughput(duration_s: float, warmup: int = 3) -> tuple[int, float]:
    """Run sustained 4096x4096 fp32 matmul on the GPU; return (count, TFLOPS)."""
    import mlx.core as mx

    DIM = 4096
    a = mx.random.normal((DIM, DIM), dtype=mx.float32)
    b = mx.random.normal((DIM, DIM), dtype=mx.float32)
    mx.eval(a, b)

    for _ in range(warmup):
        c = a @ b
        mx.eval(c)

    flops_per_matmul = 2.0 * (DIM ** 3)
    n = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        c = a @ b
        mx.eval(c)
        n += 1
    elapsed = time.perf_counter() - start
    tflops = (n * flops_per_matmul) / elapsed / 1e12
    return n, tflops


def cpu_orchestration_worker(stop_event: threading.Event, working_set: tuple) -> None:
    """STREAM-TRIAD on a per-thread 64 MB working set until stop is signaled."""
    a, b, c = working_set
    alpha = np.float32(2.5)
    while not stop_event.is_set():
        np.multiply(c, alpha, out=a)
        np.add(a, b, out=a)


def run_with_concurrency(ane_model, n_cpu_threads: int, duration_s: float, mixed: bool):
    """Spawn n_cpu_threads doing TRIAD; measure ANE inference (and optionally
    GPU TFLOPS) under that load. Returns (ane_rate, gpu_tflops_or_None)."""
    stop = threading.Event()
    cpu_threads = []
    working_sets = []

    for _ in range(n_cpu_threads):
        a = np.zeros(CPU_PAYLOAD_ELEMS, dtype=np.float32)
        b = np.ones(CPU_PAYLOAD_ELEMS, dtype=np.float32)
        c = np.full(CPU_PAYLOAD_ELEMS, 0.5, dtype=np.float32)
        working_sets.append((a, b, c))

    for ws in working_sets:
        t = threading.Thread(target=cpu_orchestration_worker, args=(stop, ws))
        t.start()
        cpu_threads.append(t)

    if n_cpu_threads > 0:
        time.sleep(0.2)  # let CPU workers spin up

    gpu_holder = [None]
    if mixed:
        def gpu_run():
            gpu_holder[0] = measure_gpu_throughput(duration_s)
        gpu_thread = threading.Thread(target=gpu_run)
        gpu_thread.start()

    ane_n, ane_rate = measure_ane_throughput(ane_model, duration_s)

    if mixed:
        gpu_thread.join()

    stop.set()
    for t in cpu_threads:
        t.join()

    return ane_rate, (gpu_holder[0][1] if mixed else None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help=f"Measurement duration per run in seconds (default {DEFAULT_DURATION}).")
    p.add_argument("--mixed", action="store_true",
                   help="Also run GPU matmul concurrently to test full SoC contention.")
    p.add_argument("--model-cache", default=os.path.expanduser("~/.cache/article-programs/tiny_cnn.mlpackage"),
                   help="Where to cache the compiled Core ML model.")
    args = p.parse_args()

    print("Demo 4: Apple Neural Engine inference under CPU memory contention")
    print(f"  Duration per run: {args.duration}s")
    print(f"  CPU threads sweep: {CPU_THREAD_SWEEP}")
    print(f"  Mode: {'MIXED (CPU + GPU + ANE)' if args.mixed else 'ANE + CPU contention only'}")
    print()

    # Build the model on first run, cache for subsequent runs.
    os.makedirs(os.path.dirname(args.model_cache), exist_ok=True)
    if not os.path.exists(args.model_cache):
        build_ane_model(args.model_cache)
    model = load_ane_model(args.model_cache)

    if args.mixed:
        print(f"  {'CPU threads':>12}  {'ANE infer/s':>12}  {'GPU TFLOPS':>10}  {'ANE vs idle':>12}  {'GPU vs idle':>12}")
        print(f"  {'-'*12:>12}  {'-'*12:>12}  {'-'*10:>10}  {'-'*12:>12}  {'-'*12:>12}")
    else:
        print(f"  {'CPU threads':>12}  {'ANE infer/s':>12}  {'ANE vs idle':>12}")
        print(f"  {'-'*12:>12}  {'-'*12:>12}  {'-'*12:>12}")

    baseline_ane = None
    baseline_gpu = None
    rows = []
    for n_cpu in CPU_THREAD_SWEEP:
        ane_rate, gpu_tflops = run_with_concurrency(model, n_cpu, args.duration, args.mixed)
        if baseline_ane is None:
            baseline_ane = ane_rate
            ane_str = "1.00x"
        else:
            ane_str = f"{ane_rate / baseline_ane:.2f}x"
        if args.mixed:
            if baseline_gpu is None:
                baseline_gpu = gpu_tflops
                gpu_str = "1.00x"
            else:
                gpu_str = f"{gpu_tflops / baseline_gpu:.2f}x"
            print(f"  {n_cpu:>12}  {ane_rate:>12.1f}  {gpu_tflops:>10.2f}  {ane_str:>12}  {gpu_str:>12}")
        else:
            print(f"  {n_cpu:>12}  {ane_rate:>12.1f}  {ane_str:>12}")
        rows.append({"n_cpu": n_cpu, "ane_rate": ane_rate, "gpu_tflops": gpu_tflops})

    # Interpretation
    print()
    print("Interpretation:")
    if rows[1]["ane_rate"] < rows[0]["ane_rate"]:
        drop_pct = (1 - rows[1]["ane_rate"] / rows[0]["ane_rate"]) * 100
        print(f"  A single concurrent CPU thread reduces ANE throughput by {drop_pct:.0f}%.")
        print(f"  ANE inference runs on entirely separate silicon from the CPU's TRIAD")
        print(f"  kernels. The collapse is caused by contention on the shared memory")
        print(f"  fabric, not by anything touching the Neural Engine itself.")
    if args.mixed and rows[1]["gpu_tflops"] is not None and rows[0]["gpu_tflops"] is not None:
        gpu_drop_pct = (1 - rows[1]["gpu_tflops"] / rows[0]["gpu_tflops"]) * 100
        print(f"  GPU drops {gpu_drop_pct:.0f}% under the same CPU pressure.")
        print(f"  The GPU and ANE are independent compute units, but both sit on the")
        print(f"  same memory fabric. CPU traffic affects both.")


if __name__ == "__main__":
    main()
