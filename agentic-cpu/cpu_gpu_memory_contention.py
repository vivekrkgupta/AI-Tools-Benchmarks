#!/usr/bin/env python3
"""
Demo 3: CPU + GPU contention under simulated agentic load on Apple Silicon.

Hypothesis: On a unified-memory SoC, a CPU "orchestration" workload generates
memory traffic that competes with the GPU's effective bandwidth -- even though
the CPU and GPU have entirely separate compute units. As CPU activity grows
(simulating concurrent agent orchestration loops), GPU throughput is dragged
down by shared-memory-fabric contention, not by lack of compute.

This extends the prior article's Mac Mini benchmark (which measured GPU vs ANE
contention) to the agentic case (CPU orchestration vs GPU inference).

Method:
  - GPU work: sustained MLX matmul (4096x4096 fp32) measured for throughput.
  - CPU work: a configurable number of concurrent threads each running TRIAD
    on a 64MB working set (simulating orchestration / KV-update activity).
  - Measure: GPU TFLOPS achieved as N_CPU goes 0 -> 1 -> 2 -> 4 -> 6 -> 8.
"""

import numpy as np
import mlx.core as mx
import threading
import time

# GPU workload: large matmul, sized to take ~50ms per op so we can stream them.
GPU_MATRIX_DIM = 4096
GPU_DTYPE = mx.float32
GPU_RUN_SECONDS = 3.0

# CPU "orchestration" workload: TRIAD on 64MB working set per thread.
CPU_ELEMS = 16 * 1024 * 1024  # 64MB float32 -- forces DRAM
CPU_RUN_SECONDS = GPU_RUN_SECONDS  # match GPU window


def cpu_orchestration_worker(stop_event, working_set):
    """Run TRIAD continuously on a per-thread working set until stop signaled."""
    a, b, c = working_set
    alpha = np.float32(2.5)
    while not stop_event.is_set():
        np.multiply(c, alpha, out=a)
        np.add(a, b, out=a)


def measure_gpu_throughput(duration_s):
    """Run sustained matmul on GPU; return TFLOPS achieved."""
    a = mx.random.normal((GPU_MATRIX_DIM, GPU_MATRIX_DIM), dtype=GPU_DTYPE)
    b = mx.random.normal((GPU_MATRIX_DIM, GPU_MATRIX_DIM), dtype=GPU_DTYPE)
    mx.eval(a, b)  # ensure inputs materialized

    # Warmup
    for _ in range(3):
        c = a @ b
        mx.eval(c)

    flops_per_matmul = 2.0 * (GPU_MATRIX_DIM ** 3)  # 2*N^3 for matmul
    n_matmuls = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        c = a @ b
        mx.eval(c)
        n_matmuls += 1
    elapsed = time.perf_counter() - start
    tflops = (n_matmuls * flops_per_matmul) / elapsed / 1e12
    return tflops, n_matmuls, elapsed


def run_with_cpu_concurrency(n_cpu_threads):
    """Measure GPU throughput while n_cpu_threads run TRIAD concurrently."""
    stop = threading.Event()
    cpu_threads = []
    working_sets = []

    for _ in range(n_cpu_threads):
        a = np.zeros(CPU_ELEMS, dtype=np.float32)
        b = np.ones(CPU_ELEMS, dtype=np.float32)
        c = np.full(CPU_ELEMS, 0.5, dtype=np.float32)
        working_sets.append((a, b, c))

    for ws in working_sets:
        t = threading.Thread(target=cpu_orchestration_worker, args=(stop, ws))
        t.start()
        cpu_threads.append(t)

    # Let CPU workers spin up
    if n_cpu_threads > 0:
        time.sleep(0.2)

    tflops, n_matmuls, elapsed = measure_gpu_throughput(GPU_RUN_SECONDS)

    stop.set()
    for t in cpu_threads:
        t.join()

    return tflops, n_matmuls


def main():
    print("Demo 3: CPU + GPU contention under simulated agentic load")
    print(f"GPU workload: {GPU_MATRIX_DIM}x{GPU_MATRIX_DIM} matmul (fp32) sustained for {GPU_RUN_SECONDS}s")
    print(f"CPU orchestration: each thread runs TRIAD on {CPU_ELEMS*4//1024//1024} MB working set")
    print()

    print(f"{'CPU threads':>12}  {'GPU TFLOPS':>12}  {'matmuls/s':>10}  {'GPU vs idle':>12}")
    print(f"{'-'*12:>12}  {'-'*12:>12}  {'-'*10:>10}  {'-'*12:>12}")

    baseline = None
    rows = []
    for n_cpu in [0, 1, 2, 4, 6, 8]:
        tflops, n_matmuls = run_with_cpu_concurrency(n_cpu)
        rate = n_matmuls / GPU_RUN_SECONDS
        rows.append({'n_cpu': n_cpu, 'tflops': tflops, 'matmuls_per_s': rate})
        if baseline is None:
            baseline = tflops
            scaling = '1.00x'
        else:
            scaling = f"{tflops/baseline:.2f}x"
        print(f"{n_cpu:>12}  {tflops:>12.2f}  {rate:>10.1f}  {scaling:>12}")

    print()
    print("Interpretation:")
    if rows[-1]['tflops'] < rows[0]['tflops']:
        drop_pct = (1 - rows[-1]['tflops'] / rows[0]['tflops']) * 100
        print(f"  GPU compute is unchanged across runs, but GPU throughput drops {drop_pct:.0f}%")
        print(f"  when {rows[-1]['n_cpu']} concurrent CPU threads contend for the shared memory fabric.")
        print(f"  CPU and GPU are separate compute units -- the contention is in memory access.")
    else:
        print("  GPU throughput unchanged by CPU concurrency in this run -- contention not visible at these")
        print("  payload sizes. Try increasing GPU_MATRIX_DIM or CPU_ELEMS to push memory pressure higher.")


if __name__ == '__main__':
    main()
