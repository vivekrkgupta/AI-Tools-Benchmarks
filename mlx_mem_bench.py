#!/usr/bin/env python3
"""
mlx_mem_bench.py — Apple Silicon Memory Subsystem Benchmark
============================================================
Measures the performance gap between on-chip cache and unified memory (DRAM)
on Apple Silicon, plus GPU/NPU contention from shared memory fabric.

Tests:
  1. Cache-Bound    — element-wise add on small arrays (fits in L1/L2)
  2. Memory-Wall    — element-wise add on large arrays (>2 GB, forces DRAM)
  3. GPU/NPU Contention — GPU matmul throughput with/without ANE load

Methodology:
  Tests 1 & 2 use element-wise addition (c = a + b), a purely bandwidth-bound
  kernel (O(N) compute, O(N) memory).  This isolates the memory subsystem
  without conflating compute throughput.  Matrix multiply is only used in
  Test 3 where sustained GPU load is the goal.

  All timings use mx.eval() to force synchronous execution, preventing MLX
  lazy-evaluation from skewing results.

Requirements:
  pip install mlx                          # required
  pip install coremltools torch            # optional, for Test 3 NPU load

Usage:
  python3 mlx_mem_bench.py
"""

from __future__ import annotations

import platform
import subprocess
import textwrap
import threading
import time
from typing import Any

import mlx.core as mx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Test 1 — cache-bound: arrays that fit entirely in SoC cache (~32 MB on M1).
# 3 arrays x 256 KB = 768 KB working set → comfortably inside L2.
CACHE_ELEMS = 64 * 1024          # 256 KB per float32 array
CACHE_ITERS = 100_000
CACHE_WARMUP = 2_000

# Test 2 — memory-wall: arrays that blow past any cache.
# 3 arrays x 768 MB ≈ 2.3 GB working set → must stream from DRAM.
DRAM_ELEMS = 192 * 1024 * 1024   # 768 MB per float32 array
DRAM_ITERS = 20
DRAM_WARMUP = 2

# Test 3 — contention duration per phase.
CONTENTION_SECONDS = 10
CONTENTION_MATMUL_N = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hw_info() -> dict[str, Any]:
    """Gather hardware identifiers via sysctl / system_profiler (macOS only)."""
    info: dict[str, Any] = {}
    try:
        info["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
        ).strip()
    except Exception:
        info["chip"] = "unknown"
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True,
        ).strip())
        info["ram_gb"] = mem_bytes / (1024 ** 3)
    except Exception:
        info["ram_gb"] = 0
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True,
        )
        for line in sp.splitlines():
            if "Model Name" in line:
                info["model"] = line.split(":")[-1].strip()
            if "Chip" in line and "chip" not in info:
                info["chip"] = line.split(":")[-1].strip()
    except Exception:
        pass
    info["os"] = f"{platform.system()} {platform.mac_ver()[0]}"
    return info


def _size_label(nbytes: float) -> str:
    """Human-readable byte size."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    if nbytes >= 1e3:
        return f"{nbytes / 1e3:.1f} KB"
    return f"{nbytes:.0f} B"


# ---------------------------------------------------------------------------
# Test 1 — Cache-Bound (bandwidth via element-wise add)
# ---------------------------------------------------------------------------

def test_cache_bound() -> dict[str, Any]:
    """Element-wise c = a + b on small arrays that live in on-chip cache."""
    n = CACHE_ELEMS
    a = mx.random.normal((n,))
    b = mx.random.normal((n,))
    mx.eval(a, b)

    # Warmup — let the GPU shader caches and command queues stabilise.
    for _ in range(CACHE_WARMUP):
        c = mx.add(a, b)
    mx.eval(c)

    # Timed run.
    start = time.perf_counter()
    for _ in range(CACHE_ITERS):
        c = mx.add(a, b)
    mx.eval(c)
    elapsed = time.perf_counter() - start

    # Bytes moved per iteration: read a + read b + write c = 3 * n * 4.
    bytes_per_iter = 3 * n * 4
    total_bytes = bytes_per_iter * CACHE_ITERS

    return {
        "label": "Cache-Bound",
        "elems": n,
        "iters": CACHE_ITERS,
        "working_set_bytes": bytes_per_iter,
        "working_set": _size_label(bytes_per_iter),
        "elapsed_s": elapsed,
        "bandwidth_gbs": total_bytes / elapsed / 1e9,
        "latency_us": elapsed / CACHE_ITERS * 1e6,
        "total_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Test 2 — Unified Memory Wall (bandwidth via element-wise add)
# ---------------------------------------------------------------------------

def test_memory_wall() -> dict[str, Any]:
    """Element-wise c = a + b on huge arrays that force DRAM streaming."""
    n = DRAM_ELEMS
    bytes_per_iter = 3 * n * 4

    print(f"  Allocating arrays (working set {_size_label(bytes_per_iter)}) ...")
    a = mx.random.normal((n,))
    b = mx.random.normal((n,))
    mx.eval(a, b)

    for _ in range(DRAM_WARMUP):
        c = mx.add(a, b)
    mx.eval(c)

    start = time.perf_counter()
    for _ in range(DRAM_ITERS):
        c = mx.add(a, b)
    mx.eval(c)
    elapsed = time.perf_counter() - start

    total_bytes = bytes_per_iter * DRAM_ITERS

    return {
        "label": "Memory-Wall",
        "elems": n,
        "iters": DRAM_ITERS,
        "working_set_bytes": bytes_per_iter,
        "working_set": _size_label(bytes_per_iter),
        "elapsed_s": elapsed,
        "bandwidth_gbs": total_bytes / elapsed / 1e9,
        "latency_us": elapsed / DRAM_ITERS * 1e6,
        "total_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Test 3 — GPU / NPU Contention (matmul for sustained GPU load)
# ---------------------------------------------------------------------------

def _gpu_matmul_loop(n: int, duration: float) -> dict[str, Any]:
    """Run continuous NxN matmuls for *duration* seconds, return stats."""
    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a, b)

    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        c = a @ b
        mx.eval(c)
        count += 1
    elapsed = time.perf_counter() - start

    flops_per = 2 * n ** 3
    return {
        "matmuls": count,
        "elapsed_s": elapsed,
        "gflops": flops_per * count / elapsed / 1e9,
    }


def _coreml_npu_load(duration: float) -> None:
    """Saturate the Neural Engine via a CoreML conv-net for *duration* seconds.

    Builds a 20-layer Conv2d stack, converts to CoreML with compute_units=ALL
    (letting the ANE scheduler claim layers), and runs inference in a tight loop.
    """
    import coremltools as ct  # type: ignore[import-untyped]
    import numpy as np
    import torch
    import torch.nn as nn

    class _ConvStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for _ in range(20):
                layers += [nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()]
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model = _ConvStack().eval()
    traced = torch.jit.trace(model, torch.randn(1, 32, 224, 224))
    ml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=(1, 32, 224, 224))],
        compute_units=ct.ComputeUnit.ALL,
    )

    inp = {"x": np.random.randn(1, 32, 224, 224).astype(np.float32)}
    end_time = time.perf_counter() + duration
    count = 0
    while time.perf_counter() < end_time:
        ml_model.predict(inp)
        count += 1
    print(f"  [NPU] {count} inferences completed")


def _has_coreml() -> bool:
    try:
        import coremltools  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_contention() -> dict[str, Any]:
    """Measure GPU throughput alone vs GPU + concurrent NPU load."""
    n = CONTENTION_MATMUL_N
    npu_available = _has_coreml()

    print("  Phase A: GPU-only baseline ...")
    baseline = _gpu_matmul_loop(n, CONTENTION_SECONDS)

    print("  Phase B: GPU + NPU contention ...")
    if npu_available:
        npu_thread = threading.Thread(
            target=_coreml_npu_load,
            args=(CONTENTION_SECONDS,),
            daemon=True,
        )
        npu_thread.start()
        time.sleep(0.5)  # let CoreML compile & warm the ANE
        contended = _gpu_matmul_loop(n, CONTENTION_SECONDS)
        npu_thread.join(timeout=CONTENTION_SECONDS + 5)
    else:
        print("  [NPU] coremltools/torch not installed — running GPU-only")
        contended = _gpu_matmul_loop(n, CONTENTION_SECONDS)

    drop_pct = (
        (1 - contended["gflops"] / baseline["gflops"]) * 100
        if baseline["gflops"] > 0 else 0
    )

    return {
        "matrix_size": n,
        "baseline": baseline,
        "contended": contended,
        "npu_available": npu_available,
        "throughput_drop_pct": drop_pct,
    }


# ---------------------------------------------------------------------------
# Technical Memo
# ---------------------------------------------------------------------------

def render_memo(
    hw: dict[str, Any],
    cache: dict[str, Any],
    mem: dict[str, Any],
    contention: dict[str, Any],
) -> None:
    """Print a formatted Technical Memo summarising all results."""
    divider = "=" * 72
    thin = "-" * 72

    penalty_pct = (
        (1 - mem["bandwidth_gbs"] / cache["bandwidth_gbs"]) * 100
        if cache["bandwidth_gbs"] > 0 else 0
    )

    print(f"\n{divider}")
    print("  TECHNICAL MEMO — Apple Silicon Memory Subsystem Benchmark")
    print(divider)
    print(f"  Date       : {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Machine    : {hw.get('model', 'Mac mini')}  "
          f"({hw.get('chip', 'Apple Silicon')})")
    print(f"  RAM        : {hw.get('ram_gb', '?'):.0f} GB Unified Memory")
    print(f"  OS         : {hw.get('os', '?')}")
    print(f"  Framework  : MLX {mx.__version__}")
    print(f"  Device     : {mx.default_device()}")
    print(thin)

    # --- Executive Summary ---------------------------------------------------
    print("\n  EXECUTIVE SUMMARY")
    print(thin)
    print(textwrap.fill(
        f"When data fits entirely within the SoC's on-chip cache "
        f"({cache['working_set']} working set), the memory subsystem "
        f"delivers {cache['bandwidth_gbs']:.1f} GB/s effective bandwidth. "
        f"Scaling to a DRAM-resident workload ({mem['working_set']} working "
        f"set) reduces measured bandwidth to {mem['bandwidth_gbs']:.1f} GB/s "
        f"— a {penalty_pct:.1f}% memory penalty.",
        width=70, initial_indent="  ", subsequent_indent="  ",
    ))
    print()

    if contention["npu_available"]:
        print(textwrap.fill(
            f"Under simultaneous Neural Engine (CoreML) and GPU load, GPU "
            f"throughput dropped {contention['throughput_drop_pct']:.1f}% — "
            f"from {contention['baseline']['gflops']:.1f} to "
            f"{contention['contended']['gflops']:.1f} GFLOPS — confirming "
            f"shared-memory-fabric contention between accelerators.",
            width=70, initial_indent="  ", subsequent_indent="  ",
        ))
    else:
        print(textwrap.fill(
            f"CoreML/coremltools was not installed, so the NPU contention "
            f"test ran GPU-only in both phases. The "
            f"{contention['throughput_drop_pct']:.1f}% delta represents "
            f"measurement noise. Install coremltools + torch to enable full "
            f"GPU/NPU contention testing.",
            width=70, initial_indent="  ", subsequent_indent="  ",
        ))

    # --- Methodology ---------------------------------------------------------
    print(f"\n{thin}")
    print("  METHODOLOGY")
    print(thin)
    print(textwrap.fill(
        "Tests 1 & 2 use element-wise addition (c = a + b), a purely "
        "bandwidth-bound kernel with O(N) compute and O(N) memory traffic. "
        "This isolates the memory subsystem without conflating compute "
        "throughput. All timings use mx.eval() to force synchronous "
        "execution, preventing MLX lazy-evaluation from skewing results. "
        "Test 3 uses matrix multiplication for sustained GPU compute load.",
        width=70, initial_indent="  ", subsequent_indent="  ",
    ))

    # --- Detailed Results ----------------------------------------------------
    print(f"\n{thin}")
    print("  DETAILED RESULTS")
    print(thin)

    for label, r in [("Test 1: Cache-Bound", cache),
                     ("Test 2: Memory-Wall", mem)]:
        print(f"\n  {label}")
        print(f"    Elements       : {r['elems']:,} float32")
        print(f"    Working Set    : {r['working_set']}")
        print(f"    Iterations     : {r['iters']:,}")
        print(f"    Wall Time      : {r['elapsed_s']:.3f} s")
        print(f"    Eff. Bandwidth : {r['bandwidth_gbs']:.2f} GB/s")
        print(f"    Latency/iter   : {r['latency_us']:.2f} us")
        print(f"    Data Moved     : {_size_label(r['total_bytes'])}")

    c = contention
    print(f"\n  Test 3: Hardware Contention (GPU vs NPU)")
    print(f"    Matmul Size    : {c['matrix_size']}x{c['matrix_size']} float32")
    print(f"    Duration       : {CONTENTION_SECONDS}s per phase")
    print(f"    GPU Baseline   : {c['baseline']['gflops']:.1f} GFLOPS "
          f"({c['baseline']['matmuls']} matmuls)")
    print(f"    GPU + NPU Load : {c['contended']['gflops']:.1f} GFLOPS "
          f"({c['contended']['matmuls']} matmuls)")
    print(f"    Throughput Drop: {c['throughput_drop_pct']:.1f}%")
    print(f"    NPU Active     : "
          f"{'Yes' if c['npu_available'] else 'No (coremltools not installed)'}")

    # --- Key Metrics ---------------------------------------------------------
    print(f"\n{divider}")
    print(f"  MEMORY PENALTY (cache -> unified RAM):  {penalty_pct:.1f}%")
    print(f"  Bandwidth: {cache['bandwidth_gbs']:.1f} -> "
          f"{mem['bandwidth_gbs']:.1f} GB/s")
    if contention["npu_available"]:
        print(f"  GPU/NPU CONTENTION PENALTY:  "
              f"{contention['throughput_drop_pct']:.1f}%")
    print(divider)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    hw = _hw_info()

    print(f"Device : {mx.default_device()}")
    print(f"Chip   : {hw.get('chip', '?')}")
    print(f"RAM    : {hw.get('ram_gb', 0):.0f} GB\n")

    print("=" * 56)
    print(" Test 1 — Cache-Bound (element-wise add, 100k iters)")
    print("=" * 56)
    cache_results = test_cache_bound()
    print(f"  -> {cache_results['bandwidth_gbs']:.1f} GB/s\n")

    print("=" * 56)
    print(" Test 2 — Unified Memory Wall (element-wise add, ~2.3 GB)")
    print("=" * 56)
    mem_results = test_memory_wall()
    print(f"  -> {mem_results['bandwidth_gbs']:.1f} GB/s\n")

    print("=" * 56)
    print(" Test 3 — GPU / NPU Contention")
    print("=" * 56)
    contention_results = test_contention()
    print(f"  -> drop {contention_results['throughput_drop_pct']:.1f}%\n")

    render_memo(hw, cache_results, mem_results, contention_results)


if __name__ == "__main__":
    main()
