#!/usr/bin/env python3
"""
Demo 2 (v2): Orchestration overhead -- state-copy vs shared-memory.

Hypothesis: Two agentic-orchestration patterns process the same task volume,
but the naive pattern (passing state by value, deep-copying per task)
generates roughly 2x the memory traffic of the optimized pattern (passing
references, operating in-place). On a memory-bandwidth-bound system, that
translates to roughly 2x worse throughput, and the gap holds under
concurrent load.

This is the software-discipline argument: the same silicon investment looks
very different depending on how the orchestration layer treats memory.

Method:
  - AgentContext represents an agent's persistent KV/state buffer, sized
    to vastly exceed cache so every operation hits DRAM.
  - "Naive" pattern: each task constructs a fresh AgentContext from a
    template (deep copy), simulating frameworks that pass full state by
    value between coordinator and worker.
  - "Optimized" pattern: one AgentContext is shared by reference across
    all tasks, simulating frameworks that pass pointers and operate in
    place.
  - Each task simulates one agent step (an in-place update of the state).

Each measurement does 1 warmup pass + median over N runs, so the reported
numbers are stable across re-runs. Parameters are CLI-configurable.

Usage:
    python3 demo2_orchestration_overhead_v2.py
    python3 demo2_orchestration_overhead_v2.py --payload-mb 128 --workers 8
"""

import argparse
import threading
import time
from statistics import median

import numpy as np


class AgentContext:
    """An agent's persistent state buffer.

    The buffer is the analog of a KV cache: large enough that any operation
    on it has to round-trip through DRAM. The naive orchestration pattern
    constructs one of these per task (paying the copy cost every time);
    the optimized pattern reuses one across all tasks.
    """

    # Memory traffic constants (in units of one full payload pass).
    # COPY: one read of template + one write of the new buffer.
    STREAMS_PER_COPY = 2
    # STEP: one in-place multiply -- one read + one write.
    STREAMS_PER_STEP = 2

    def __init__(self, kv_state: np.ndarray) -> None:
        self.kv_state = kv_state

    @classmethod
    def from_template(cls, template: np.ndarray) -> "AgentContext":
        """Deep-copy the template into a fresh buffer (the naive cost)."""
        return cls(template.copy())

    def step(self, alpha: float) -> None:
        """Run one agent step: in-place update of the full state."""
        self.kv_state *= alpha


def naive_run(template: np.ndarray, n_tasks: int) -> tuple[int, float]:
    """Naive: each task constructs a fresh AgentContext from the template.

    Memory traffic per task = (copy: 2 streams) + (step: 2 streams) = 4 streams.
    """
    streams_per_task = AgentContext.STREAMS_PER_COPY + AgentContext.STREAMS_PER_STEP
    bytes_per_stream = template.nbytes
    bytes_moved = 0

    start = time.perf_counter()
    for _ in range(n_tasks):
        agent = AgentContext.from_template(template)
        agent.step(1.0001)
        bytes_moved += bytes_per_stream * streams_per_task
    elapsed = time.perf_counter() - start
    return bytes_moved, elapsed


def optimized_run(template: np.ndarray, n_tasks: int) -> tuple[int, float]:
    """Optimized: one shared AgentContext, all tasks operate in-place.

    Memory traffic per task = (step: 2 streams) = 2 streams.
    The one-time copy on construction is excluded from per-task accounting.
    """
    streams_per_task = AgentContext.STREAMS_PER_STEP
    bytes_per_stream = template.nbytes

    agent = AgentContext.from_template(template)  # one-time setup, not measured
    bytes_moved = 0

    start = time.perf_counter()
    for _ in range(n_tasks):
        agent.step(1.0001)
        bytes_moved += bytes_per_stream * streams_per_task
    elapsed = time.perf_counter() - start
    return bytes_moved, elapsed


def run_concurrent(impl_fn, template, n_tasks_per_worker, n_workers):
    """Run impl_fn across n_workers concurrent threads."""
    results: list[tuple[int, float] | None] = [None] * n_workers
    barrier = threading.Barrier(n_workers)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = impl_fn(template, n_tasks_per_worker)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start

    total_bytes = sum(r[0] for r in results)
    return total_bytes, wall


def measure(impl_fn, template, n_tasks, n_runs=3, warmup=1):
    """Warmup + median over n_runs of impl_fn. Returns (bytes_med, elapsed_med)."""
    for _ in range(warmup):
        impl_fn(template, n_tasks)
    runs = [impl_fn(template, n_tasks) for _ in range(n_runs)]
    return median(r[0] for r in runs), median(r[1] for r in runs)


def measure_concurrent(impl_fn, template, n_tasks, n_workers, n_runs=3, warmup=1):
    """Warmup + median over n_runs of run_concurrent."""
    for _ in range(warmup):
        run_concurrent(impl_fn, template, n_tasks, n_workers)
    runs = [run_concurrent(impl_fn, template, n_tasks, n_workers) for _ in range(n_runs)]
    return median(r[0] for r in runs), median(r[1] for r in runs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--payload-mb", type=int, default=64,
                   help="Per-agent state size in MB (default 64).")
    p.add_argument("--tasks", type=int, default=60,
                   help="Tasks per worker (default 60).")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent workers in the parallel run (default 4).")
    p.add_argument("--runs", type=int, default=3,
                   help="Measurement runs after warmup (default 3).")
    args = p.parse_args()

    payload_elems = args.payload_mb * 1024 * 1024 // 4  # float32 = 4 bytes
    template = np.full(payload_elems, 0.5, dtype=np.float32)

    print("Demo 2 (v2): Orchestration overhead -- state-copy vs shared-memory")
    print(f"  Per-agent state: {args.payload_mb} MB ({payload_elems:,} float32)")
    print(f"  Tasks per worker: {args.tasks}")
    print(f"  Workers (concurrent run): {args.workers}")
    print(f"  Measurement: 1 warmup + median over {args.runs} runs")
    print()

    # --- Single-threaded baseline ---
    print("Single-threaded:")
    print(f"  {'pattern':<12}  {'tasks/s':>10}  {'GB/s':>8}  {'mem/task':>10}")
    print(f"  {'-'*12:<12}  {'-'*10:>10}  {'-'*8:>8}  {'-'*10:>10}")

    bytes_n, t_n = measure(naive_run, template, args.tasks, n_runs=args.runs)
    naive_tps = args.tasks / t_n
    naive_bw = bytes_n / t_n / 1e9
    naive_mb = bytes_n / args.tasks / 1024 / 1024
    print(f"  {'naive':<12}  {naive_tps:>10.1f}  {naive_bw:>8.1f}  {naive_mb:>7.0f} MB")

    bytes_o, t_o = measure(optimized_run, template, args.tasks, n_runs=args.runs)
    opt_tps = args.tasks / t_o
    opt_bw = bytes_o / t_o / 1e9
    opt_mb = bytes_o / args.tasks / 1024 / 1024
    print(f"  {'optimized':<12}  {opt_tps:>10.1f}  {opt_bw:>8.1f}  {opt_mb:>7.0f} MB")

    speedup_st = opt_tps / naive_tps
    print(f"\n  Optimized vs naive (single-threaded): {speedup_st:.2f}x throughput")

    # --- Concurrent: N workers each running the pattern ---
    print()
    print(f"Concurrent ({args.workers} workers):")
    print(f"  {'pattern':<12}  {'tasks/s (all)':>14}  {'agg GB/s':>10}  {'wall':>8}")
    print(f"  {'-'*12:<12}  {'-'*14:>14}  {'-'*10:>10}  {'-'*8:>8}")

    bytes_nc, wall_n = measure_concurrent(naive_run, template, args.tasks, args.workers, n_runs=args.runs)
    naive_tps_c = (args.tasks * args.workers) / wall_n
    naive_bw_c = bytes_nc / wall_n / 1e9
    print(f"  {'naive':<12}  {naive_tps_c:>14.1f}  {naive_bw_c:>10.1f}  {wall_n:>6.2f}s")

    bytes_oc, wall_o = measure_concurrent(optimized_run, template, args.tasks, args.workers, n_runs=args.runs)
    opt_tps_c = (args.tasks * args.workers) / wall_o
    opt_bw_c = bytes_oc / wall_o / 1e9
    print(f"  {'optimized':<12}  {opt_tps_c:>14.1f}  {opt_bw_c:>10.1f}  {wall_o:>6.2f}s")

    speedup_c = opt_tps_c / naive_tps_c
    print(f"\n  Optimized vs naive (concurrent): {speedup_c:.2f}x throughput")

    # --- Interpretation ---
    print()
    print("Interpretation:")
    print(f"  Naive moves {naive_mb:.0f} MB per task; optimized moves {opt_mb:.0f} MB per task.")
    print(f"  Same agentic work, half the memory traffic, {speedup_st:.2f}x throughput on identical hardware.")


if __name__ == "__main__":
    main()
