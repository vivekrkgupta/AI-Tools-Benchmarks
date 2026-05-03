#!/usr/bin/env python3
"""
Demo 2: Orchestration overhead -- state-copy vs shared-memory.

Hypothesis: Two agentic-orchestration patterns process the same task volume,
but the naive pattern (passing state by value, copying per task) generates
roughly 2x the memory traffic of the optimized pattern (passing references,
operating in-place). On a memory-bandwidth-bound system, that translates to
roughly 2x worse throughput -- and the gap grows under concurrent load.

This is the software-discipline argument: the same silicon investment looks
very different depending on how the orchestration layer treats memory.

Method: Both implementations process N_TASKS through a worker that touches
all the data. Naive copies state per task; optimized reuses shared memory.
"""

import numpy as np
import threading
import time

# Per-task payload size: large enough to exceed cache and force DRAM traffic.
# 16M float32 = 64MB per state buffer.
PAYLOAD_ELEMS = 16 * 1024 * 1024
N_TASKS = 60
N_WORKERS = 4


def naive_run(payload_template, n_tasks):
    """Naive orchestration: each task copies state in, worker processes copy.

    Memory traffic per task:
      - Copy: 1 read of template + 1 write of new buffer = 2 * payload bytes
      - Work (in-place op): 1 read + 1 write = 2 * payload bytes
      Total: 4 * payload bytes per task.
    """
    bytes_moved = 0
    start = time.perf_counter()
    for _ in range(n_tasks):
        worker_state = payload_template.copy()  # full copy -- 2 streams
        worker_state *= 1.0001                  # work in-place -- 2 streams
        _ = worker_state.sum()                  # forces evaluation
        bytes_moved += worker_state.nbytes * 4
    elapsed = time.perf_counter() - start
    return bytes_moved, elapsed


def optimized_run(payload_template, n_tasks):
    """Optimized orchestration: shared in-place buffer, no copies.

    Memory traffic per task:
      - Work in-place: 1 read + 1 write = 2 * payload bytes.
      Total: 2 * payload bytes per task (half the naive cost).
    """
    shared_state = payload_template.copy()  # one allocation up front
    bytes_moved = 0
    start = time.perf_counter()
    for _ in range(n_tasks):
        shared_state *= 1.0001                  # in-place -- 2 streams
        _ = shared_state.sum()
        bytes_moved += shared_state.nbytes * 2
    elapsed = time.perf_counter() - start
    return bytes_moved, elapsed


def run_concurrent(impl_fn, payload_template, n_tasks_per_worker, n_workers):
    """Run impl_fn across n_workers concurrent threads, each processing n_tasks."""
    results = [None] * n_workers
    barrier = threading.Barrier(n_workers)

    def worker(idx):
        barrier.wait()
        b, t = impl_fn(payload_template, n_tasks_per_worker)
        results[idx] = (b, t)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start

    total_bytes = sum(r[0] for r in results)
    avg_thread_time = sum(r[1] for r in results) / n_workers
    aggregate_bw = total_bytes / wall / 1e9
    return total_bytes, avg_thread_time, wall, aggregate_bw


def main():
    print("Demo 2: Orchestration overhead -- state-copy vs shared-memory")
    print(f"Payload: {PAYLOAD_ELEMS*4//1024//1024} MB per state buffer")
    print(f"Tasks: {N_TASKS} per worker")
    print(f"Workers (concurrent runs): {N_WORKERS}")
    print()

    payload_template = np.random.rand(PAYLOAD_ELEMS).astype(np.float32)

    # --- Single-threaded baseline ---
    print("Single-threaded:")
    print(f"{'pattern':<14}  {'tasks/s':>10}  {'GB/s':>8}  {'mem traffic per task':>22}")
    print(f"{'-'*14:<14}  {'-'*10:>10}  {'-'*8:>8}  {'-'*22:>22}")

    bytes_naive, t_naive = naive_run(payload_template, N_TASKS)
    naive_tps = N_TASKS / t_naive
    naive_bw = bytes_naive / t_naive / 1e9
    naive_per_task_mb = bytes_naive / N_TASKS / 1024 / 1024
    print(f"{'naive':<14}  {naive_tps:>10.1f}  {naive_bw:>8.1f}  {naive_per_task_mb:>20.0f} MB")

    bytes_opt, t_opt = optimized_run(payload_template, N_TASKS)
    opt_tps = N_TASKS / t_opt
    opt_bw = bytes_opt / t_opt / 1e9
    opt_per_task_mb = bytes_opt / N_TASKS / 1024 / 1024
    print(f"{'optimized':<14}  {opt_tps:>10.1f}  {opt_bw:>8.1f}  {opt_per_task_mb:>20.0f} MB")

    speedup = opt_tps / naive_tps
    print(f"\nOptimized vs naive (single-threaded): {speedup:.2f}x throughput")

    # --- Concurrent: 4 workers each running the pattern ---
    print()
    print(f"Concurrent ({N_WORKERS} workers):")
    print(f"{'pattern':<14}  {'tasks/s (all)':>14}  {'agg GB/s':>10}  {'wall time':>10}")
    print(f"{'-'*14:<14}  {'-'*14:>14}  {'-'*10:>10}  {'-'*10:>10}")

    total_bytes_n, avg_t_n, wall_n, bw_n = run_concurrent(naive_run, payload_template, N_TASKS, N_WORKERS)
    naive_tps_c = (N_TASKS * N_WORKERS) / wall_n
    print(f"{'naive':<14}  {naive_tps_c:>14.1f}  {bw_n:>10.1f}  {wall_n:>9.2f}s")

    total_bytes_o, avg_t_o, wall_o, bw_o = run_concurrent(optimized_run, payload_template, N_TASKS, N_WORKERS)
    opt_tps_c = (N_TASKS * N_WORKERS) / wall_o
    print(f"{'optimized':<14}  {opt_tps_c:>14.1f}  {bw_o:>10.1f}  {wall_o:>9.2f}s")

    speedup_c = opt_tps_c / naive_tps_c
    print(f"\nOptimized vs naive (concurrent): {speedup_c:.2f}x throughput")

    print()
    print("Interpretation:")
    print(f"  Naive moves ~{naive_per_task_mb:.0f} MB per task; optimized moves ~{opt_per_task_mb:.0f} MB per task.")
    print(f"  Same agentic work, half the memory traffic, ~{speedup:.1f}x throughput.")


if __name__ == '__main__':
    main()
