#!/usr/bin/env python3
"""
Demo 1: Multi-agent KV cache memory bandwidth scaling on Apple Silicon.

Hypothesis: As we add concurrent "agents" each maintaining its own large
persistent working set (analogous to per-agent KV cache + context state),
aggregate memory bandwidth saturates DRAM and per-thread bandwidth drops --
demonstrating that memory access, not compute, is the binding constraint for
agentic workloads.

Method: Each "agent" runs a STREAM-TRIAD kernel (a = b + alpha*c) across a
working set of three 64MB arrays = 192MB per agent. Working set is sized to
vastly exceed Apple M1's 4MB L2 / 12MB SLC, forcing DRAM access on every pass.
NumPy operations release the GIL, so threads progress concurrently.
"""

import numpy as np
import threading
import time

# Per-agent working set: 3 arrays of 64MB each = 192MB total.
# Forces DRAM traffic since 192MB >> 12MB SLC.
ELEM_PER_ARRAY = 16 * 1024 * 1024  # 16M float32 elements = 64MB
TARGET_WALL_TIME_S = 2.5

# TRIAD kernel: a = b + alpha*c
# Per element: 2 reads (b, c) + 1 write (a) = 12 bytes of DRAM traffic.
BYTES_PER_ELEM_TRAFFIC = 12


def agent_kernel(working_set, results, idx, target_time, barrier):
    """Each agent thread: run TRIAD until target_time elapsed; record bytes moved."""
    a, b, c = working_set
    alpha = np.float32(2.5)
    iterations = 0
    bytes_moved = 0

    barrier.wait()  # synchronize start across all agent threads
    start = time.perf_counter()
    while True:
        # a = alpha*c + b -- expressed as two NumPy ops that release the GIL.
        np.multiply(c, alpha, out=a)
        np.add(a, b, out=a)
        iterations += 1
        bytes_moved += a.nbytes * 3  # 1 write + 2 reads
        if time.perf_counter() - start >= target_time:
            break
    elapsed = time.perf_counter() - start
    results[idx] = (iterations, bytes_moved, elapsed)


def run_with_n_agents(n_agents):
    """Run TRIAD kernel with n_agents concurrent threads. Return aggregate stats."""
    # Each agent owns its own independent working set (KV-cache analog).
    working_sets = []
    for _ in range(n_agents):
        a = np.zeros(ELEM_PER_ARRAY, dtype=np.float32)
        b = np.ones(ELEM_PER_ARRAY, dtype=np.float32)
        c = np.full(ELEM_PER_ARRAY, 0.5, dtype=np.float32)
        working_sets.append((a, b, c))

    results = [None] * n_agents
    barrier = threading.Barrier(n_agents)
    threads = []

    for i in range(n_agents):
        t = threading.Thread(
            target=agent_kernel,
            args=(working_sets[i], results, i, TARGET_WALL_TIME_S, barrier),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    total_bytes = sum(r[1] for r in results)
    avg_elapsed = sum(r[2] for r in results) / n_agents
    aggregate_bw = total_bytes / avg_elapsed / 1e9  # GB/s
    per_thread_bw = aggregate_bw / n_agents
    total_iters = sum(r[0] for r in results)

    return {
        'n_agents': n_agents,
        'aggregate_GB_per_s': aggregate_bw,
        'per_thread_GB_per_s': per_thread_bw,
        'total_iterations': total_iters,
        'avg_elapsed_s': avg_elapsed,
    }


def main():
    print("Demo 1: Multi-agent KV cache memory bandwidth scaling")
    print(f"Per-agent working set: 3 x {ELEM_PER_ARRAY*4//1024//1024} MB = {ELEM_PER_ARRAY*4*3//1024//1024} MB")
    print(f"Target wall time per run: {TARGET_WALL_TIME_S}s")
    print(f"Kernel: TRIAD (a = b + alpha*c) - 12 bytes/element of DRAM traffic")
    print()
    print(f"{'N agents':>10}  {'aggregate GB/s':>16}  {'per-thread GB/s':>18}  {'per-thread vs N=1':>20}")
    print(f"{'-'*10:>10}  {'-'*16:>16}  {'-'*18:>18}  {'-'*20:>20}")

    baseline = None
    rows = []
    for n in [1, 2, 4, 6, 8]:
        r = run_with_n_agents(n)
        rows.append(r)
        if baseline is None:
            baseline = r['per_thread_GB_per_s']
            scaling = "1.00x"
        else:
            scaling = f"{r['per_thread_GB_per_s']/baseline:.2f}x"
        print(f"{r['n_agents']:>10}  {r['aggregate_GB_per_s']:>16.1f}  {r['per_thread_GB_per_s']:>18.1f}  {scaling:>20}")

    return rows


if __name__ == '__main__':
    main()
