# Mac Mini benchmark results

Hardware: Apple M1, 8GB unified memory, 4 performance + 4 efficiency cores, ~12MB SLC. Measurements taken on macOS (Darwin 24.5.0), Python 3.9.6, NumPy 2.0.2, MLX 0.29.3.

## Demo 1: Multi-agent KV cache memory bandwidth scaling

Each "agent" runs STREAM-TRIAD (`a = b + alpha*c`) on a 192MB working set (3 x 64MB arrays, sized to vastly exceed cache). Measures aggregate and per-thread DRAM bandwidth as concurrent agent count scales.

| N agents | Aggregate GB/s | Per-thread GB/s | Per-thread vs N=1 |
|---------:|---------------:|----------------:|------------------:|
|    1     |          ~33.4 |           ~33.4 |             1.00x |
|    2     |          ~33.4 |           ~16.7 |             0.50x |
|    4     |          ~33.0 |            ~8.3 |             0.25x |
|    6     |          ~29.2 |            ~4.9 |             0.15x |
|    8     |          ~19.6 |            ~2.5 |             0.07x |

**Headline findings**:
- A single agent already saturates achievable DRAM bandwidth at ~33 GB/s.
- Adding a second agent does not increase aggregate throughput. The bandwidth pie just gets split.
- At 8 concurrent agents, aggregate bandwidth drops ~40% from the saturation point due to memory subsystem contention.
- Per-thread bandwidth at 8 agents is **7% of single-agent** throughput.

## Demo 2: Orchestration overhead — state-copy vs shared-memory

Both implementations process the same task volume. Naive copies state per task (4x payload bytes of memory traffic per task). Optimized operates in-place on shared memory (2x payload bytes per task).

### Single-threaded
| Pattern    | Tasks/sec | GB/s  | Memory traffic per task |
|:-----------|----------:|------:|------------------------:|
| Naive      |      90.6 |  24.3 |                  256 MB |
| Optimized  |     199.4 |  26.8 |                  128 MB |

**Optimized vs naive: 2.20x throughput.**

### 4 concurrent workers
| Pattern    | Tasks/sec (all) | Aggregate GB/s | Wall time |
|:-----------|----------------:|---------------:|----------:|
| Naive      |           146.4 |           39.3 |     1.64s |
| Optimized  |           246.8 |           33.1 |     0.97s |

**Optimized vs naive: 1.68x throughput.**

**Headline finding**: Same agentic workload, half the memory traffic per task, ~1.7-2.2x throughput improvement. The gap reflects exactly the difference between a framework that passes state by value and one that passes state by reference.

## Demo 3: CPU + GPU contention under simulated agentic load

GPU workload: sustained 4096x4096 fp32 matmul via MLX. CPU "orchestration" workload: N threads running TRIAD on per-thread 64MB working sets.

| CPU threads | GPU TFLOPS | matmuls/sec | GPU vs idle |
|------------:|-----------:|------------:|------------:|
|       0     |       1.69 |        12.4 |       1.00x |
|       1     |       0.86 |         6.5 |       0.51x |
|       2     |       0.84 |         6.3 |       0.49x |
|       4     |       0.81 |         6.0 |       0.48x |
|       6     |       0.80 |         6.0 |       0.47x |
|       8     |       1.02 |         7.7 |       0.60x |

**Headline finding**: A *single* concurrent CPU orchestration thread cuts GPU throughput in half — from 1.69 TFLOPS to 0.86 TFLOPS. The GPU has its own compute units; the contention is entirely in the shared memory fabric. The slight recovery at 8 CPU threads is consistent with Demo 1's finding that aggregate CPU bandwidth itself degrades under oversubscription (so the CPU generates less memory pressure on the GPU).

## What this collectively shows

Three angles on the same constraint:

1. **Memory bandwidth is the agentic ceiling**, even at one concurrent workload. Adding agents doesn't add throughput — it splits the pie.
2. **Software discipline at the orchestration layer matters as much as silicon spend.** Half the memory traffic per task, ~2x the throughput, on identical hardware.
3. **On a unified-memory SoC, compute units are independent but the memory fabric is shared.** A single CPU orchestration thread can halve GPU throughput.

These are intentionally simple isolations of single effects. Real agent runtimes are messier — but the underlying constraint is the same, and these benchmarks make it legible.
