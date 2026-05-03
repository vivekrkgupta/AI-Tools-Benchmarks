# Article-program benchmarks

Mac Mini benchmarks that substantiate the thesis of the article *"Agentic AI Is a CPU Story"*: **CPU memory bandwidth and the on-package interconnect, not raw compute, are the binding constraints for agentic AI workloads — both in the data center and on-device.**

Hardware used:
- Apple M1 (4 performance + 4 efficiency cores, 8 GB unified memory)
- L1d 64 KB, L2 4 MB per cluster, ~12 MB system-level cache
- Theoretical DRAM bandwidth: ~68 GB/s

## Benchmarks cited in the article

### `orchestration_overhead.py` — section 4 ("The software side matters as much as the silicon")

Compares two orchestration patterns processing identical task volume:
- **Naive**: each task triggers a full state copy (frameworks that pass payloads by value)
- **Optimized**: shared-memory in-place operation (frameworks that pass references)

Same agentic work, half the memory traffic, ~2x throughput on identical hardware. Substantiates the claim that software discipline at the orchestration layer determines whether high-core-count silicon investment pays off.

A more developed version with `argparse`, warmup + median, and an explicit `AgentContext` class is in `orchestration_overhead_v2.py`. The v1 numbers match the article body; v2 is structurally cleaner and produces a sharper gap (~3.3x single-threaded) that better reflects the actual orchestration-overhead delta once an unnecessary `.sum()` materialization is removed.

### `memory_fabric_contention.py` — section 5 ("At the edge, the same constraint shows up earlier and harder")

Approximates a real agentic-edge workload running concurrently on a single SoC:
- GPU: sustained matmul via MLX (analog of LLM token generation)
- ANE: CNN inference via Core ML (analog of perception / multimodal heads)
- CPU: configurable streaming-kernel threads on 64 MB working sets (analog of an orchestration loop)

Run with `--mixed` for the canonical edge receipt cited in the article. Headline finding: simply running GPU and ANE concurrently already costs the GPU ~25% of its solo throughput, because they compete for the shared memory fabric even though their compute pipelines are entirely separate. Adding CPU orchestration traffic compounds the contention.

## Supporting benchmarks (not cited in the article)

### `cpu_gpu_memory_contention.py`

Simpler GPU-only variant of `memory_fabric_contention.py`. Runs sustained MLX matmul on the GPU, sweeps N concurrent CPU threads doing TRIAD on 64 MB working sets, measures GPU TFLOPS at each step. Headline result: a single CPU thread cuts GPU throughput nearly in half on the M1 Mac Mini.

### `multi_agent_memory_bandwidth.py`

Sweeps N concurrent simulated "agents" (1 → 8), each holding a 192 MB working set and running a STREAM-TRIAD kernel against it. Demonstrates that DRAM bandwidth is the ceiling: a single agent saturates achievable bandwidth at ~33 GB/s, and adding agents splits the same pipe rather than scaling throughput.

This was a candidate edge receipt during drafting but didn't make the final article cut. Available for a follow-up post on per-agent memory pressure.

## Other code in this directory

### `mlx_mem_bench.py`

The original Mac Mini MLX/CoreML benchmark from the prior article *"The Hardware Reality of AI Inference"*. Measures memory bandwidth in-cache vs out-of-cache and demonstrates GPU-vs-ANE contention on the unified memory architecture.

## Running

```bash
python3 orchestration_overhead.py
python3 memory_fabric_contention.py --mixed
python3 cpu_gpu_memory_contention.py
python3 multi_agent_memory_bandwidth.py
```

`memory_fabric_contention.py` requires `coremltools` and `torch`, plus a one-time CNN compilation step that takes ~10 seconds on first run (cached at `~/.cache/article-programs/heavy_cnn.mlpackage`).

## Notes

- Apple M1 has a unified memory architecture; absolute bandwidth and TFLOPS numbers will differ on M1 Pro/Max/Ultra, but the *qualitative* shape of the contention curves should hold.
- These benchmarks isolate one effect each rather than approximating a full agent runtime. They make the constraint legible.
