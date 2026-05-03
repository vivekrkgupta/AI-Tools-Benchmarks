# AI Tools and Benchmarks

Source code for benchmarks discussed in articles on AI infrastructure and silicon.

## Contents

### [`agentic-cpu/`](./agentic-cpu)

Benchmarks for *"Agentic AI Is a CPU Story."* The article argues that CPU memory bandwidth and on-package interconnect, not raw compute, are the binding constraints for agentic AI workloads at both data-center scale and on-device. The two benchmarks cited in the article body:

- [`orchestration_overhead.py`](./agentic-cpu/orchestration_overhead.py) — state-copy vs shared-memory orchestration patterns. Same agentic work, half the memory traffic, ~2x throughput on identical hardware.
- [`memory_fabric_contention.py`](./agentic-cpu/memory_fabric_contention.py) — CPU + GPU + ANE concurrent on a single Mac Mini. Just running GPU and ANE together, with the CPU idle, costs the GPU ~25% of its solo throughput. Adding a single CPU thread drops it another 30%.

Plus three supporting benchmarks (multi-agent bandwidth scaling, GPU-only contention, and an alternative orchestration-overhead implementation). See the [agentic-cpu README](./agentic-cpu/README.md) for details.

### [`mlx_mem_bench.py`](./mlx_mem_bench.py)

Mac Mini memory bandwidth benchmark from the prior article *"The Hardware Reality of AI Inference."* Measures in-cache vs out-of-cache memory bandwidth on Apple Silicon and demonstrates GPU-vs-ANE contention on the unified memory architecture.
