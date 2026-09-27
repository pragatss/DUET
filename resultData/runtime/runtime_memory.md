# Runtime and memory

generated 2026-09-27 12:21 by gen_runtime_memory.py; stdc: Tesla T4, torch 1.1.0, measured 2026-09-27 11:57; hrnet: Tesla T4, torch 1.1.0, measured 2026-09-27 12:20

Training = one full train iteration (fwd + all losses + bwd + SGD) at the host training batch; inference = batch 1, 1024x2048 image -> label map. Throughput pooled over interleaved rounds after a heat soak; d% = median per-round ratio to the host baseline. Peak memory = torch max_memory_allocated, isolated pass, cudnn.benchmark off.

| Method | Train | Train mem | Infer | Infer mem | Params | GMACs |
|---|---|---|---|---|---|---|
| STDC-Seg | 0.530 it/s | 10.81 GB | 11.82 img/s | 0.46 GB | 16.08M | 66.8 |
| STDC-Seg + BPM only | 0.519 it/s | 10.81 GB | 11.83 img/s | 0.46 GB | 16.08M | 66.8 |
| STDC-Seg + RSR only | 0.492 it/s | 11.39 GB | 10.62 img/s | 0.46 GB | 16.17M | 73.1 |
| STDC-Seg + Ours | 0.489 it/s | 11.39 GB | 11.28 img/s | 0.46 GB | 16.17M | 73.1 |
| HRNet-W48 | 0.468 it/s | 8.21 GB | 1.34 img/s | 1.86 GB | 65.86M | 747.5 |
| HRNet-W48 + Ours | 0.452 it/s | 8.74 GB | 1.27 img/s | 1.86 GB | 65.95M | 792.5 |
