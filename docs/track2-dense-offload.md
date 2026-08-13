# Track 2: dense RAM-resident partial offload

Status: baseline established 2026-08-12. Successor to the closed MoE expert-locality track (`docs/moe-track-plan.md`).

## Track-2 question

How well does unmodified llama.cpp run a dense over-VRAM model from RAM with partial GPU offload on this RTX 3070 Ti (8 GiB VRAM, 32 GiB RAM), and how does that compare to the MoE 30B-A3B result from Track 1?

## Model

Qwen3-32B, official `Qwen3-32B-Q4_K_M.gguf`, 19,762,149,024 bytes (downloaded 2026-08-12 to `models/Qwen3-32B-GGUF/`). 64 layers, dense — every token computes all ~32B parameters. Chosen per `docs/model-selection.md`: same family/toolchain as the rest of the project, exceeds 8 GiB VRAM, fits 32 GiB RAM with headroom. A true dense 70B (Llama 3.1 70B Q4, ~40+ GB) exceeds current RAM and remains a documented capacity boundary, not a runnable target.

## Measured placement sweep (2026-08-12)

llama-bench b10355 (unmodified), 128 prompt / 32 generated tokens, 3 repetitions, mmap, f16 KV. Script: `experiments/dense_offload/placement_sweep.py`; raw: `results/track2-dense/placement-sweep-qwen3-32b.json`.

| GPU layers | Prompt tok/s | Generation tok/s |
|---:|---:|---:|
| 0 | 72.8 | 2.07 |
| 8 | 86.6 | 2.10 |
| 16 | 104.0 | 2.46 |
| 20 | 111.2 | 2.61 |
| 22 | 115.3 | 2.66 |
| 24 | 44.1 | 2.79 |
| 32 | 47.2 | 2.58 |

Best measured: **~2.7 tok/s generation at ngl 22–24** (flat plateau; prompt processing collapses at ngl ≥ 24 from VRAM pressure, so ngl 22 is the practical point).

## Interpretation

- Dense over-VRAM generation is CPU/RAM-bound: at 385 ms/token over a 19.8 GB working set, every token touches roughly the whole model on the CPU. Offloading ~1/3 of layers to GPU moves the number only 2.07 → 2.7 tok/s; there is no placement that escapes the sequential CPU cost of the remaining layers.
- Comparison with Track 1 at the same bench protocol: **MoE Qwen3-30B-A3B delivers ~33–35 tok/s (best placement) vs ~2.7 tok/s for dense Qwen3-32B — a ~12× gap.** On RAM-bound consumer hardware, sparse activation (3.3B active params) is worth more than an order of magnitude over dense weights of similar total size. This is now measured, not assumed.
- Consequence for the 70B+ goal: a dense 70B is doubly excluded on this machine (RAM capacity + this CPU-bound wall). Any 70B-class path must be MoE (e.g. larger-A MoE rungs) or much lower-bit — which is what Track 3 evaluates.

## Track-2 state

Baseline question answered; no follow-up experiment is currently justified inside Track 2. Possible extensions if the user wants them: KV-cache/quantization variants (Q3/IQ-quants to shrink the CPU working set), or a NUMA/thread-count sensitivity pass. Neither changes the order-of-magnitude conclusion.

## Next

Track 3: low-bit/ternary models — compatibility, quality, memory footprint, and generation rate on this hardware. Alternatively, the deferred async-prefetch MoE expert-cache iteration (see the negative PoC result in `docs/moe-track-plan.md` for exactly what a revival requires).
