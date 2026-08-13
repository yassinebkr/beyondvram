# Track 3: low-bit and ternary models

Status: baseline complete 2026-08-12. Successor to the Track-2 dense-offload baseline (`docs/track2-dense-offload.md`).

## Track-3 question

Do sub-4-bit quantizations or a ternary model change the quality/speed/footprint picture for over-VRAM inference on this machine — in particular, do they redeem dense models against the measured MoE advantage (Track 1: ~33–35 tok/s, Track 2: dense 32B Q4 ~2.7 tok/s)?

## Candidates and protocol

| Candidate | Source | Size |
|---|---|---:|
| Qwen3-32B Q4_K_M | official (anchor, Track 2) | 19.76 GB |
| Qwen3-32B Q3_K_M | bartowski/Qwen_Qwen3-32B-GGUF | 15.97 GB |
| Qwen3-32B IQ2_XXS | bartowski/Qwen_Qwen3-32B-GGUF | 9.02 GB |
| BitNet b1.58 2B 4T (i2_s) | microsoft/bitnet-b1.58-2B-4T-gguf | 1.19 GB |
| Qwen3-30B-A3B Q4_K_M | official (MoE anchor, Track 1) | 18.56 GB |

Generation: llama-bench b10355, 128 prompt / 32 gen, 3 reps, mmap, f16 KV (`experiments/low_bit/bench_quants.py`, raw `results/track3-low-bit/bench-quants.json`). Quality: llama-perplexity over a fixed project-docs corpus — **relative comparison only**, same corpus for all (`experiments/low_bit/perplexity_quants.py`, raw `results/track3-low-bit/perplexity.json`).

## Measured generation rates (best config per model)

| Model | Best ngl | Prompt tok/s | Generation tok/s | vs dense Q4 |
|---|---:|---:|---:|---:|
| Qwen3-32B Q4_K_M | 22 | 115 | 2.66 | 1× |
| Qwen3-32B Q3_K_M | 28 | 153 | 3.48 | 1.3× |
| Qwen3-32B IQ2_XXS | 56 | 109 | **10.62** | **4×** |
| BitNet i2_s | — | — | failed to load | — |
| Qwen3-30B-A3B Q4_K_M (MoE) | 48, `--n-cpu-moe 33` | ~172 | **33–35** | ~13× |

IQ2_XXS detail: 2.73 (ngl 22) → 4.13 (ngl 40) → 10.62 (ngl 56) → 9.47 (ngl 64, full offload regresses — KV/context pressure). The 9 GB footprint nearly fits VRAM; at ngl 56 only ~8 layers stay on CPU, which breaks the dense CPU-bound wall documented in Track 2. Note: IQ2_XXS at ngl 40 posts the best prompt rate (314 tok/s) — prompt and generation optima differ.

## Measured perplexity (fixed docs corpus, relative only)

| Model | PPL | vs dense Q4 |
|---|---:|---:|
| Qwen3-32B Q4_K_M | 12.29 ± 0.48 | 1× |
| Qwen3-32B Q3_K_M | 12.56 ± 0.50 | +2% |
| Qwen3-32B IQ2_XXS | 15.96 ± 0.64 | +30% |
| Qwen3-30B-A3B Q4_K_M (MoE) | 16.19 ± 0.71 | +32% |

## BitNet compatibility finding

The official microsoft `ggml-model-i2_s.gguf` does not load at llama.cpp b10355 (both llama-bench and llama-perplexity fail at model load). Root cause: the file uses quantization type id 36 (`I2_S`), a format from Microsoft's BitNet llama.cpp fork; mainline ggml at this revision does not implement it (mainline ternary support is via `TQ1_0`/`TQ2_0` instead). Running BitNet on this setup would require the BitNet fork or a weight conversion to TQ2_0 — not pursued; recorded as a compatibility boundary. The 1.19 GB download is retained.

## Interpretation

- **Sub-4-bit dense narrows but does not close the gap.** Q3_K_M is a marginal speed win (+31%) at negligible quality cost (+2% PPL). IQ2_XXS is a real win — 4× generation at +30% PPL — because 9 GB nearly fits VRAM, not because low-bit CPU compute is fast (at ngl 22 it is barely faster than Q4).
- **The MoE Q4_K_M remains the frontier point on this machine**: ~3.3× faster than the best low-bit dense result at comparable corpus PPL (16.2 vs 16.0). The measured ordering is: MoE Q4 ≫ dense IQ2 (near-VRAM-fit) ≫ dense Q3/Q4 (RAM-bound).
- For the 70B+ goal: low-bit helps exactly when it makes the model (nearly) VRAM-resident. A 70B dense at ~2 bits is still ~20 GB — RAM-bound and slow. This strengthens the Track-2 conclusion: scale on this hardware comes from sparsity (MoE) or from models small enough after quantization to leave the RAM-resident regime entirely.

## Track-3 state

Baseline questions answered: compatibility (BitNet i2_s boundary recorded), quality (relative PPL), footprint, and generation rate. No further low-bit measurements are currently justified. Open follow-ups, in the project's deferred order: the async-prefetch MoE expert-cache iteration (see the negative PoC in `docs/moe-track-plan.md` for prerequisites), or BitNet via its fork/TQ2_0 conversion if ternary becomes the research focus.
