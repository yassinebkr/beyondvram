# Baseline model selection (Q09)

Status: archived dense-streaming baseline, updated 2026-08-11 — **Qwen3-8B** was the correct 7B-class control model, but is no longer the active target because Q4_K_M runs well with llama.cpp on the RTX 3070 Ti. The successor MoE, RAM-offload, and low-bit tracks require new model-selection decisions.

## Requirements

From `initial_prompt.md`: accessible weights, suitable license, well-understood architecture, compatible tokenizer, reference implementation available, useful quantization ecosystem, and large enough to exceed 8 GiB VRAM in at least one representation. Added by the user at decision time: the family must offer a credible path to 30B and 70B+ models so early harness work carries forward.

## Candidates considered

| Family | Ladder | License | Architecture | Notes |
|---|---|---|---|---|
| Qwen3 (Apr 2025) | dense 0.6/1.7/4/8/14/32B; MoE 30B-A3B, 235B-A22B | Apache 2.0, all sizes | vanilla GQA decoder | no dense ~70B rung; the big rungs are MoE |
| Qwen3.5/3.6 (2026) | dense 27B; MoE 35B-A3B, 122B-A10B, 397B-A17B | Apache 2.0 | hybrid Gated DeltaNet/attention, multimodal | linear-attention state and vision tower are the wrong complexity for a fundamentals-first baseline |
| Llama 3.1 | 8B/70B | Meta community license, gated download | vanilla GQA decoder | true dense 70B sibling, but no ~30B rung |
| Qwen2.5 | 7/14/32/72B dense | Apache 2.0 at 7B; 72B under restrictive Qwen license | vanilla GQA decoder | complete dense ladder but licensed at the top; older generation, no MoE path |

Sources: [Qwen3 technical report](https://arxiv.org/html/2505.09388v1), [Qwen3 release blog](https://qwenlm.github.io/blog/qwen3/), [Qwen3.5 MoE in HF transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5_moe.md), [Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json).

## Decision and rationale

Qwen3-8B is Apache 2.0, has a plain, well-documented Llama-like architecture suited to learning transformer internals, a reference implementation in HF transformers, first-class llama.cpp/GGUF support (the measured-baseline runtime from the path analysis), and a ladder that matches the project ambition: 8B → 14B → 32B dense, with MoE rungs (30B-A3B, 235B-A22B) for the 70B+ goal — MoE expert streaming is already on the future-research list. Its FP16 footprint (~16.4 GB) exceeds the 8 GiB VRAM, Q8 (~8.5 GB) is borderline, and Q4 (~5 GB) fits — giving a real quantization axis at every rung.

The honest trade-off: the 70B+ rung will be either Qwen3 MoE (expert streaming, different weight-locality problem) or a cross-family dense model (e.g. Llama 3.1 70B). The streaming harness must therefore stay architecture-generic across Llama-like decoders rather than hard-coding Qwen3 specifics.

## Architecture facts (from the official config.json)

36 layers, hidden 4096, 32 attention heads, 8 KV heads (GQA), head_dim 128, SwiGLU MLP with intermediate 12288, vocab 151936, untied embeddings, BF16 checkpoint. Derived per-layer parameter count: attention ≈ 42.6M (q/o 4096×4096, k/v 4096×1024), MLP ≈ 151.0M, total ≈ 0.194B params/layer; embeddings + untied lm_head ≈ 1.24B.

## Estimated pipeline numbers (estimates, not measurements)

Layer byte sizes: BF16/FP16 ≈ 0.387 GB/layer (369 MiB, exactly 387,203,072 bytes and 4096-aligned); INT8 ≈ 0.194 GB (185 MiB); INT4 ≈ 0.118 GB (113 MiB). Against the measured medians (B01 = 2.526 GB/s NVMe, B05 = 26.06 GB/s pinned H2D), one FP16 layer costs ≈ 153 ms from NVMe vs ≈ 15 ms H2D — storage dominates roughly 10:1, exactly as the B05 result predicted. A full 36-layer FP16 pass (13.9 GB) implies ≈ 5.5 s/token if every layer is re-read from storage with no caching; INT4 (≈ 4.25 GB total) implies ≈ 1.7 s/token storage-bound or ≈ 0.16 s/token H2D-bound from a full RAM cache. These are arithmetic estimates from config sizes and measured bandwidths, stated to size the overlap experiment — not performance claims.

## Consequences for the next experiment

The synthetic overlap benchmark should use blocks of ≈ 387 MiB (one FP16 Qwen3-8B layer) over 36 blocks, plus a quantized-block variant (≈ 113 MiB) since streamed-from-NVMe inference will almost certainly run quantized. Correctness starts with known checksums on synthetic blocks; real weights (and the ~16 GB download) are only needed at the hidden-state/logit comparison stage. The 1.24 GB embedding + lm_head is a first candidate for the permanently-resident set (Q13).
