# Track 4: gpt-oss MXFP4 — second-architecture roofline validation and the 100B-class path

Status: **baseline in progress** (20b validated; 120b gated on the 64 GiB RAM upgrade).
Date opened: 2026-08-13.

## Why this track exists

The external deep-research survey (`docs/research-report-deep-research.md`, commissioned brief in `docs/research-brief-deep-research.md`) consolidated the post-Track-3 landscape: the requested 100B-total / 20–30B-active class is empty on this envelope (capacity ≥ 132.9 GiB at Q4-class and a ~2 tok/s DDR4 ceiling for the smallest member), and the closest honest neighbor is **gpt-oss-120b** (116.8B total / 5.1B active, native-trained MXFP4 ≈ Q4-class quality, 59.02 GiB GGUF — the only ≥100B file that fits 64 GiB), with a DDR4 roofline ceiling of ~9.5–10.2 tok/s and a realistic band of 5–8 tok/s. Its experts-RAM / dense-GPU regime is exactly the regime Track 1 optimized for Qwen3-30B-A3B, so the placement knowledge transfers.

gpt-oss-20b (24 layers, 128 experts, top-4, 21B total / 3.6B active, 12.1 GB MXFP4) is the same architecture at a size that runs today. Its role: validate MXFP4 on sm_86 and re-measure the roofline model on a second, independently architected MoE before the 120b download. All raw artifacts live in `results/gpt-oss/` + `results/gpt-oss-20b-smoke.*`.

## Smoke: MXFP4 works on sm_86 (2026-08-13)

Stock b10355 `llama-completion`, default `--fit`, 47 greedy tokens: **29.52 tok/s**, coherent output (`results/gpt-oss-20b-smoke.out.txt`). No custom kernels needed; the packed dequant/dot CUDA path in b10355 handles MXFP4 on compute capability 8.6.

## Placement grid (b10355 llama-bench, 128 pp / 32 tg, mmap, f16 KV)

`experiments/gpt_oss/placement_grid.py`; raw: `results/gpt-oss/placement-grid.json`, `placement-grid-refine.json`.

| ngl | n-cpu-moe | generation tok/s | note |
|---:|---:|---:|---|
| 24 | 24 | 21.00 ± 1.09 | all attention GPU, all experts CPU — the 120b-analog regime |
| 24 | 16 | 33.83 ± 2.56 | |
| 24 | 12 | 42.62 ± 2.22 | first-pass peak (refined below) |
| 24 | 8 | 37.56 ± 0.40 | |
| 24 | 4 | 27.55 ± 0.15 | |
| 24 | 0 | 22.79 ± 0.08 | no OOM: `--fit` re-placed gracefully (expected boundary record did not trigger) |
| 0 | 0 | 15.57 ± 0.13 | pure CPU floor |

Refinement (5 reps): K=14 → 32.73 ± 2.79, K=12 → 38.45 ± 2.39, K=10 → **44.79 ± 1.48**. Between-session variance for identical K is ~4 tok/s (K=12: 42.62 vs 38.45), so treat K=8–12 as a plateau, not a sharp peak.

**Measured optimum: `-ngl 24 --n-cpu-moe 10` ≈ 44.8 generation tok/s** — attention of all 24 layers plus a suffix of 14 layers' experts in VRAM, first 10 layers' experts from RAM. Same interior-optimum shape as Qwen3 (both ends worse), second architecture, independent confirmation that `--n-cpu-moe` decoupling is the controlling knob.

## Roofline check: the ceiling is min(bandwidth, CPU compute), not bandwidth alone

Per-token expert traffic for gpt-oss-20b ≈ 24 layers × top-4 × ~3.3 MB MXFP4/expert ≈ **0.31 GB/token** → DDR4 payload ceiling 17.8/0.31 ≈ **57 tok/s**. The all-experts-CPU point measured 21.00 tok/s = 37% of that ceiling, whereas Qwen3-30B-A3B's equivalent point (16.87 tok/s) sat at ~90% of its own byte ceiling (18.7 tok/s). The Qwen3 CPU expert path is bandwidth-saturated; the gpt-oss path is not — on 6 Zen 3 cores the MXFP4 CPU dequant/dot throughput binds first.

Refined model: `tg ≈ min(DDR4 bytes/s ÷ expert bytes/token, CPU expert-kernel tokens/s)`. For gpt-oss-120b on this machine the per-token expert traffic grows ~6× (to ~1.9 GB) while CPU expert compute grows comparably, so both limits converge on the predicted **5–8 tok/s** — the grid here is consistent with the survey's band, and the 120b experiment will test it directly.

## Deferred / next

- **64 GiB RAM upgrade (ordered)**: re-check ConfiguredClockSpeed, re-run B01–B06, then download gpt-oss-120b MXFP4 (59 GiB) and run the survey's base experiment: `-ngl 99 --n-cpu-moe 36 -fa`, `-t 12`, batch/ubatch sweep, sanity gate on WDDM VRAM overcommit (any swapped VRAM ⇒ fix placement before trusting numbers). Predicted 5–8 tok/s; >10 falsifies the roofline (publish config); <5 indicates misconfiguration (XMP, channel seating, overcommit swap).
- **Speculative decoding on gpt-oss**: EAGLE-3 draft heads for gpt-oss-20b are already downloaded (`models/gpt-oss-20b-GGUF/`); the survey records hybrid-spec decode as weak-to-negative on 8 GB + CPU-experts rigs (−19% to +2%), and Track-1 measured draft speculation negative on Qwen3. Test only after the placement baseline is documented; expectations low.
- **MTP**: llama.cpp wires multi-token prediction only for qwen35/qwen35moe + Gemma4 as of b10355, not gpt-oss. No action.
- **YALIS-style cross-layer probe**: quality-preserving on gpt-oss-120b specifically (survey, single paper). A possible later code-level iteration if the 120b baseline lands in band and predictor accuracy on gpt-oss routing justifies it — measure routing predictability on the 20b first (cheap).
