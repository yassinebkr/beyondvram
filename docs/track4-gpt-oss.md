# Track 4: gpt-oss MXFP4 — second-architecture roofline validation and the 100B-class path

Status: **baseline in progress** (20b validated; 120b gated on the 64 GiB RAM upgrade).
Date opened: 2026-08-13.

## Why this track exists

The external deep-research survey (`docs/research-report-deep-research.md`, commissioned brief in `docs/research-brief-deep-research.md`) consolidated the post-Track-3 landscape: the requested 100B-total / 20–30B-active class is empty on this envelope (capacity ≥ 132.9 GiB at Q4-class and a ~2 tok/s DDR4 ceiling for the smallest member), and the closest honest neighbor is **gpt-oss-120b** (116.8B total / 5.1B active, native-trained MXFP4 ≈ Q4-class quality, 59.02 GiB GGUF — the only ≥100B file that fits 64 GiB), with a DDR4 roofline ceiling of ~9.5–10.2 tok/s and a realistic band of 5–8 tok/s. Its experts-RAM / dense-GPU regime is exactly the regime Track 1 optimized for Qwen3-30B-A3B, so the placement knowledge transfers.

gpt-oss-20b (24 layers, 32 experts, top-4 — trace-verified id range 0–31, 21B total / 3.6B active, 12.1 GB MXFP4) is the same architecture at a size that runs today. Its role: validate MXFP4 on sm_86 and re-measure the roofline model on a second, independently architected MoE before the 120b download. All raw artifacts live in `results/gpt-oss/` + `results/gpt-oss-20b-smoke.*`.

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

## Roofline check: corrected expert bytes, effective-bandwidth model

Expert byte size measured from the GGUF header (`gguf` reader): each `blk.*.ffn_{gate,up,down}_exps.weight` is `[2880, 2880, 32]` MXFP4 = 141,004,800 B per layer per matrix → **13.22 MB per expert** (3 matrices), so gpt-oss-20b moves 24 layers × top-4 × 13.22 MB ≈ **1.27 GB of expert bytes per token**. (An earlier draft of this section used 3.3 MB/expert — a 4× arithmetic error — and concluded the CPU path was kernel-bound at 37% of the byte ceiling; the corrected bytes reverse that conclusion.)

Implied effective bandwidth at the measured points:

- All-experts-CPU (24,24): 21.00 tok/s × 1.27 GB/token = **26.6 GB/s of DDR4 read throughput** — 1.5× the B03 memcpy-payload figure (17.8 GB/s; a copy physically moves ~2× its payload) and ~46% of the 57.6 GB/s dual-channel DDR4-3600 theoretical peak.
- Best placement (24,10): 10 layers × top-4 × 13.22 MB = 0.53 GB/token from RAM × 44.79 tok/s = **23.7 GB/s** from DDR4, the remaining 14 layers' experts served from VRAM.
- For comparison, Qwen3-30B-A3B's all-experts-CPU point (16.87 tok/s × ~0.95 GB/token, Track 1) sustained ~16 GB/s: 384 small expert reads (2.5 MB) stream markedly worse than gpt-oss's 96 fat ones (13.2 MB). The earlier Track-4 claim that gpt-oss was the kernel-bound case is backwards; per-token byte volume and read shape, not dequant-kernel throughput, separate the two.

Refined model: `tg ≈ min(effective DDR4 read bytes/s ÷ CPU-resident expert bytes/token, GPU-path throughput)`, with the effective read rate measured per architecture (~26.6 GB/s for gpt-oss MXFP4 on this machine). The constant is sweep-validated: a `-t 4–12` thread sweep (`experiments/gpt_oss/thread_sweep.py`, `results/gpt-oss/thread-sweep.json`) shows (24,24) flat at 19.3–19.7 tok/s for `-t 4–8` and degrading at `-t 10–12`, and (24,10) peaking at the default `-t 6` (41.78) — the CPU expert path is saturated at the ceiling, no thread configuration exceeds it. For gpt-oss-120b (36 layers × top-4 × ~13.2 MB ≈ 1.9 GB/token — same per-expert size as the 20b, matching the survey's figure) the all-experts-RAM ceiling is ~26.6/1.9 ≈ **14 tok/s**, above the survey's 9.4–10.2 estimate (the survey anchored on the memcpy-payload rate). The realistic band stays 5–8 tok/s by neighbor measurements; the 120b run will adjudicate.

## Routing predictability: gpt-oss-20b expert traces (2026-08-13)

The Track-1 trace harness (read-only `ffn_moe_topk` hook, pinned dd1ea52 build) was re-pointed at gpt-oss-20b with no code change: `build_moe_ffn` names the topk tensor `ffn_moe_topk-<il>` for every MoE architecture, OPENAI_MOE included. Same 10 fixed prompts × 128 tokens, greedy, seed 42, `-ngl 24`; 24 layers × top-4 over a trace-verified 32-expert pool, 131,652 expert requests. Raw: `results/gpt-oss/traces/trace-*.jsonl`; analysis: `experiments/gpt_oss/analyze_predictability.py` and `experiments/moe_trace/analyze_locality.py` → `results/gpt-oss/predictability.json`, `results/gpt-oss/traces/locality-analysis.json`.

**Parity.** The trace executable is token-exact against `llama-completion` built from the same pinned tree and toolchain (`results/gpt-oss/traces/parity-*.txt`): the hook is read-only and changes nothing. The local build differs from the CI-release b10355 binary by one greedy near-tie flip mid-continuation (both continuations coherent); stock output is identical across batch geometries (`-b 128` and `-b 2048` byte-identical), so the flip is toolchain numerics on the MXFP4 kernel path (local MSVC 19.44 + CUDA 13.3 vs the CI build), not the hook and not batching. Immaterial for distributional routing statistics.

**Cheap-predictor recall** (request-weighted; over-fetch = experts fetched ÷ experts needed):

| predictor | Qwen3-30B-A3B (128 experts) | gpt-oss-20b (32 experts) |
|---|---:|---:|
| temporal-1 (repeat t−1) | 0.425 @ 1.00× | 0.450 @ 1.00× |
| cross-layer (L's selection → L+1) | 0.062 @ 1.00× | 0.130 @ 1.00× |
| FATE-union | 0.464 @ 1.93× | 0.526 @ 1.86× |
| FATE-union-2 | 0.557 @ 2.46× | 0.626 @ 2.33× |
| oracle-last8 (any function of last 8 token ids) | 0.743 @ 3.67× | 0.825 @ 3.23× |

**Per-layer LRU hit rates** (pooled, request-weighted):

| capacity | Qwen3-30B-A3B | gpt-oss-20b |
|---:|---:|---:|
| 8 | 0.422 | 0.638 |
| 16 | 0.574 | 0.865 |
| 24 | 0.673 | 0.940 |
| 32 | 0.758 | 0.954 (pool fits) |
| 48 | 0.873 | 0.954 |

Interpretation:

- gpt-oss routing is more predictable than Qwen3 on every cheap predictor, and its 32-expert pool makes small caches far more effective — the survey's model-dependence direction (YALIS quality-preserving on gpt-oss-120b, collapsing on Qwen3-30B) is confirmed on the id-history evidence.
- The runnable predictor class (token-id/history) still caps at 0.53–0.63 recall with 1.9–2.3× over-fetch — the same band as the FATE fork's real 54% achieved hit rate, which measured ~6.7× slower end-to-end on this machine (`docs/moe-track-plan.md`, FATE addendum). Over-fetch worsens the loss: fetched-but-unused experts are pure traffic on the binding resource.
- The oracle bound says no id-history function reaches the ≥90% no-over-fetch regime; only hidden-state probes report that (ETH pre-attention probes: 93–97.6% exact-match, no public code).
- **Verdict: no YALIS-style cache build.** Predictability is necessary but not sufficient — measured implementation overheads already ate a 54%-hit cache once. gpt-oss-120b has a 128-expert pool, so its cache difficulty is Qwen3-like, not 20b-like. Prediction levers stay parked unless a runnable learned-probe artifact ships.

## Deferred / next

- **64 GiB RAM upgrade (ordered)**: re-check ConfiguredClockSpeed, re-run B01–B07, then download gpt-oss-120b MXFP4 (59 GiB) and run the survey's base experiment: `-ngl 99 --n-cpu-moe 36 -fa`, default threads (`-t 6`; the 20b thread sweep measured `-t 10/12` degrading tg 15–25% on this 6-core Zen 3, `results/gpt-oss/thread-sweep.json`), batch/ubatch sweep, sanity gate on WDDM VRAM overcommit (any swapped VRAM ⇒ fix placement before trusting numbers). Predicted 5–8 tok/s by neighbor measurements; the 20b-corrected ceiling is ~14 tok/s (roofline section), so only >14 falsifies the model (publish config); <5 indicates misconfiguration (XMP, channel seating, overcommit swap).
- **Speculative decoding on gpt-oss — MEASURED (2026-08-13/14): parity, not a lever.** EAGLE-3 Q8 draft heads at the (24,10) optimum via llama-server (`experiments/gpt_oss/speculative_bench.py`; raw `results/gpt-oss/speculative-bench*.json`). The drift-canceling interleaved run (8 reps, arms alternating rep-outer, server restart per arm; `speculative-bench-interleaved.json`): **32.37 ± 2.36 draft-CPU vs 32.60 ± 0.77 no-draft = −0.7%, pure noise** — no consistent per-round direction, draft-arm stdev 3× the baseline's. Two earlier runs are superseded: a config-outer 8-rep run that measured +5.7% (a session-drift alias — inside the ~15% drift band of `docs/next-experiments.md`) and a 4-rep run that measured +45% (cold page cache depressing the no-draft arm: 12 GB lazy-paging during decode). The interleaved result lands on the survey's "+2% best" and Track-1's negative. Speculation is genuinely active (106 decode passes for 128 tokens, server logs) but each accepted draft token only shaves ~20% off a verify pass. **Support boundary**: the EAGLE-3 draft dies on GPU placement at b10355 — `-ngld 99` and `-ngld 1` both fail at draft load with `invalid vector subscript`; CPU draft is the only working mode.
- **MTP**: llama.cpp wires multi-token prediction only for qwen35/qwen35moe + Gemma4 as of b10355, not gpt-oss. No action.
- **YALIS-style cross-layer probe — MEASURED, parked (2026-08-13).** gpt-oss-20b routing predictability (section above) caps id-history predictors at 0.53–0.63 recall with ~2× over-fetch — the band that already measured negative end-to-end in the FATE PoC; the ≥90% regime requires hidden-state probes with no runnable artifact. No action.
