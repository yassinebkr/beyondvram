# BeyondVRAM

Status: **active research**. Tracks 0–3 closed/baselined; the FATE-fork external speedup claim measured **non-reproducible**; Track 4 (gpt-oss MXFP4, 100B-class path) open — gpt-oss-20b validated at **44.8 tok/s**; next hardware step: 64 GiB RAM (arriving in 2–3 weeks) to unlock gpt-oss-120b. Five interim experiments (bandwidth headroom, WSL2/ik_llama A/Bs as secondary setups, trace-guided mixed-precision experts, n-gram speculation) are tracked in [docs/next-experiments.md](docs/next-experiments.md).

This repository records a measurement-first investigation of running LLMs larger than VRAM on an RTX 3070 Ti (8 GiB), 32 GiB DDR4 RAM (soon 64 GiB), and NVMe storage. The overarching question: what actually determines practical inference speed on consumer hardware when the model does not fit in VRAM — and which walls are physics versus engineering?

## Results so far

| Track | Question | Measured answer |
|---|---|---|
| 0 — dense NVMe streaming (archived) | Can a custom `NVMe -> RAM -> pinned -> VRAM` pipeline make over-VRAM dense models usable? | **No.** Storage dominates decode by orders of magnitude; no overlap gain demonstrated. [Closure](docs/archive-dense-streaming.md) |
| 1 — MoE expert locality (closed) | Is expert activation local enough to cache, and does a VRAM expert cache pay off? | Locality is **strong** (median reuse 2 tokens, LRU@48/128 = 0.87) but every monetization path measured **negative**: sync LRU cache (18.7/15.0 vs 19.7 tok/s), async predictive prefetch (predictor structurally vacuous), frequency-static residency (VRAM-capped), speculative decoding (29.0 vs 29.5 tok/s). Best llama.cpp placement: **33–35 tok/s** for Qwen3-30B-A3B at `-ngl 48 --n-cpu-moe 33`. The external FATE-fork claim (33.74→64.45 tok/s) was measured **non-reproducible**: corrupt output as shipped (staging race), 4.05 vs 27.08 tok/s at 54% pool-size-invariant hit once corrected. [Full record](docs/moe-track-plan.md) |
| 2 — dense RAM offload (baseline) | How well does dense 32B run from RAM with partial GPU offload? | **~2.7 tok/s** (Qwen3-32B Q4_K_M, ngl 22) — MoE beats dense **~12×** at similar total size on this RAM-bound machine. [Full record](docs/track2-dense-offload.md) |
| 3 — low-bit/ternary (baseline) | Do sub-4-bit quants or ternary models change the picture? | Q3_K_M marginal (3.48 tok/s, +2% PPL); IQ2_XXS reaches **10.62 tok/s** because 9 GB nearly fits VRAM (+30% PPL); official BitNet i2_s GGUF is incompatible with mainline llama.cpp (fork-only quant type 36). MoE Q4 remains the quality/speed frontier. [Full record](docs/track3-low-bit.md) |
| 4 — gpt-oss MXFP4 (open) | Does the roofline hold on a second architecture, and what does the 100B-class path look like? | MXFP4 validated on sm_86. gpt-oss-20b optimum **44.79 ± 1.48 tok/s** at `-ngl 24 --n-cpu-moe 10` (plateau K=8–12); all-experts-RAM regime 21.0 tok/s, implying ~26.6 GB/s effective DDR4 read (GGUF-measured 1.27 GB expert bytes/token). EAGLE-3 draft +5.7% steady-state; routing predictability caps id-history caches below break-even. gpt-oss-120b (59 GiB) gated on the 64 GiB upgrade, predicted 5–8 tok/s. [Full record](docs/track4-gpt-oss.md) |

Reference points: llama.cpp b10355 (pinned, commit `dd1ea52`), Qwen3-8B Q4_K_M fully on GPU ≈ 85 tok/s; system characterization (NVMe 2.53 GB/s, pinned H2D 26.06 GB/s) in `docs/system-characterization.md`.

## What the measurements establish

- On RAM-bound consumer hardware, **sparse activation is worth an order of magnitude**: MoE 30B-A3B (3.3B active) reaches ~34 tok/s where dense 32B reaches 2.7 tok/s at the same total footprint class.
- At batch 1, expert routing is known only at layer time, so expert-residency decisions are always reactive; combined with ~1 GiB of free VRAM at the best placement, no hand-crafted cache or prefetch scheme measured here can pay off.
- Low-bit quants pay off exactly when they make the model (nearly) VRAM-resident, not before.
- The remaining levers for bigger models are physical (RAM capacity for larger MoE rungs, VRAM capacity for more resident experts) or research-grade (a learned cross-token expert predictor).

## Method rules

- Record, don't invent: missing dependencies produce `skipped` rows, never estimates.
- Keep every repetition; failures are data.
- Exact-output parity gates any modified runtime (the Track-1 cache prototypes were validated byte-exact before any performance claim).
- Docs distinguish measured vs hypothesized; see `docs/` for per-track reports.

## Repository map

- `docs/` — per-track research reports and closure decisions.
- `benchmarks/system/` — B01–B07 hardware characterization suite (`run_all.py`).
- `experiments/` — `overlap/` + `real_layer/` (archived Track 0), `moe_trace/` (Track 1: tracing, locality analysis, cache cost model, placement grids, prefetch simulation, speculative bench, FATE-fork repro), `dense_offload/` (Track 2), `low_bit/` (Track 3), `gpt_oss/` (Track 4).
- `tools/llama.cpp-source/` — llama.cpp pinned at `dd1ea52` with additive, env-gated experiments (moe-trace, moe-cache v1/v2; stock behavior when env vars unset). Build: `tools/build-scripts/build-trace.bat`.
- `tools/llama.cpp-b10355/` — unmodified release binaries (baselines, parity reference).
- `tools/llama-moe-cache/` — gitignored FATE-fork clone for the claim check; local patches under `results/moe-locality/fate-repro/`.
- `results/` — all raw measurements, traces, and analysis JSON; nothing extrapolated.
- `models/` — local checkpoints (not tracked in git; see `models/README.md` for the manifest).

## Reproduction

Run the existing pure tests:

```powershell
.venv\Scripts\python.exe -m pytest tests -v
```

Per-track scripts and commands are listed in `AGENTS.md` and each track's doc. Model checkpoints are large downloads from Hugging Face (see the manifest in `models/README.md`); the archived experiments are retained for reproducibility, not as an endorsed inference architecture.
