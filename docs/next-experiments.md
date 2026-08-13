# Next experiments: the post-Track-4 program

Status: **active** (opened 2026-08-13). Experiment 1 done (no bandwidth headroom); 2–5 queued.
Owner directive: every alternative environment below is a **secondary/testing setup** — nothing replaces the native Windows 11 + b10355 reference environment, and every comparison is measured against the same-configuration native baseline.

## Why these five

Tracks 1–4 measured their way to the walls: dense streaming is storage-bound, MoE caches/prefetch/speculation lose to their own overheads, low-bit quants trade quality the wrong way, and the corrected roofline (`docs/track4-gpt-oss.md`) puts the binding constraint on DDR4 read bandwidth and bytes-per-token. gpt-oss-120b is gated on the 64 GiB RAM upgrade (~3 weeks out). These five experiments attack what remains soft — the effective-bandwidth constant, the OS tax, the engine constant, and bytes-per-token itself — with hardware already present.

## 1 — Effective-bandwidth headroom probe (DONE 2026-08-13: no headroom)

**Question.** Is llama.cpp's CPU expert path already at the practical DDR4 read ceiling, or is there 30–50% of free bandwidth a better configuration could unlock (raising every roofline, including the 120b's)?

**Measurements.** New B07 pure-read benchmark (`benchmarks/system/benchmark_ram_read.py`): float64 `np.sum`, 1/6/12 threads. Result: practical read ceiling **~24.3–25.9 GB/s** (mt6 25.85, mt12 24.30 medians), no SMT scaling — against llama.cpp's implied **26.6 GB/s** expert-streaming rate. An initial uint8 batch measured the sum kernel (11.0/2.4 GB/s), not DRAM; superseded, kept in the raw CSV. Second arm: `experiments/gpt_oss/thread_sweep.py` — llama-bench `-t 4,6,8,10,12` at placements (24,24) and (24,10). A first sweep batch was contaminated by a foreground game and discarded unwritten; the clean re-run is `results/gpt-oss/thread-sweep.json`.

**Verdict: no headroom; ~26.6 GB/s stands as the machine constant.** (24,24) is flat at 19.3–19.7 tok/s for `-t 4/6/8` and degrades to 17.7–17.9 at `-t 10/12`; (24,10) peaks at `-t 6` (41.78) — the llama.cpp default thread count — and degrades beyond (32.2–35.2 at `-t 8–12`). No configuration beats the defaults: the CPU expert path is saturated at the practical read ceiling, and SMT contention only subtracts. "Optimize the CPU read path" ideas are dead on this box; the roofline constant needs no revision. Practical note: never run llama.cpp with `-t 12` here — it costs 15–25%.

## 2 — WSL2 OS-tax A/B (QUEUED, secondary setup)

**Question.** The survey cites a 15–25% same-hardware llama.cpp advantage for Linux over Windows. Is it real on this machine, for this workload?

**Protocol.** Identical model (gpt-oss-20b MXFP4), identical placements (24,24) and (24,10), identical llama-bench protocol, 3+ repetitions, native Windows vs WSL2 CUDA (same llama.cpp revision built inside WSL). Record vmmem dynamics and clock state alongside; WSL2 memory management can distort RAM-bound workloads, which is itself a finding.

**Decision rule.** A sustained ≥10% advantage makes WSL2 the benchmarking host for the 120b base experiment; parity or worse closes it. The native Windows setup remains the reference either way.

## 3 — ik_llama.cpp engine A/B (QUEUED, secondary setup)

**Question.** Does the performance-focused fork beat mainline b10355 on this CPU-heavy MoE regime? Its CPU/quant paths are frequently reported faster, and its per-tensor quant control is what experiment 4 needs.

**Protocol.** Build ik_llama.cpp under `tools/` (the pinned b10355 tree stays untouched), same model and placements, llama-bench protocol identical to the native grid. A/B only; no migration of the reference setup.

**Decision rule.** A sustained ≥10% generation advantage at equal placement earns a follow-up full grid; otherwise recorded as parity/negative and closed.

## 4 — Trace-guided mixed-precision experts (QUEUED)

**Question.** Expert activation is heavily skewed (measured traces: top experts carry a disproportionate share of requests). Does a per-expert bit allocation — hot experts at Q4_K/Q6_K, cold tail at IQ2-class — cut bytes-per-token ~25–35% while staying within the Q4-class quality envelope? In a bandwidth-bound regime, bytes/token is the direct speed lever.

**Protocol.** Per-layer expert frequencies from the Track-4 traces → hot/cold split → produce two full GGUFs (Q4_K_M and IQ2-class) with stock `llama-quantize` → transplant cold-expert tensors at GGUF level (per-tensor types are legal in llama.cpp) → quality gate: Track-3 perplexity harness on the fixed corpus, vs the Q4_K_M baseline → speed: placement-grid protocol.

**Decision rule.** Ship the variant only if perplexity stays within a few % of the Q4_K_M baseline AND generation improves ≥15% at the best placement. Either arm failing closes the idea at this model size. The technique, if validated, transfers directly to gpt-oss-120b.

## 5 — n-gram self-speculation support check (QUEUED; support CONFIRMED 2026-08-13)

**Question.** EAGLE-3 measured +5.7% steady state (weak). N-gram/lookup self-speculation needs no resident draft model — does it beat EAGLE-3 on repetitive content?

**Support check (done).** b10355 `llama-server --help` exposes `--spec-type` values `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` with per-mode lookup/draft/min-hits tunables. Runnable with the Track-4 `speculative_bench.py` harness extended by a spec-type parameter.

**Protocol.** Same server placement as the EAGLE-3 bench (24,10), greedy, repetitive-content prompts (code, structured text), modes `ngram-simple` and `ngram-mod` first, 8-rep steady-state protocol (first rep discarded — the cold-cache artifact from the EAGLE-3 bench).

**Decision rule.** <EAGLE-3's +5.7% closes it; >10% on realistic workloads earns a variant bench at the (24,10) optimum.

## Sequencing

1 runs now (pure measurement). 2 and 3 are independent secondary setups; 4 depends on 3 only if stock `llama-quantize` proves unable to express the split (ik's per-tensor control is the fallback); 5 folds into whichever bench runs next. Each lands with raw results, a doc verdict, and a commit — negative results are recorded, not dropped.
