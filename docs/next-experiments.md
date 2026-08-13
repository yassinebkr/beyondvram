# Next experiments: the post-Track-4 program

Status: **active** (opened 2026-08-13). Experiments 1 (no bandwidth headroom), 2 (WSL2: no advantage — native stays host), and 5 (n-gram: +9.9% only on repetitive content — per-workload flag) done; 3–4 queued.
Owner directive: every alternative environment below is a **secondary/testing setup** — nothing replaces the native Windows 11 + b10355 reference environment, and every comparison is measured against the same-configuration native baseline.

## Why these five

Tracks 1–4 measured their way to the walls: dense streaming is storage-bound, MoE caches/prefetch/speculation lose to their own overheads, low-bit quants trade quality the wrong way, and the corrected roofline (`docs/track4-gpt-oss.md`) puts the binding constraint on DDR4 read bandwidth and bytes-per-token. gpt-oss-120b is gated on the 64 GiB RAM upgrade (~3 weeks out). These five experiments attack what remains soft — the effective-bandwidth constant, the OS tax, the engine constant, and bytes-per-token itself — with hardware already present.

## 1 — Effective-bandwidth headroom probe (DONE 2026-08-13: no headroom)

**Question.** Is llama.cpp's CPU expert path already at the practical DDR4 read ceiling, or is there 30–50% of free bandwidth a better configuration could unlock (raising every roofline, including the 120b's)?

**Measurements.** New B07 pure-read benchmark (`benchmarks/system/benchmark_ram_read.py`): float64 `np.sum`, 1/6/12 threads. Result: practical read ceiling **~24.3–25.9 GB/s** (mt6 25.85, mt12 24.30 medians), no SMT scaling — against llama.cpp's implied **26.6 GB/s** expert-streaming rate. An initial uint8 batch measured the sum kernel (11.0/2.4 GB/s), not DRAM; superseded, kept in the raw CSV. Second arm: `experiments/gpt_oss/thread_sweep.py` — llama-bench `-t 4,6,8,10,12` at placements (24,24) and (24,10). A first sweep batch was contaminated by a foreground game and discarded unwritten; the clean re-run is `results/gpt-oss/thread-sweep.json`.

**Verdict: no headroom; ~26.6 GB/s stands as the machine constant.** (24,24) is flat at 19.3–19.7 tok/s for `-t 4/6/8` and degrades to 17.7–17.9 at `-t 10/12`; (24,10) peaks at `-t 6` (41.78) — the llama.cpp default thread count — and degrades beyond (32.2–35.2 at `-t 8–12`). No configuration beats the defaults: the CPU expert path is saturated at the practical read ceiling, and SMT contention only subtracts. "Optimize the CPU read path" ideas are dead on this box; the roofline constant needs no revision. Practical note: never run llama.cpp with `-t 12` here — it costs 15–25%.

## Cross-cutting: session drift (found 2026-08-13)

Native pure-CPU llama-bench on the same (0,0) config measured 15.57 ± 0.13 tok/s in the placement-grid session and 13.15 ± 0.56 in a later same-day recheck — a ~15% swing with no config change (a resident WSL VM holding 24 GB may have contributed to the recheck value). Consequence: any claimed effect smaller than ~15% is invisible to cross-session comparisons. All A/B designs in this program interleave arms within one session (alternating order), warm the page cache before each arm, and stop the WSL VM before native arms. The Track-4 EAGLE-3 +5.7% verdict was measured config-outer (not interleaved); the interleaved re-run collapsed it to −0.7% ± noise — a drift alias, confirming the protocol's necessity (`docs/track4-gpt-oss.md`). The experiment-1 thread sweep ran sequentially within one session; its non-monotone shape (`-t 6` > `-t 4` at (24,10)) argues against drift aliasing, and its conclusions stand.

## 2 — WSL2 OS-tax A/B (DONE 2026-08-14: no WSL2 advantage — native stays host)

**Question.** The survey cites a 15–25% same-hardware llama.cpp advantage for Linux over Windows. Is it real on this machine, for this workload?

**Setup (done).** Dedicated secondary distro `BeyondVRAM-Test` (ubuntu-base 24.04.4 rootfs, `wsl --import`, root-only) — the pre-existing docker-desktop distro untouched. `C:\Users\yassi\.wslconfig` created (none existed before): `memory=24GB` so the 12.1 GB GGUF stays page-cache-resident, `processors=12` — global WSL2 settings, delete the file to restore defaults. (Note: a unitless `memory=24576` value broke VM creation — units are mandatory.) Model copied into the ext4 vhdx (9P-mounted model reads would poison the benchmark); scripts in `tools/wsl2-ab/`.

**First signal (2026-08-13), superseded the same day by the drift discovery.** Official b10355 ubuntu-x64 CPU-only release inside WSL2 vs the grid-era native baseline measured tg 13.63 ± 0.35 vs 15.57 ± 0.13 (−12.5%) and pp 41.94 vs 157.96 (−73%). A fresh native recheck then landed at tg 13.15 ± 0.56 — the native baseline itself had drifted ~15% between sessions (see "Cross-cutting: session drift"), invalidating that comparison. The paired interleaved A/Bs below replace it.

**Paired pure-CPU A/B** (`tools/wsl2-ab/ab_paired.sh` → `results/gpt-oss/ab2-cpu00-*.json`; native official CI binary vs official ubuntu-x64 CPU release, `-t 6` both, arms alternating in one session, VM stopped before native arms, page cache warmed). tg: native 13.95/15.09/15.42 vs WSL 13.35/14.94 (plus a 14.07 repeat) — per-round deltas −1.0 to −8.8%, near parity with a slight WSL deficit, both arms drifting up together within the session (pairing was necessary). pp: WSL 34.91–43.47 vs native 147.25–157.69 — a stable 3.5–4.3× WSL deficit, thread-count-independent (`-t 12` inside WSL: pp 47.70, tg 10.67 — the native `-t 12` penalty reproduces in the VM). A forced-zen4 backend load test ruled out a CPU-variant dispatch artifact: the zen4 build refuses to load on this Zen 3 (sse42 fallback), so haswell was the correct dispatch on both OSes. The pp pathology survives pairing — real, unexplained (suspected vCPU scheduling/spin-wait interaction), not pursued further: WSL is a test bench, not a target.

**Paired hybrid CUDA A/B** (`tools/wsl2-ab/ab_hybrid.sh` → `results/gpt-oss/ab3-*.json`; native official CUDA CI binary vs in-VM gcc build of the same pinned commit dd1ea52, `-DGGML_CUDA=ON` sm_86; rounds a/b at placements (24,24) and (24,10), VM stop + cache warm between arms). In-VM CUDA verified live: `ggml_cuda_init` finds the 3070 Ti, and hybrid tg lands ~2.5× above pure-CPU. Generation: **WSL slower in all four round pairs** — (24,24) 21.21 vs 22.96 (−7.6%), (24,10) 37.94 vs 41.72 (−9.1%). Prompt processing inverts the sign: WSL +35% at (24,24) (196.7 vs 145.7) and a stable 310–347 at (24,10) where native went bimodal across rounds (149.66 vs 293.72) — consistent with lower Linux driver overhead vs WDDM launch/sync on the GPU-fed path.

**Verdict: closed — native Windows stays the benchmarking and reference host.** The cited 15–25% Linux advantage does not reproduce on this box for this workload; the sign flips per phase (Linux wins GPU-fed pp, loses memory-bound tg by 7.6–9.1%, loses CPU-only pp 3.5–4.3×). The 120b base experiment is tg-dominated with all experts on CPU — precisely the phase where WSL2 pays its hypervisor tax — so the plan is unchanged. Caveats: arms compare the stacks as shipped (MSVC CI binary vs gcc source build — compiler not isolated from OS); native (24,24) tg this session (22.96) ran ~9% above the grid-era 21.00 ± 1.09 — pairing within one session, not cross-session comparison, is what makes the deltas readable at all.

## 3 — ik_llama.cpp engine A/B (QUEUED, secondary setup)

**Question.** Does the performance-focused fork beat mainline b10355 on this CPU-heavy MoE regime? Its CPU/quant paths are frequently reported faster, and its per-tensor quant control is what experiment 4 needs.

**Protocol.** Build ik_llama.cpp under `tools/` (the pinned b10355 tree stays untouched), same model and placements, llama-bench protocol identical to the native grid. A/B only; no migration of the reference setup.

**Decision rule.** A sustained ≥10% generation advantage at equal placement earns a follow-up full grid; otherwise recorded as parity/negative and closed.

## 4 — Trace-guided mixed-precision experts (QUEUED)

**Question.** Expert activation is heavily skewed (measured traces: top experts carry a disproportionate share of requests). Does a per-expert bit allocation — hot experts at Q4_K/Q6_K, cold tail at IQ2-class — cut bytes-per-token ~25–35% while staying within the Q4-class quality envelope? In a bandwidth-bound regime, bytes/token is the direct speed lever.

**Protocol.** Per-layer expert frequencies from the Track-4 traces → hot/cold split → produce two full GGUFs (Q4_K_M and IQ2-class) with stock `llama-quantize` → transplant cold-expert tensors at GGUF level (per-tensor types are legal in llama.cpp) → quality gate: Track-3 perplexity harness on the fixed corpus, vs the Q4_K_M baseline → speed: placement-grid protocol.

**Decision rule.** Ship the variant only if perplexity stays within a few % of the Q4_K_M baseline AND generation improves ≥15% at the best placement. Either arm failing closes the idea at this model size. The technique, if validated, transfers directly to gpt-oss-120b.

## 5 — n-gram self-speculation (DONE 2026-08-14: real but narrow — a per-workload flag)

**Question.** EAGLE-3 measured parity (−0.7% interleaved). N-gram/lookup self-speculation needs no resident draft model — does it beat EAGLE-3 on repetitive content?

**Support check (done 2026-08-13).** b10355 `llama-server --help` exposes `--spec-type` values `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` with per-mode lookup/draft/min-hits tunables. Runnable via the Track-4 `speculative_bench.py` harness, `--ngram` mode.

**Measurement (interleaved, 8 reps, first discarded; `results/gpt-oss/speculative-bench-ngram.json`).** Same (24,10) server placement, greedy, repetitive code-shaped prompt (twenty identical-pattern dataclasses). `ngram-simple`: **31.87 ± 3.24 vs 29.00 ± 3.25 no-spec = +9.9%**, 6 of 7 rounds positive, mechanism confirmed in server logs — draft acceptance 0.598 (49/82 tokens, mean drafted length 17.33). `ngram-mod`: 28.89 ± 1.32 = parity — acceptance 0.000, it never fired on this prompt, and the miss cost is ~0.4%.

**Verdict: real but narrow — a per-workload flag, not a roofline mover.** +9.9% at 60% acceptance is genuine free speed, but only on highly repetitive/structured output (code, tables, templates); verification is exact so it is lossless, and a non-firing mode costs ~0.4%. The mechanism (literal n-gram lookup) implies the gain shrinks on open-ended prose — not separately measured, and it does not touch the bandwidth wall (accepted tokens still re-read the same experts in the verify pass). Recorded as a usable per-workload option (`--spec-type ngram-simple`), carried into the 120b protocol as the code-workload variant; no further bench rounds on the 20b.

Experiments 2 and 3 are independent secondary setups. Experiment 4 depends on 3 only if stock `llama-quantize` cannot express the per-expert split (ik's per-tensor control is the fallback).
