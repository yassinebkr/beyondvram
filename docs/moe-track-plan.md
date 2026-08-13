# MoE expert-locality track

Status: active research plan, 2026-08-11. This is the successor to the archived dense NVMe-streaming track.

## Why MoE

Qwen3-30B-A3B has 30.5B total parameters but activates 3.3B per token through 8 of 128 experts across 48 layers. Its official Q4_K_M GGUF is 18.6 GB: too large for 8 GiB VRAM, but small enough for the present 32 GiB RAM if other memory pressure is controlled. This makes it a materially different locality problem from Qwen3-8B dense inference: total model size exceeds VRAM, while the active expert subset is limited.

Sources: [official Qwen GGUF model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF), [llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix). The feature matrix reports CUDA MoE support, but that is compatibility evidence—not a performance guarantee on this machine.

## Track-1 question

How well does unmodified llama.cpp run Qwen3-30B-A3B Q4_K_M from RAM with partial GPU offload on this RTX 3070 Ti, and is expert activation locality stable enough to justify a later cache experiment?

## Baseline protocol

1. Use the official `Qwen3-30B-A3B-Q4_K_M.gguf` and pin the llama.cpp b10355 binary already recorded in this repository.
2. Record model load outcome, host working set, free VRAM, prompt processing, and generation for CPU-only and several GPU-layer counts that fit without out-of-memory failure.
3. Keep context, batch size, KV-cache type, power state, and background workload fixed within a comparison set.
4. Retain every successful run and failures; do not treat a model-load failure as zero performance.

## Measured placement baseline

On 2026-08-11, the official 18,556,685,824-byte Q4_K_M GGUF loaded from the current 32 GiB RAM configuration. A llama.cpp b10355 placement sweep used three repetitions each, 128 prompt tokens, 32 generated tokens, mmap loading, and f16 KV cache. The JSON outputs are `results/moe-qwen3-30b-a3b-placement-sweep.json` and `results/moe-qwen3-30b-a3b-placement-refine.json`.

| GPU layers | Prompt tok/s mean ± SD | Generation tok/s mean ± SD |
|---:|---:|---:|
| 0 | 105.92 ± 18.34 | 15.09 ± 1.19 |
| 6 | 105.56 ± 32.13 | 17.31 ± 2.13 |
| 12 | 129.15 ± 20.15 | 21.59 ± 1.50 |
| 18 | 145.19 ± 21.45 | **25.15 ± 2.40** |
| 24 | 48.23 ± 9.77 | 18.61 ± 0.49 |
| 30 | 46.22 ± 19.18 | 17.61 ± 0.26 |

The placement relation is non-monotonic: 18 GPU layers is the best tested generation point, while 24 and 30 layers regress. This is measured evidence that partial offload can make a 30.5B-total, 3.3B-active MoE model usable on the current machine, but it does not identify why the optimum occurs or prove that expert locality is responsible. The next Track-1 action is therefore observational expert-routing tracing, not an eviction-policy implementation.

An additional three-repetition ablation at 18 GPU layers forced all 48 MoE layers to CPU with `--n-cpu-moe 48`. Generation fell to 16.87 ± 0.80 tok/s (`results/moe-qwen3-30b-a3b-experts-cpu.json`). This shows that placing MoE work on the GPU is materially beneficial in the tested configuration; it does not reveal individual expert IDs or establish that a small expert cache would have a high hit rate.

## Expert-trace implementation finding

A source inspection found that llama.cpp names the selected expert-index tensor `ffn_moe_topk` in `llm_graph_context::build_moe_ffn`. Its existing scheduler evaluation-callback API can request a tensor after graph evaluation, so a small trace executable can capture actual selected IDs without changing routing or model math. The release binary used for the placement benchmark has no trace flag and does not ship that example. The required native toolchain is now installed and verified: CMake 4.4.2, MSVC 19.44, and CUDA Toolkit 13.3 (`nvcc` 13.3.73). The next session must first pin the source checkout to the b10355 baseline commit `dd1ea5243`, then add only a callback that copies `ffn_moe_topk` I32 tensors with layer/token coordinates to CSV or JSONL. It must not alter scheduling, weights, routing, or expert placement.

## Trace build and parity validation (2026-08-11)

The trace executable exists and is validated:

- `tools/llama.cpp-source` is pinned at `dd1ea52` (tag b10355). The only source changes are additive: `examples/moe-trace/moe-trace.cpp` (+ its `CMakeLists.txt`) and one `add_subdirectory(moe-trace)` line in `examples/CMakeLists.txt`. The callback copies only `ffn_moe_topk-<layer>` I32 tensors ([8, n_tokens]) after evaluation and writes JSONL records `{"pos", "layer", "experts"}` to the path in `MOE_TRACE_OUT`. Routing, weights, scheduling, and placement are untouched.
- CUDA build: Ninja + vcvars64 + CUDA 13.3 via `tools/build-scripts/configure-trace-build.bat` and `build-trace.bat` (the VS generator cannot resolve CudaToolkitDir; Git Bash needs `MSYS2_ARG_CONV_EXCL="*"` for `cmd /c`). Runtime DLLs `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` are copied next to `build/bin/llama-moe-trace.exe`.
- One trace-specific behavior to remember: the last MoE layer only routes tokens whose logits are requested, so prompt-phase positions produce no layer-47 records. The analyzer is position-aware and tolerates these gaps.
- Parity (`experiments/moe_trace/parity_check.py`): on the fixed prompt "The capital of France is", greedy, seed 42, `-ngl 18`, 32 tokens, the trace build's continuation is byte-identical to the unmodified b10355 `llama-completion.exe -no-cnv` output ("Paris. The capital of Italy is Rome. ..."). Raw outputs: `results/moe-locality/parity-*.txt`. Note: b10355 `llama-cli` is unusable for scripted comparison — it stays in an interactive input loop even with `-no-cnv` and stdin closed; use `llama-completion -no-cnv` instead. `--no-warmup` is not accepted by the trace executable.

## Measured expert-routing locality (2026-08-11)

Ten fixed diverse prompts (prose, code, math, chat, French, reasoning), 128 greedy generated tokens each at `-ngl 18`, captured by `experiments/moe_trace/run_traces.py` into `results/moe-locality/trace-*.jsonl` (~6.6k–8k records per file; generated text kept alongside as `gen-*.txt`). Analysis by `experiments/moe_trace/analyze_locality.py`; full aggregates in `results/moe-locality/locality-analysis.json`. Random-overlap reference: two independent 8-of-128 expert sets share 6.25% on average.

- **Consecutive-token overlap**: mean |E(t) ∩ E(t+1)| / 8 per layer, pooled per file, is 0.393–0.472 across the ten traces. Overlap decays with distance but stays far above random: ~0.45 at d=1, ~0.36 at d=2, ~0.30 at d=8–16.
- **Reuse distance**: median time between activations of the same expert within a layer is 2 tokens; median per-layer p90 is 14 tokens.
- **Frequency skew**: per layer, 78–106 of 128 experts are used over ~140 tokens; top-1 expert share 3–9%, top-8 share 20–38%, entropy 5.5–6.3 bits (uniform would be 7). Later layers are more skewed than early layers.
- **Hypothetical per-layer LRU expert-cache hit rates** (request-weighted, pooled over all layers and traces): capacity 8 → 0.42, 16 → 0.57, 24 → 0.67, 32 → 0.76, 48 → 0.87, 64 → 0.91, 96 → 0.93.

## Track-1 decision: expert-cache PoC is justified

Traces show strong reuse by every measured criterion, so the fourth decision gate applies: build a narrowly scoped RAM-to-VRAM expert-cache proof of concept with exact-output validation. Sizing guidance from the LRU simulation: a 48-expert-per-layer cache (37.5% of the 128 experts) would have served ~87% of expert requests from cache; 64 experts ~91%. Caveats that bound the claim: hit rates come from a single sequence at a time (batch-1, short 128-token continuations, greedy), and the simulation is per-layer with perfect LRU — it says nothing yet about transfer latency, bandwidth, or prefetch scheduling, which is exactly what the PoC must measure.

## Cache cost model (2026-08-11)

`experiments/moe_trace/cache_cost_model.py` joins the LRU hit curves with exact per-expert bytes from the GGUF (2.65–3.06 MB per expert depending on layer, gate+up+down), the measured B05 pinned H2D bandwidth (26.06 GB/s), and the two measured generation anchors at the `-ngl 18` split (25.15 tok/s stock, 16.87 tok/s with `--n-cpu-moe 48`), which give a measured CPU-expert-compute cost of `c - g = 1.084 ms` per layer per token. Note on layer numbering: `-ngl 18` offloads the **last** 18 layers (31–47, since `i_gpu_start = n_layer + 1 - n_gpu_layers`), so the CPU-computed MoE layers are 0–30. The cost model's sums were computed over layers 18–47 (30 layers); because per-layer hit rates and expert sizes vary only modestly across layers, the per-layer estimates transfer to any 30-layer cached range, but the PoC cache targets the CPU layers, i.e. a prefix of layers 0–30. For a hypothetical expert cache on 30 CPU-computed layers, estimated generation rate assuming **synchronous** miss transfers and pinned staging:

| Cache/layer | VRAM | Misses/tok/layer | Est. tok/s (pinned, sync) |
|---:|---:|---:|---:|
| 8 | 0.68 GB | 4.56 | 45.0 (34.8 at anchor −20%) |
| 16 | 1.36 GB | 3.30 | 55.2 |
| 32 | 2.73 GB | 1.80 | 76.0 |
| 48 | 4.09 GB | 0.91 | 97.9 |
| 128 (all) | 10.9 GB | 0.54 (compulsory) | 111.1 |

Compute-only ceiling if transfers hide perfectly under CPU attention work: ~138 tok/s. Full sensitivity table (pageable vs pinned, anchor ±20%): `results/moe-locality/cache-cost-model.json`.

Two measured constraints shape the PoC design:

- **VRAM headroom is ~650 MiB.** Peak VRAM during a `-ngl 18` run is 7540 of 8192 MiB (sampled with nvidia-smi during a trace run). Only a capacity-8 cache fits without changing the layer split; larger caches require trading GPU layers for cache space.
- **Pinned staging is mandatory.** With pageable transfers (B04: 11.1 GB/s) a capacity-8 cache only matches the 25.15 tok/s baseline in the worst case; pinned transfers beat it even at anchor −20% (34.8 vs 25.15 tok/s). This matches the CUDA pinned-memory requirement already noted in the implementation-path analysis.

Conclusion: the arithmetic closes with margin under conservative assumptions (synchronous transfers, anchor −20%, smallest fitting cache), so the PoC is worth building. The PoC must validate exact outputs against the parity method used for the trace build, and must measure whether real transfers approach the pinned-B05 assumption when interleaved with compute.

## Expert-cache PoC result (2026-08-11, second session): NEGATIVE at cap 8

A synchronous per-expert VRAM cache was implemented in the pinned build (additive, env-gated `LLAMA_MOE_CACHE=<cap>:<first>[:<last>]`, stock behavior unchanged when unset): exps tensors of cached layers become cap-slot CUDA tensors, full expert data is loaded into pinned host buffers (`--no-mmap` required), and `ggml_cuda_mul_mat_id` is intercepted to D2H the top-k ids, LRU-resolve them against a per-tensor slot table, `cudaMemcpyAsync` misses from pinned host, and run the stock mmvq kernels with slot ids (`ggml/src/ggml-cuda/moe-cache.cu`, ~3 touch points in `llama-model-loader.cpp`, fusion guards in `ggml-cuda.cu`, new `examples/moe-cache/` driver with generation timer and stats JSON). Routing, weights, and kernels are untouched.

Validation and measurements (raw: `results/moe-locality/moe-cache-poc.json`, `cache-stats-*.json`, `cache-parity32-*.txt`, `pinned-ladder.txt`):

- **Correctness**: cache-on vs cache-off is byte-identical over 32 greedy tokens when caching GPU-resident layers (`8:42`), and cache-on is cap-invariant (`8:0:8` ≡ `16:0:8`). For CPU-resident prefix layers the continuation diverges from the CPU baseline at token ~17 — traced to kernel-variant numerics (per-token CUDA mmvq vs CPU batch mul_mat_id), not to wrong expert data: an `--override-tensor` control placing the same layers' exps on stock CUDA reproduces the baseline output exactly, and wrong data produces immediate gibberish, not a late fluent flip.
- **Pinned-host limit**: yesterday `cudaMallocHost` failed at ~4 GiB cumulative; today 8.3 GiB pinned loads fine (24 layers × 3 tensors). The wall is transient system memory pressure, not a fixed ceiling.
- **Performance** (same binary, `GGML_CUDA_DISABLE_GRAPHS=1`, `--no-mmap`, 3 prompts × 3 reps × 128 tokens; stock b10355 reference is 25.15 ± 2.40 tok/s with mmap + graphs on): cache-off **19.67 ± 1.16**, `8:0:8` **18.73 ± 0.28**, `8:0:23` **15.04 ± 0.39** tok/s. The cache is slower than its own baseline, monotonically worse with more cached layers.
- **Hit rate**: measured 0.253 (layers 0–8) and 0.313 (layers 0–23) at cap 8 vs 0.389/0.424 simulated — systematically ~0.10 lower (prompt mix + cold-start; the gap partly closes on longer runs).

Why the cost model was wrong, now measured: transfer volume matched the model (~400 MB/token at cap 8, and the implied effective bandwidth ~25 GB/s confirms pinned B05 holds even for ~1 MB expert-slice transfers). What the model missed is the **per-call resolve overhead**: at batch 1 the cache does a D2H of ids + stream sync + remap + up to 72 small kernel launches per token (24 layers × 3 tensors), which costs about as much per layer as the 1.084 ms CPU expert compute it replaces. Transfers are then pure added latency (~15.7 ms/token at 24 layers) with zero overlap. Synchronous LRU caching at cap 8 cannot win on this machine; the model's compute-saving assumption failed, not its bandwidth assumption.

Track-1 state: locality is real (median reuse distance 2, LRU@48 = 0.87) but a naive sync cache does not monetize it. Any revival needs (a) batch-1-friendly resolve without per-layer stream syncs, (b) larger capacity (VRAM-bound — would require trading GPU layers), and (c) prefetch/overlap, likely using a learned or heuristic next-token expert predictor. That is a substantially larger engineering bet with uncertain payoff; per the decision gates, the default next move is Track 2 (RAM-resident partial offload for dense models) unless the user chooses to fund the async-prefetch iteration.

## Beyond-baseline attempts (2026-08-12/13): all four measured, none pays off yet

After the decoupled placement grid established the ~33–35 tok/s optimum, four "beyond stock" directions were evaluated. All raw artifacts in `results/moe-locality/`.

1. **Async predictive prefetch (v2 cache)** — implemented in `moe-cache.cu` behind `LLAMA_MOE_PREFETCH=1`: dedicated copy stream, per-slot events, token-boundary prefetch pass, drain-event ordering with no host sync. Parity: byte-identical vs cache-off on GPU-resident layers, cap-invariant, v1 path preserved (`v2-parity32-*.txt`). **Structural negative**: with a last-token(s) id-history predictor, prefetch fetches are zero by construction — at cap ≥ 2×8 slots, everything the predictor names is already resident (its own resolves made it MRU). Misses are dominated by *new* experts no id-history predictor can foresee. Offline simulation agreed exactly: predictor+LRU ≡ LRU at every footprint (`prefetch-simulation.json`). Smoke perf: 15.86 vs 14.92 tok/s (v1) at 16:0:16 — far below the 33–35 baseline; the per-layer D2H+resolve overhead (v1's killer) is unchanged by design.
2. **Frequency-static residency (simulation only)** — pinning per-layer top-K frequent experts: top-32 hits 0.56, top-32+LRU-16 0.78, top-48 0.73. Plain LRU@48 alone (0.87) beats every static scheme per footprint, and VRAM at the best placement has only ~1 GiB headroom (~10 experts/layer across 33 CPU-expert layers). No headroom; not built.
3. **Speculative decoding (stock llama.cpp)** — Qwen3-0.6B Q8 draft, `--spec-draft-n-max 8`, best placement. Note: `llama-cli -md` at b10355 is **parsed but silently unused** (no speculative code in `tools/cli/`); the valid path is llama-server (`experiments/moe_trace/speculative_bench.py`, raw `speculative-bench.json`). Warm runs (4 reps, first discarded): **29.03 ± 0.21 with draft vs 29.45 ± 0.15 without** — no gain. Verifying draft tokens costs batched target-model passes through the same RAM/CPU-bound expert path, so speculation cannot amortize here the way it does on GPU-bound setups.
4. **Sync LRU cache (v1, prior section)** — 18.7/15.0 vs 19.7 tok/s: negative.

**Final Track-1 verdict: ~33–35 tok/s (`-ngl 48 --n-cpu-moe 33`) is the measured practical ceiling for Qwen3-30B-A3B on this machine.** The locality is real but every measured monetization path fails on the same structural fact: at batch 1, expert routing is known only at layer time, so residency decisions are always reactive; and VRAM has no headroom for the capacity that would make reactive good enough. The levers that would actually move the number are physical (more RAM → larger MoE rungs; more VRAM → more resident experts) or research-grade (a learned cross-token expert predictor accurate enough to name non-resident experts — the measured ~0.45 adjacent overlap bounds id-history schemes).

## Track-1 closure: decoupled placement grid (2026-08-11, second session)

The original sweep varied only `-ngl` (whole-layer offload of the last N layers). A two-knob grid over `-ngl` × `--n-cpu-moe K` (forces the first K layers' expert tensors to CPU, decoupling attention from expert placement) found a better interior point (`experiments/moe_trace/placement_grid.py`; raw: `results/moe-locality/placement-grid.json`, `placement-grid-refine.json`). Same bench protocol as the original baseline (llama-bench b10355, 128 prompt / 32 gen tokens, mmap, f16 KV).

First pass (3 reps): all-attention-GPU configs dominate; best `-ngl 48 --n-cpu-moe 34` at 28.02 tok/s. Refinement (5 reps, idle machine): peak zone K=32–33:

| ngl | n-cpu-moe | Prompt tok/s | Generation tok/s |
|---:|---:|---:|---:|
| 48 | 31 | 153.0 | 32.76 ± 3.23 |
| 48 | 32 | 175.3 | 33.68 ± 5.27 |
| 48 | 33 | 155.9 / 172.0 | 34.71 ± 3.04 / 33.47 ± 4.61 (repeat) |
| 48 | 34 | 147.8 | 30.66 ± 3.98 |
| 48 | 35 | 144.0 | 29.16 ± 3.22 |

**New best measured placement: `-ngl 48 --n-cpu-moe 33`, ~33–35 generation tok/s — a ~35% improvement over the previous 25.15 ± 2.40 one-knob optimum.** Interpretation: all 48 layers' attention (~12 MB/layer, always fits) runs on GPU, experts of the last 15 layers (~5.5 GB) sit in VRAM, and the remaining 33 layers' experts compute on CPU from mmap'd RAM. Variance between runs is high (±3–5 tok/s), so K=32–34 should be treated as a plateau, not a sharp peak. This supersedes the earlier recommendation; the earlier table is kept above as evidence of the one-knob sweep.

With this, Track 1 (MoE expert locality + llama.cpp placement for Qwen3-30B-A3B on this machine) is closed: locality measured, sync expert cache PoC measured negative, and the practical placement optimum recorded. The expert-cache machinery stays in the pinned build, env-gated, as preserved evidence for a possible later async-prefetch iteration.

## External-claim check: FATE fork reproduction (2026-08-13): NON-REPRODUCIBLE

The deep-research survey (docs/research-report-deep-research.md) flagged one unreplicated llama.cpp fork claiming predictive expert caching beats the Track-1 ceiling: ongunm/llama-moe-cache (AGPL-3.0), claiming Qwen3-30B-A3B Q4_K_M 33.74 → 64.45 tok/s at 99.50% hit on an RTX 4070 Ti 12 GB via a GPU expert pool + cross-layer/temporal prefetch. The fork was cloned (commit 77c8767, gitignored `tools/llama-moe-cache/`), built with CUDA/MSVC, and measured here (RTX 3070 Ti 8 GiB; the author's regime — default `--fit` placement, mmap, GPU computes all experts, per-use H2D). Local patches required to build and to obtain correct output are preserved at `results/moe-locality/fate-repro/fork-patches.patch`; the measurement script is `experiments/moe_trace/fate_repro.py`; raw per-rep logs and `fate-repro.json` sit alongside.

Three fork defects were root-caused during bring-up (all fixed locally, recorded in the patch):

1. **Teardown crash**: the fork never calls `fate_system::shutdown()`; the joinable prefetch worker `std::thread` inside the function-static `fate_system` hits `std::terminate()` at process exit (rc=127 after all output has printed). Fixed by a `llama_fate_shutdown()` called from `llama_free`.
2. **`--no-mmap` crash**: partial `cudaHostRegister` on malloc'd weights (49/144 pinned) enables true-async H2D into reused ggml input_cpy buffers → CUDA invalid argument at layer transitions on this 8 GB placement. The author's mmap config avoids it (0/144 pinned; pageable copies self-serialize). All runs below use mmap.
3. **Output corruption (disqualifier as shipped)**: prefetch loop `memcpy`s every expert into **one shared pinned staging buffer** while its previous async H2D may still be in flight → pool slots receive whatever staging holds at DMA time → FATE-on emits `????…` garbage from token 1. A `FATE_NO_STAGING=1` toggle (direct pageable H2D, synchronizing but correct) makes on-vs-off **token-exact** over 96 greedy tokens. The 98.96% hit rate briefly observed under the race was an artifact: corrupted runs repeat a single token, making temporal prediction trivially perfect.

Measured matrix (fork llama-completion, 3 prompts × 3 reps × 128 tokens, greedy, seed 42, mmap, graphs on, `FATE_NO_STAGING=1` for the on rows; raw: `results/moe-locality/fate-repro/fate-repro.json` + per-rep logs):

| config | generation tok/s | prompt tok/s | hit rate | pool |
|---|---:|---:|---:|---:|
| `--fate` off (fork baseline) | 27.08 ± 0.78 | 41.26 | — | — |
| on, 2048 MB | 4.05 ± 0.02 | 9.05 | 54.00% | 1663 slots |
| on, 1536 MB | 4.05 ± 0.02 | 9.05 | 54.00% | 1247 slots |
| on, 1024 MB | 4.05 ± 0.03 | 9.07 | 54.00% | 831 slots |

(stock b10355 same prompt/seed: 26.46 tok/s; output coherent with fork-off up to a mid-stream token flip from version-drift numerics.)

**Verdict: the claim is non-reproducible in every dimension** — correctness fails as shipped; hit rate is 54%, not 99.5%; generation is 6.7× *slower* than the fork's own off switch, not 1.9× faster. The failure is structural, not tuning: the predictor prefetches the union `cur[layer-1] ∪ prev[layer]` (~2.3× over-fetch, ≈2555 copies/token), which rewrites any 8-GB-fittable pool several times per token, so hit rate and speed are **identical across 831–1663 slots** (counts equal to the unit). The surviving hits are within-transition prefetch→use — a staging delay line, not cross-token caching. Total H2D rises to ~3175 copies/token vs ~1342 for stock (misses + prefetches, 2048 MB row), mostly as synchronous pageable copies on the hook thread; prefill regresses 4.6× (author-reported 4 t/s matches).

This is the Track-1 reactive-routing wall confirmed on an independent implementation: with routing known only at layer time, a heuristic predictor's over-fetch destroys the reuse it tries to monetize. No further iteration on this fork; the earlier verdict (33–35 tok/s placement ceiling; levers are physical or research-grade-predictor) stands reinforced.

## Locality experiment, only after a baseline

The first locality experiment is observational. Capture expert IDs selected per token/layer from an instrumented runtime or an independently validated reference path, then report reuse-distance and hit-rate curves for hypothetical GPU expert-cache capacities. Do not implement an eviction policy until traces show measurable temporal locality.

## Decision gates

- If the model cannot run reliably in 32 GiB RAM, stop and document the capacity boundary; revisit after a RAM upgrade.
- If it runs but GPU offload does not materially help, compare ordinary RAM partial offload before designing expert caching.
- If traces show weak reuse, avoid expert streaming: the NVMe conclusion still applies to cold experts.
- If traces show strong reuse, build a narrowly scoped RAM-to-VRAM expert-cache proof of concept with exact-output validation.

## Later tracks

Track 2 will compare RAM-resident partial offload for dense over-VRAM models. Track 3 will evaluate low-bit/ternary models on compatibility, quality, memory footprint, and generation rate. Neither is selected yet.
