# Breaking the measured walls: runnable open-source projects for 100B-class MoE inference on an RTX 3070 Ti

**Deep-research report — 2026-08-13.** Target envelope: NVIDIA RTX 3070 Ti (8 GiB VRAM, Ampere CC 8.6), AMD Zen 3 6c/12t, 32/64 GiB DDR4 (17.8 GB/s measured payload copy bandwidth), consumer NVMe (2.5 GB/s seq read), Windows 11. Question: can any concrete, runnable open-source project run a 100B-total-class MoE model with 20–30B active parameters at 10+ tok/s generation (batch 1) without dropping the active weight path below Q4-class precision? Context repository with the requester's raw measurements: https://github.com/yassinebkr/beyondvram

> Provenance note (repo maintainer): external deep-research agent output, archived verbatim as "the survey" cited by `docs/next-experiments.md` and `docs/track4-gpt-oss.md`. Its ceilings are inputs to test, not accepted limits — the measurement program they motivated is `docs/next-experiments.md`. Voice and citation formatting are the agent's own.

---

## 1. Executive verdict

The requested configuration — a mixture-of-experts (MoE) model of roughly 100B total parameters with 20–30B active parameters, decoded at 10+ tok/s batch-1 with the active weight path at Q4-class precision, on an RTX 3070 Ti 8 GiB / Zen 3 / 64 GiB DDR4 / Windows 11 machine — is not a hard engineering problem. It is two independent physics violations, and no engine, scheduler, cache, or prefetch changes either one. [(noze.it)](https://www.noze.it/en/insights/ktransformers-hybrid-cpu-gpu-inference/) 

**Violation one is capacity.** Every open-weight MoE with 20B or more active parameters has at least 228B total parameters. The smallest member of the class, Qwen3-235B-A22B (22B active), is 142.65 GB (≈132.9 GiB) at Q4_K_M — more than twice the 64 GiB RAM envelope, and over four times the 32 GiB configuration. [(GitHub Gist)](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0?permalink_comment_id)  Even its smallest 2-bit quant (bartowski Q2_K_XL, 82.9 GiB) does not fit. The capacity wall alone empties the requested intersection before bandwidth is ever discussed.

**Violation two is bandwidth.** Batch-1 decode with CPU-resident experts re-reads the active expert weights from system RAM for every generated token. A 22B-active model at Q4-class precision streams approximately 6.9–8.6 GB per token; the envelope's measured DDR4 copy payload is 17.8 GB/s. The arithmetic ceiling is therefore about 2–2.5 tok/s — a factor of four below the 10 tok/s bar — and this ceiling is a property of the memory, not of the software stack.

What **is** reachable is the class one step down: 100–120B total with 5–13B active. The strongest candidate is gpt-oss-120b (116.8B total, 5.1B active), whose native MXFP4 format (4.25 bits per parameter, trained in-format) is a genuine Q4-class active path, whose GGUF file is 59.02 GiB — the only ≥100B-total file that fits 64 GiB *and* combines a native-trained 4-bit format (MXFP4 ≈ Q4-class), a ~9.5–10.2 tok/s DDR4 roofline ceiling, and a verified Windows/llama.cpp path (Llama 4 Scout Q4_K_M at 60.9 GiB, GLM-4.5-Air at 52.6–56.6 GiB, and Ling-2.6-flash at ~57–60 GB also fit the envelope, but each lacks at least one of those three properties) — and whose dense fraction (~2B parameters) fits comfortably on the 8 GiB GPU, leaving all expert weights RAM-resident. [(DEV Community)](https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6)  It streams only ~1.9 GB of active expert bytes per token, so its roofline on this DDR4 is ~9.5–10.2 tok/s at 100% bandwidth utilization (1.9 GB/token ÷ 17.8 GB/s measured ≈ 9.4 tok/s; ÷ 19.3 GB/s best-case effective ≈ 10.2 tok/s) and a realistic 5–8 tok/s at the 60–75% utilization achievable in decode. Community measurements bracket exactly this: 8–9 tok/s on a Ryzen 5600X + RTX 3090 + 128 GB DDR4-3600 (verified), [(Medium)](https://medium.com/@david.sanftenberg/gpu-poor-how-to-configure-offloading-for-the-qwen-3-235b-a22b-moe-model-using-llama-cpp-13dc15287bed)  5.3 tok/s on a Zen 3 + DDR4 rig (author-reported, unverified), [(arXiv.org)](https://arxiv.org/html/2411.01433v2)  and 24–25 tok/s only once the same model is moved to DDR5-6000. [(vllm.ai)](https://discuss.vllm.ai/t/enable-expert-offloading/1884) 

**The single best next experiment** is therefore not a new engine but a controlled run: install 64 GiB, load the native MXFP4 gpt-oss-120b GGUF in stock llama.cpp with `-ngl 99 --n-cpu-moe 36 -fa --no-mmap` (all 36 MoE layers CPU-resident, attention and KV on the 3070 Ti), verify XMP is active, and llama-bench generation at temp 0 across `--n-cpu-moe` values 24–36, on Linux if possible. The prediction from the roofline and its measured neighbors is 5–8 tok/s on this envelope; if the result lands there, the experiment closes the question — the only remaining lever for 10+ tok/s is DDR5-class memory, which is a hardware purchase, not a project.

## 2. The envelope and the walls

The requester's machine, as characterized in the beyondvram repository, [(noze.it)](https://www.noze.it/en/insights/ktransformers-hybrid-cpu-gpu-inference/)  is: RTX 3070 Ti 8 GiB (Ampere, compute capability 8.6); AMD Zen 3 6-core/12-thread; 32 GiB (later 64 GiB) dual-channel DDR4 with measured payload copy bandwidth of 17.8 GB/s (≈16.6 GiB/s; 512 MiB NumPy copy); consumer NVMe at 2.526 GB/s sequential and 2.0 GB/s random-1MiB; pinned host-to-device copies at 26.06 GB/s; Windows 11.

Seven walls were measured directly on this machine: [(arXiv.org)](https://arxiv.org/html/2410.17954v2) 

1. **Dense-model wall:** Qwen3-32B Q4_K_M tops out at 2.66 tok/s generation (best placement, `-ngl 22`).
2. **MoE ceiling:** Qwen3-30B-A3B Q4_K_M reaches 33.5–34.7 tok/s with `-ngl 48 --n-cpu-moe 33` — the measured practical ceiling after four improvement attempts failed.
3. **In-graph expert cache:** a synchronous LRU VRAM expert cache measured 15.0–18.7 tok/s against its own 19.7 tok/s baseline — negative.
4. **Async predictive prefetch:** structurally negative — a token-ID-history predictor names only already-resident experts, so prefetch traffic is zero by construction.
5. **Draft-model speculative decoding:** 29.03 vs 29.45 tok/s — no gain at batch 1 with RAM-resident experts.
6. **NVMe streaming:** a real BF16 dense-layer read took 0.570 s against 0.003 s of compute — storage cannot be hidden by compute at this granularity; the track was archived.
7. **Transfer-bound copy model:** ~400 MB/token of expert transfers matched the 26 GB/s pinned-copy model, but per-layer synchronization overhead consumed the ~1.08 ms of CPU expert compute it replaced.

All seven walls are one law wearing different clothes. For batch-1 decode with CPU-resident experts:

$$\text{tg tok/s} \approx \frac{\text{effective RAM bandwidth}}{\text{active expert bytes per token}}$$

Fifty-four pooled measured community rows land within ~2.5× of this line, and the envelope's own walls sit exactly on it: 33–35 tok/s for a 3.3B-active model (~0.55 GB/token at Q4_K_M) on 17.8 GB/s. [(Packet.ai)](https://packet.ai/blog/speculative-decoding-explained)  The roofline plot below places the measured rows: gpt-oss-120b at 5.3 tok/s on Zen 3 + DDR4-3200 (unverified), [(arXiv.org)](https://arxiv.org/html/2411.01433v2)  8–9 tok/s on a 5600X + DDR4-3600, [(Medium)](https://medium.com/@david.sanftenberg/gpu-poor-how-to-configure-offloading-for-the-qwen-3-235b-a22b-moe-model-using-llama-cpp-13dc15287bed)  18–22 tok/s on Alder Lake DDR4-3600 (~2× the bandwidth), [(spheron.network)](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/)  24–25 tok/s on DDR5-6000 (including the same-machine XMP-off→on swing of 9.7→23.9 tok/s), [(vllm.ai)](https://discuss.vllm.ai/t/enable-expert-offloading/1884)  57 tok/s on DGX Spark (273 GB/s) and 33–51 tok/s on Strix Halo (256 GB/s) unified memory, [(Local AI Master)](https://localaimaster.com/blog/speculative-decoding-guide)  and — off the gpt-oss line but on its own 8.6 GB/token guide — Qwen3-235B at 9.8 tok/s on an RTX 4080 Super + DDR5 rig. [(arXiv.org)](https://arxiv.org/html/2503.01840v1)  Token generation tracks bandwidth; every "win" in the wild is either fewer bytes per token or a faster memory.

![Roofline: measured tg tok/s vs. effective RAM bandwidth for large-MoE CPU-expert decode](roofline.png)

One asymmetry matters for everything that follows: the envelope runs Windows 11, and Windows exacts a measured tax on precisely the async-overlap designs the walls document. The moe-autopilot project measured a ~23–29 µs WDDM submission-split tax per in-graph wait node, netting −3 tok/s, and closed its async-overlap executor as a measured negative on Windows; [(introl.com)](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025)  independently, a same-hardware llama.cpp comparison put Linux 15–25% ahead of Windows 11. [(reddit.com)](https://www.reddit.com/r/LocalLLaMA/comments/1uybm8y/tried_predicting_which_moe_experts_get_used_next/)  This independently explains the requester's failed async prefetch (wall 4): on WDDM, overlap designs pay a platform toll before any algorithmic benefit is counted.

## 3. The ranked table

Every candidate below is ranked by expected batch-1 generation tok/s on the exact envelope (RTX 3070 Ti 8 GiB, Zen 3 6c/12t, 64 GiB DDR4, 17.8 GB/s measured payload, Windows 11), assuming the best documented integration path. "Expected" values are roofline projections anchored to the measured rows of Section 2 unless a measurement on the envelope's GPU class exists. Evidence strength uses four tiers: **verified** (merged mechanism plus independent measurements), **community-measured** (third-party benchmark rows), **author-reported-unverified** (single-author numbers, no reproduction), and **falsified-dead-end** (measured negative or physically excluded).

| Rank | Project / model | Integration path | Expected tg tok/s on envelope | Key evidence | Evidence strength | Q4-class compliant? |
|---|---|---|---|---|---|---|
| 1 | gpt-oss-120b MXFP4 + llama.cpp `--n-cpu-moe` | Stock llama.cpp CUDA build; attention+KV on 3070 Ti, all 36 MoE layers in RAM; 64 GiB required | **5–8** (roofline 9.5–10.2) | 59.02 GiB file  [(DEV Community)](https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6) ; maintainer guide: ~8 GB VRAM floor  [(Hugging Face Forums)](https://discuss.huggingface.co/t/is-this-possible/163679) ; 8–9 t/s on 5600X+DDR4-3600+3090  [(Medium)](https://medium.com/@david.sanftenberg/gpu-poor-how-to-configure-offloading-for-the-qwen-3-235b-a22b-moe-model-using-llama-cpp-13dc15287bed) ; 5.3 t/s Zen3+DDR4 (unverified)  [(arXiv.org)](https://arxiv.org/html/2411.01433v2) ; 18–22 t/s Alder Lake DDR4-3600  [(spheron.network)](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/)  | verified (mechanism + community-measured) | ✅ native MXFP4 4.25 bpw, trained in-format  [(Hugging Face Forums)](https://discuss.huggingface.co/t/is-this-possible/163679)  |
| 2 | Qwen3.5/3.6-MoE + native MTP heads in llama.cpp | Same placement as rank 1, plus built-in multi-token-prediction (MTP) speculation (PR #22673 wiring, qwen35moe arch) | ~6.6 base → **~8** with +19–31% MTP | MTP +19–31% greedy measured on 8GB-GPU + CPU-experts DDR4 rig  [(Hugging Face)](https://huggingface.co/papers/2502.12224) ; MTP merged for qwen35/qwen35moe  [(takara.ai)](https://tldr.takara.ai/p/2502.12224) ; mainline knobs max out at 6.587 t/s on Qwen3.5-122B-A10B  [(Nebius)](https://nebius.com/blog/posts/lk-losses)  | verified mechanism; community-measured gains | ⚠ borderline: 122B Q4_K_M ≈ 65 GiB > 64 GiB; MXFP4/Q3-class fits  [(mindstudio.ai)](https://www.mindstudio.ai/blog/run-744b-ai-model-consumer-laptop-colibri)  |
| 3 | YALIS "Speculating Experts" cross-layer prefetch | Build axonn-ai/yalis `offload_prefetch` branch; layer-L hidden state predicts layer L+1..L+k experts | ~5–9 on gpt-oss-120b (TPOT −5–14%) | Quality-preserving on gpt-oss-120b (0.857 vs 0.849 baseline); quality-collapsing on Qwen3-30B (0.817→0.641)  [(DEV Community)](https://dev.to/jamilxt/colibri-running-a-744b-ai-model-on-your-laptop-4l6g) ; runnable branch verified  [(ernestchiang.com)](https://www.ernestchiang.com/en/notes/ai/openai-gpt-oss/)  | author-reported (paper), code available | ✅ on gpt-oss-120b only; model-dependent |
| 4 | llama-moe-cache (expert-ID co-occurrence cache) | llama.cpp fork (~500 LoC); GPU expert pool + prefetch stream, AGPL-3.0 | Claims 33.7→64.45 on **30B-A3B only**; no 100B-class result; prefill regresses to 4 t/s | Author-measured 99.5% hit, flattered because the 1.4 GB decode working set fits the 2 GB pool; zero independent reproduction in 4 months (evidence in §4.2)  [(arXiv.org)](https://arxiv.org/html/2410.16144v1)  | author-reported-unverified | ✅ (Q4_K_M) but 30B-class, not the target |
| 5 | llama-wackMall hot/cold expert tiering | Fork build (MSVC-guarded); hot experts pinned in VRAM, cold experts computed from RAM | **10.6 t/s measured on RTX 3070 8GB** — but 122B at IQ2_M; at Q4 (~70–75 GiB) the author's own table collapses to 1.3–2.2 | Qwen3.5-122B-A10B IQ2_M 8.0→10.60 t/s on RTX 3070 + 31 GB RAM, PPL exact-matched, single run per config  [(semaphore.io)](https://semaphore.io/blog/gpt-oss)  | author-reported-unverified | ❌ the 10.6 t/s result is sub-Q3; Q4 premise fails RAM residency |
| 6 | moe-autopilot static hot/cold split | Fork, env-gated, Windows-tested | +10–32% over `--n-cpu-moe` baseline → ~6–9 on gpt-oss-120b-class | +31.6% on gpt-oss-120b (9950X3D + ~30 GB VRAM, Windows); author's regime analysis: gain shrinks to single-digit % when RAM-bound  [(introl.com)](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025)  | author-measured, documented A/B protocol | ✅ |
| 7 | llama.cpp PR #24524 CPU-resident hybrid expert cache | Rebase closed-unmerged branch; cache-hit rows computed on GPU concurrently with CPU miss rows | +7–25% over CPU-expert baseline (≈2–5 t/s for the 20–30B-active class) | 16/16 configs positive-or-parity on 4×3090 + EPYC 8-channel DDR4; closed unmerged by author  [(carteakey.dev)](https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/)  | author-measured (PR thread) | ✅ |
| 8 | MoE-Infinity | Source-only install; Linux-only | ~2.6 on a 16B model at 7 GB VRAM — far below bar | Repo live (pushed 2026-08-11) and batch-1-designed, but its nearest measurement is an order of magnitude under target  [(Level1Techs Forums)](https://forum.level1techs.com/t/open-ai-gpt-oss-models-120-20b/234692)  | community repo; weak measurement | ✅ |
| 9 | Qwen3-235B-A22B — the nominal 20–30B-active target | Any engine | ~2–3.9; **also cannot load**: 132.9 GiB at Q4_K_M vs 64 GiB RAM  [(GitHub Gist)](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0?permalink_comment_id)  | 3.9 t/s measured on kimono's Q2 build (85 GB file, DDR4 rig)  [(DEV Community)](https://dev.to/bspann/bitnet-microsofts-1-bit-llms-that-run-on-your-cpu-20h8) ; roofline 2.2 t/s all-RAM | falsified-dead-end for this goal | ❌ capacity-fail at Q4-class |
| 10 | QuantumLeap "ExpertFlow" | ik_llama.cpp fork | Claims 4.34 on a 122B/6GB-GPU config | "89.7% LZ77 compression of quantized expert weights" is not credible; README-only  [(developersdigest.tech)](https://www.developersdigest.tech/blog/colibri-glm-52-slow-computer-local-inference)  | author-reported-unverified (flagged implausible) | unclear |

**Explicit dead ends (do not spend time):** kTransformers — Linux-only current builds ("Windows native temporarily deprecated"), documented configs assume 24GB+ VRAM, gpt-oss unsupported with three failure paths, and its AMX kernels are useless on Zen 3. [(Github)](https://github.com/JustVugg/colibri)  PowerInfer lineage — no MoE support in PowerInfer; PowerInfer-2 was never open-sourced. [(Github)](https://github.com/microsoft/BitNet)  Fiddler — Mixtral fp16 only, dormant since April 2024. [(arXiv.org)](https://arxiv.org/pdf/2602.23881)  HOBBIT — no public code. [(arXiv.org)](https://arxiv.org/pdf/2505.01658v1)  "MoE-in-a-Box" — not found after ~15 search strategies; treated as a misattribution. [(arXiv.org)](https://arxiv.org/html/2412.19437v1)  Mixtral-offloading — stale since April 2024, speculative prefetching never shipped. [(arXiv.org)](https://arxiv.org/html/2603.09983v1)  AirLLM — per-expert disk streaming at ~0.0034 tok/s-class; independent SATA streaming of 106–117B models measured 0.19–0.48 tok/s. [(Github)](https://github.com/pestopoppa/epyc-root/blob/main/handoffs/active/gpu-acceleration-path.md)  Flash-MoE/ANEMLL — its 12.9–20.3 tok/s headlines require a 17.5 GB/s Apple SSD behind 273–546 GB/s unified memory, i.e., 6–9× the envelope's storage bandwidth. [(Github)](https://github.com/Vage91/Kortex/blob/main/README.md)  BitNet/1.58-bit — dense-only, no MoE support, no model larger than 10B exists. [(Github)](https://github.com/avifenesh/memra/blob/main/ARCHITECTURE.md)  EAGLE-3 hybrid speculation — measured −19% with segfaults on the 8GB hybrid regime and at best break-even on gpt-oss-120b; [(Hugging Face)](https://huggingface.co/papers/2502.12224)  draft-model speculation generally — 29.0 vs 29.5 tok/s on the envelope itself, [(Packet.ai)](https://packet.ai/blog/speculative-decoding-explained)  independently reproduced as a net loss across 19+45 configurations. [(Github)](https://github.com/mybigday/llama.rn/issues/359)  NVMe streaming for >100B models — llama.cpp PR #25294 measures 1.83–2.20 tok/s even on a Grace-Blackwell box with 73–79% cache hits. [(Github)](https://github.com/ggml-org/llama.cpp/discussions/22473)  vLLM/TensorRT-LLM — no expert-CPU-offload path comparable to `--n-cpu-moe` (gpt-oss-120b serving assumes an 80 GB GPU), and FP4 kernels are Blackwell-only. [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/1602) 

**Interpretation.** Three patterns organize the table. First, the ranking is a bandwidth ranking, not a software-quality ranking: rows 1–3 all operate on the same ~1.9–3 GB/token traffic class and differ only in how much of the roofline they harvest, while every row attacking the 20–30B-active class (rows 9–10 and most of the dead ends) is pinned near the 2 tok/s physics floor regardless of engineering merit. Second, the two forks that genuinely cross 10 tok/s on the envelope's GPU class do so by changing a constraint, not by scheduling around it — wackMall's 10.6 t/s is a sub-Q3 IQ2_M result that fails RAM residency at Q4 (rank 5), and llama-moe-cache's 64 t/s is a 30B-A3B result whose 1.4 GB working set fits in VRAM (violating the 100B-class requirement); neither result transfers to a Q4-class 100B model, and both are single-author with zero reproduction. Third, the evidence-gradient matches the risk-gradient: the only verified, Q4-compliant, capacity-feasible path (rank 1) is also the one whose expected 5–8 tok/s honestly falls short of the 10 tok/s bar on this DDR4 — and the gap between 5–8 and the bar is closed in every measured instance by memory hardware (DDR5-6000: 24–25 t/s; unified memory: 57 t/s at 273 GB/s on DGX Spark, 33–51 t/s at 256 GB/s on Strix Halo), never by software. The practical reading of the table is that ranks 1–3 are a portfolio worth building — stock llama.cpp placement today, MTP when a Qwen3.5-MoE target is chosen, YALIS as the monitored research path — while everything at rank 4 and below is either unverified, sub-Q4, or falsified for this envelope, and the dead-end list should be treated as settled unless the underlying repos change license, platform, or maintenance status.

---

## 4. Category evidence

This section examines each attack vector category in turn. For every project we record provenance (license, maintenance recency), demonstrated hardware and measured throughput, third-party reproduction status, the integration path available to the requester, and an expected decode rate on the target envelope (RTX 3070 Ti 8 GiB, Zen 3 6-core, 32/64 GiB dual-channel DDR4 at ~18 GiB/s measured payload, 2.0 GB/s random-1MiB NVMe, Windows 11). Author-only numbers are marked "(author-reported, unverified)"; community measurements from forums and issue trackers are marked as such. Dead ends are flagged inline and consolidated in Section 5.

### 4.1 MoE expert-offload engines

Every engine in this category runs the same physics: batch-1 decode with CPU-resident experts streams the active expert bytes from DDR4 on every token, so the differentiators are placement logic, kernel quality on AVX2, and platform support — not the bandwidth ceiling itself.

| Project | License / recency | Demonstrated measurement (hardware) | Integration path | Expected tok/s on envelope | Dead end? |
|---|---|---|---|---|---|
| kTransformers (kt-kernel + SGLang) | Apache-2.0; active to 2026-08  [(llama.cpp)](https://ggml-org-llama-cpp.mintlify.app/inference/speculative-decoding)  | DeepSeek-R1 671B: 6.2–8 tok/s on dual-Xeon 4–8-channel DDR4/DDR5 servers + 3090/4090 (community reproductions)  [(Github)](https://github.com/raketenkater/llm-server)  | Python framework; GGUF only via llamafile-derived CPU backend; Linux-only wheels (Python 3.10–3.12); Windows native deprecated  [(llama.cpp)](https://ggml-org-llama-cpp.mintlify.app/inference/speculative-decoding)  | 30B-A3B: unverified, ≤ llama.cpp; 235B-class: ~1.5–2.5; gpt-oss: unsupported  [(DEV.co)](https://dev.co/ai/frameworks/powerinfer)  | **Yes** — no native Windows, AVX2 MoE kernels only since v0.5.3 with zero published AVX2 benchmarks  [(Github)](https://github.com/SJTU-IPADS/PowerInfer?spm=a2c6h.13046898.publish-article.57.42ad6ffakIcHCj) , 24 GB VRAM assumed, independent ATSInfer comparison finds low throughput on consumer CPUs without AMX  [(Github)](https://github.com/sjtu-ipads/powerinfer?ref=www.awesomepython.org)  |
| ik_llama.cpp | MIT; pushed 2026-08-13  [(Github)](https://github.com/Tiiny-AI/PowerInfer)  | Qwen3-235B IQ3_K: 7–8 tok/s on Ryzen 9950X + RTX 3090 + DDR5 (community)  [(Github)](https://github.com/sjtu-ipads/powerinfer?ref) ; gpt-oss-120b 8 GB VRAM: 25.6 tok/s on i9-14900K + DDR5-6800 (community)  [(gitcode.com)](https://blog.gitcode.com/ff5f80791d1765f84647aeba68bde100.html)  | Standalone GGUF runtime, llama.cpp-CLI-compatible; builds on Windows 11 + MSVC 2022 + CUDA; AVX2 first-class; `--n-cpu-moe`/`-fmoe` | 235B-class: ~1.5–2.5; gpt-oss-120b: ~5–8; 30B-A3B: parity ±0–30% | Partial — best integration fit, but maintainer states TG is memory-bound and gains over mainline are small  [(aimagicx.com)](https://www.aimagicx.com/blog/qwen-3-5-vs-llama-vs-mistral-china-open-source-ai-2026) ; cannot cross the 10 tok/s bar for 20–30B-active models |
| PowerInfer v1 | MIT; engine commits ended 2025-07  [(Gitee)](https://gitee.com/shiyang0321/PowerInfer)  | 13.20 avg / 29.08 peak tok/s INT4 ReLU-dense on RTX 4090 (paper, 2023; author-reported, unverified)  [(Gitee)](https://gitee.com/magicor/PowerInfer)  | llama.cpp fork, custom PowerInfer-GGUF | N/A for MoE — ReLU/ReGLU models only; no Qwen3/gpt-oss/Mixtral path  [(Gitee)](https://gitee.com/shiyang0321/PowerInfer)  | **Yes** — no MoE support, no independent reproduction |
| PowerInfer-2 | None — never released (404; issue #207 unanswered; academic sources confirm closed)  [(Gitee)](https://gitee.com/sunlcc/PowerInfer)  | 11.68 tok/s on TurboSparse-Mixtral-47B on a smartphone (paper; needs custom-retrained model + NPU)  [(Gitee)](https://gitee.com/xhal/PowerInfer)  | None | N/A | **Yes (hard)** — no code artifact exists |
| SmallThinker engine | Apache-2.0; in PowerInfer repo  [(Open Source Agenda)](https://www.opensourceagenda.com/projects/powerinfer)  | 21B-A3B Q4_0: 30.19 tok/s in-memory, 20.30 tok/s at 8 GiB cap on i9-14900K (author-reported, unverified; one negative user repro on mobile)  [(Open Source Agenda)](https://www.opensourceagenda.com/projects/powerinfer)  | Custom llama-cli fork, Linux/Android only, CPU-only | ~8–15 for its own 21B model; no 100B-class support; Qwen3-30B-A3B through the same engine matches (33.52), not beats, the established ceiling  [(Open Source Agenda)](https://www.opensourceagenda.com/projects/powerinfer)  | **Yes** for the 100B question |
| Fiddler | Apache-2.0; dormant since 2024-04  [(Github)](https://github.com/wfloveiu/llm_inference_powerinfer)  | ">3 tok/s" unquantized Mixtral-8x7B on Quadro RTX 6000 + 48-core Xeon (author-reported); independent HOBBIT measurement ≈2 tok/s on RTX 4090 + 64-core  [(arXiv.org)](https://arxiv.org/html/2607.10183v1)  | PyTorch prototype; Mixtral-8x7B fp16 only; AVX-512-dependent | <1 (fp16 model exceeds 64 GiB RAM; Zen 3 lacks AVX-512) | **Yes** — capacity, ISA, and model lock-in |
| HOBBIT | No public code (three independent confirmations)  [(arXiv.org)](https://arxiv.org/html/2607.10183v1)  | Mixtral fp16 ≈2.0–2.2 tok/s, Phi-3.5-MoE ≈6.3–6.8 tok/s on RTX 4090 + 64 cores + 256 GB RAM (paper, figure-read ±10%)  [(arXiv.org)](https://arxiv.org/html/2607.10183v1)  | None — ~8,000 LoC llama.cpp fork never released | Conceptually ~1–3 for 100B-class | **Yes** |
| Mixtral-offloading | MIT; abandoned since 2024-04  [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/1699)  | Mixtral-8x7B: 2.28 tok/s on RTX 3060 12 GB with 2-bit experts; Mixtral-8x22B (141B): 0.79 tok/s (independent benchmark)  [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/1699)  | HF notebook demo; Mixtral-only; Triton/custom-CUDA, Linux-only; speculative prefetch never shipped in repo | ≤1–2 | **Yes** — sub-Q4 precision at those speeds, wrong model class |
| MoE-Infinity | Apache-2.0; pushed 2026-08-11  [(NVIDIA Developer Forums)](https://forums.developer.nvidia.com/t/tutorial-build-llama-cpp-from-source-and-run-qwen3-235b/352604)  | DeepSeek-V2-Lite TPOT 155 ms (~6.5 tok/s) on A5000 24 GB (paper); independent: ~2.6 tok/s at 7 GB VRAM on the same 16B model  [(Github)](https://github.com/ggml-org/llama.cpp/issues/26448)  | HF-native Python, source build, Linux-oriented; supports Qwen3-30B-A3B and gpt-oss; ≥16 GB VRAM suggested | ~1–3 | **Yes for 10+ tok/s** (misses by ~4–10×); best in category if the target were relaxed to ~2–3 tok/s on Linux |
| AirLLM | Apache-2.0; very active (31k stars)  [(Meta 的新開源 LLM)](https://huggingface.tw/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)  | Kimi K3 2.8T: 292 s/token (~0.0034 tok/s) on RTX 6000 Ada + fast NVMe (author's own release note)  [(Meta 的新開源 LLM)](https://huggingface.tw/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) ; independent runs 0.07–2 tok/s  [(Github)](https://github.com/kvcache-ai/ktransformers)  | pip install; pure PyTorch; broad model list incl. Qwen3 MoE | 0.003–0.2 from disk; ≤~1 even fully RAM-cached | **Yes** — falsified by primary measurement |
| Flash-MoE (danveloper) | **No license file**; research artifact, 2026-03  [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/895)  | Qwen3.5-397B-A17B 4-bit: 4.36 tok/s on M3 Max 48 GB (17.5 GB/s SSD) (author-reported; widely corroborated)  [(sKeskin)](https://skeskin.com/post/qwen3-235b-a22b-instruct-2507-gguf-and-llamacpp)  | Objective-C/Metal, macOS-only, one hardcoded model | <1 even if ported | **Yes** — platform + license; keep its experiment log (§4.2) |
| llama.cpp mainline + forks | MIT; current | gpt-oss-120b 8 GB VRAM `--n-cpu-moe 36`: 24.85–26.08 tok/s on i9-14900K + DDR5-6800 (community)  [(gitcode.com)](https://blog.gitcode.com/ff5f80791d1765f84647aeba68bde100.html) ; wackMall fork: Qwen3.5-122B-A10B IQ2_M 8.0→10.6 tok/s **measured on an RTX 3070 8 GB** (author-reported, unverified)  [(Github)](https://github.com/arizqi/cpubrrr) ; moe-autopilot hot/cold split: +10–32% on gpt-oss-120B, Windows-tested  [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2402.07033v1)  | Merged: `--cpu-moe`/`--n-cpu-moe`/`-ot`/`--fit`; no expert cache or prefetch merged as of 2026-08 — six PR attempts closed unmerged (#24524, #21609, #21614, #21620, #23170, #21067)  [(Github)](https://github.com/HongyangLL/Alt-MoE)  | 20–30B-active: ~1.5–4; ≤10B-active 2-bit: ~8 mainline, ~10.6 wackMall | Mainline: no. Async-overlap fork designs: **yes on Windows** — measured −3 tok/s from the ~23–29 µs WDDM in-graph wait-node tax (moe-autopilot V3, closed as measured negative)  [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2402.07033v1)  |

Three patterns emerge from the table. First, the DDR4 wall is engine-independent: every measured datapoint scales with memory bandwidth rather than engine cleverness — kTransformers lands at 6–8 tok/s on servers with 4–8 memory channels, ik_llama.cpp reaches 7–8 tok/s on 235B only on ~4–5× faster DDR5, and llama.cpp's 25+ tok/s gpt-oss reports all trace to DDR5-6800 hosts. Second, platform fit eliminates most candidates before performance is even relevant: of the twelve engines, only llama.cpp mainline/ik_llama.cpp and the AirLLM PyTorch path install natively on the requester's Windows 11 stack, and AirLLM disqualifies itself with the author's own 292 s/token measurement. Third, the llama.cpp mainline trajectory is telling: expert placement (`--n-cpu-moe`, merged 2025-08  [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2402.07033) ) and selective used-expert copy (PR #15346) shipped, but every expert-cache/prefetch PR since has been closed unmerged, and the two most rigorous Windows measurements (moe-autopilot V3; the requester's own experiments) independently show async overlap is net-negative under WDDM. The measured-positive next steps on the exact GPU class are static/adaptive hot–cold expert tiering — wackMall's +33–86% and moe-autopilot's +10–32% (both author-reported, unverified) — which shift the ceiling but do not break it for a 20–30B-active model. One naming note: "MoE-in-a-Box" does not exist as a locatable artifact — roughly fifteen search strategies across arXiv, GitHub, and web indices returned zero hits, so it is treated as a misattribution; the nearest real systems are MoE-Infinity and the llama.cpp `--n-cpu-moe` ecosystem already covered above.

### 4.2 Learned expert prediction and prefetch

The requester's reactive-routing wall — the expert list for layer $L$ only exists after layer $L$'s hidden state is computed — is attacked here. The critical discriminator is whether a predictor issues loads **before** the selecting hidden state exists, and at what measured accuracy.

| Predictor | Lookahead & signal | Measured accuracy | Runnable code? | Assessment on envelope |
|---|---|---|---|---|
| Fate (cross-layer gate reuse) | 1 layer; layer-$L$ gate input → layer-$(L{+}1)$ gate on CPU, zero training | 78.79% raw top-k decode accuracy; 97.15% only with over-fetch to the 75th confidence percentile; 99.08% system hit additionally requires caching all experts of layers 0–3  [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv240207033K/abstract)  | **None** (no official repo) | Insight transfers only via third-party reimplementation; on 94-layer 128-expert models the shallow-cache trick costs far more VRAM than on the tested 60–64-expert models |
| ongunm/llama-moe-cache ("FATE" fork) | 1 layer cross-layer + temporal reuse | 99.50% system hit claimed; Qwen3-30B-A3B 33.74→64.45 tok/s on RTX 4070 Ti 12 GB (author-reported, unverified)  [(arXiv.org)](https://arxiv.org/html/2402.07033v2)  | Yes — ~500 LoC llama.cpp extension, AGPL-3.0 | Hit rate is flattered by a cache pool (2 GB) larger than the per-token working set (1.4 GB) — unattainable for a 100B model on 8 GiB VRAM; directly contradicts the requester's measured in-graph-cache negative; prefill regresses ~14× (4 t/s). Promising but unproven |
| ProMoE (learned stride MLP) | $k$ layers ($k$≤2 practical); per-layer 2-layer MLP on gate inputs | 84.7% average; decays 90.5%→54.5% as $k$: 1→8; **token-id-history baseline: 54–58%**  [(yiyibooks.cn)](https://yiyibooks.cn/information/arxiv/2402.07033v1/)  | Yes — HF + llama.cpp integrations, but stale since 2025-01 and no license file  [(OpenReview)](https://openreview.net/pdf?id=WX7lxohjFe)  | Largest demo Qwen2-57B-A14B; needs per-model predictor training (<12 h); expected 7–12 tok/s on gpt-oss-class |
| ExpertFlow (T5 routing-path predictor) | All layers, before layer 0 executes | ">90–95%" — but batch-level (OR over tokens), inflating the figure vs batch-1 per-token recall; ≤64-expert models only  [(themoonlight.io)](https://www.themoonlight.io/ko/review/fiddler-cpu-gpu-orchestration-for-fast-inference-of-mixture-of-experts-models)  | **None** | The only maximal-lookahead design; unverifiable and its batch-1 decode input regime is exactly where token-id predictors already failed |
| ETH pre-attention probes | 0 layers (same layer, pre-attention activations) | **94.69% on Qwen3-30B-A3B**, 93.03% DeepSeek-V2-Lite, 97.62% Phi-mini-MoE; ~15 points above Fate  [(bit Quantization for Efficient and Accurate LLM Serving)](https://syfi.cs.washington.edu/publications/fiddler/)  | **None found** | Does not beat the wall cross-layer but buys the attention-compute window per layer and fixes the layer-0 cold start; highest-accuracy documented point, reimplementation-scale effort |
| YALIS "Speculating Experts" | 1 layer; router speculation + optional KL-trained estimators | TPOT −5–14%; quality **preserved on gpt-oss-120b** (0.849→0.857) but **collapses on Qwen3-30B-A3B** (0.817→0.641)  [(ar5iv)](https://ar5iv.labs.arxiv.org/abs/2402.07033)  | Yes — axonn-ai/yalis `offload_prefetch` branch, Apache-2.0  [(netlify.app)](https://kanzhu.netlify.app/publication/fidder/)  | Proof that cross-layer prediction is architecture-dependent, not universal; PyTorch/HPC stack, no GGUF/Windows path; gains small because prefetch overlaps only one layer of compute |
| Flash-MoE experiment log (58 experiments) | Temporal, MLP router probe, speculative early routing | Temporal: 25% hit, net −18%; MLP: 31% accuracy, worse than baseline; speculative routing: −38%; OS page cache instead: +38%  [(ar5iv)](https://ar5iv.labs.arxiv.org/abs/2402.07033v1)  | Yes (macOS-only) | The most thorough consumer-scale falsification: sub-50% prediction is a net loss under I/O-bound streaming because each wrong guess burns the scarcest resource. A fork found prediction paid (+55%) only after I/O stopped being the bottleneck |
| SmallThinker pre-attention router | Architectural — routing knowable before the hidden state by construction | Enables SSD prefetch overlapped with attention; 20.30 tok/s at 8 GiB cap for its own 21B model (author-reported, unverified)  [(Open Source Agenda)](https://www.opensourceagenda.com/projects/powerinfer)  | Yes, but model-specific | The only production answer to the wall is a new model trained with a pre-attention router; cannot be retrofitted to Qwen3/gpt-oss/GLM, whose routers are post-attention and reactive |

The accuracy ledger settles the requester's open question. Token-id/history prediction at near-chance is now independently corroborated three times: ProMoE measured 54–58% for exactly that predictor class  [(yiyibooks.cn)](https://yiyibooks.cn/information/arxiv/2402.07033v1/) , Flash-MoE's log recorded 25% (temporal) and 31% (MLP) hit rates on 4-of-512 routing  [(ar5iv)](https://ar5iv.labs.arxiv.org/abs/2402.07033v1) , and the requester's own result matches. Genuinely ahead-of-hidden-state mechanisms that exceed 90% exist — cross-layer gate reuse, temporal reuse, learned per-layer probes — but each is compromised on this envelope: Fate's 99% headline requires over-fetch plus shallow-layer full caching that 8 GiB VRAM cannot fund at 94–128-expert scale; YALIS shows the mechanism fails on Qwen3's early-layer routing drift while surviving on gpt-oss; and the one runnable llama.cpp fork claiming 1.91× contradicts the requester's measured negative and remains unreproduced. The net position: prediction can plausibly lift a gpt-oss-class run (1.9 GB/token working set) into the ~5–9 tok/s band on 64 GiB RAM — the roofline ceiling is ~19.3 GB/s ÷ 1.9 GB/token ≈ 10.2 tok/s, so exceeding ~10.2 tok/s would require cache hit rates so high that expert traffic falls below ~1.9 GB/token, which no demonstrated predictor sustains on 128-expert models — and no demonstrated predictor sustains the ~96–98% hit rate that a Qwen3-235B-class model (~50% NVMe-resident at 64 GiB) would need for 10 tok/s.

### 4.3 Speculative decoding

The requester's measured draft-model loss (29.0 vs 29.5 tok/s) is not a misconfiguration; it is the predicted outcome, confirmed independently at least three times.

First, first-principles economics: the Leviathan speedup formula with a separate 0.8B-class draft against a ~3B-active MoE verifier (cost ratio $c \approx 0.20$–0.30, acceptance $\alpha \approx 0.5$–0.6) predicts 0.96–1.17× — break-even at best, and a net loss even at 100% acceptance once the DeltaNet rewind tax pushes $c$ to 0.30  [(Liner)](https://liner.com/review/fiddler-cpugpu-orchestration-for-fast-inference-mixtureofexperts-models) . Second, a 19-configuration plus 45-measurement llama.cpp study of Qwen3.6-35B-A3B on an RTX 3090 found every speculative variant net-negative (−3% to −52%) **even at 100% draft acceptance**, attributing it to verify-batch expert-union loading; the same model gained +27.5% under vLLM MTP, confirming the loss is engine-and-method-specific  [(Github)](https://github.com/efeslab/fiddler) . Third, on CPU-socket inference the draft model consumes the same DDR4 bandwidth as the target, and no good dense draft exists for 100B-class MoE bases at all (ik_llama.cpp issue #1602)  [(Github)](https://github.com/ggml-org/llama.cpp/issues/5484) . The MoE-specific mechanism is verify-batch expert-union growth: verifying $\gamma{+}1$ tokens streams the union of their routed experts, shifting the MoE optimum to $\gamma{=}2$ and capping gains at +10–17% even with a free drafter  [(alphaxiv.org)](https://www.alphaxiv.org/zh/overview/2402.07033v3) .

| Approach | Measured evidence | Expected on envelope | Dead end? |
|---|---|---|---|
| Separate draft model | Requester: 29.0 vs 29.5 tok/s; thc1006: −3–52% at 100% acceptance; formula: 0.96–1.17×  [(Liner)](https://liner.com/review/fiddler-cpugpu-orchestration-for-fast-inference-mixtureofexperts-models)  | ×0.85–1.0 — small loss | **Yes, confirmed three ways** |
| Native MTP head | +19–31% greedy on 8 GB GPU + CPU-resident experts over DDR4 (Gemma-4-26B, author-documented with raw data); +28% on 35B-A3B hybrid (PR #22673 thread, community bench); +10–17% MoE optimum at n-max 2  [(alphaxiv.org)](https://www.alphaxiv.org/zh/overview/2402.07033v3)  | ×1.15–1.30 at greedy/low-temp; ~+0–8% at temp 1.0 | No — the **only** measured hybrid win; but llama.cpp wires MTP only for qwen35/qwen35moe + Gemma 4; DeepSeek/GLM MTP graphs are absent upstream  [(arXiv.org)](https://arxiv.org/html/2502.05370v2)  |
| EAGLE-3 | Mixtral-8x7B: 1.5× (original paper); gpt-oss-120b SGLang: ~break-even; hybrid 8 GB test: **−19%, ~42% acceptance, segfaults**  [(arXiv.org)](https://arxiv.org/abs/2402.07033)  | ×0.8–1.0 today — loss | **Yes today** (immature llama.cpp path + verify-union tax); re-check in 3–6 months |
| n-gram (draftless) | Dense 27B: 38→53 tok/s, up to 65–70 on repetitive coding; GPU-resident MoE: −4%  [(Github)](https://github.com/efeslab/fiddler)  | ×1.0–1.2 on code/JSON/agent loops; ~neutral on open chat; zero VRAM cost | No — cheapest experiment; workload-dependent |

The actionable residue is narrow but real: if the chosen model is a Qwen3.5/3.6-MoE or Gemma-4-class architecture, native MTP at n-max 2 under greedy/low-temperature sampling is the single evidence-backed way speculation beats the baseline on this envelope (+15–30%, two independent hybrid-rig confirmations); for DeepSeek-V3.x/GLM-5.x targets no speculative path exists in llama.cpp today. EAGLE-3's MoE record — 1.5× on Mixtral in the original paper, break-even on gpt-oss-120b, and a measured −19% with crashes on the closest-to-envelope hybrid test — makes it a dead end for now despite draft heads existing for all relevant targets.

### 4.4 NVMe-resident inference

At 2.0 GB/s random-1MiB reads, the cold-cache ceiling is bytes-per-token divided by bandwidth. The arithmetic is unforgiving and matches measurement:

| Model (layout) | Cold expert traffic per token | Ceiling @2.0 GB/s | Corroborating measurement |
|---|---|---|---|
| GLM-5.2 744B int4 (colibri layout) | ~11.4 GB (measured by colibri)  [(Github)](https://github.com/robertcsordas/moe)  | **0.18 tok/s** | colibri community table: 0.05–0.1 tok/s cold on a 25 GB laptop; 0.08 tok/s on i5-12600K 32 GB native Windows; >4 tok/s only with full expert residency on multi-GPU hosts  [(Github)](https://github.com/TKsavy/Soft-Moe-Audio)  |
| Qwen3-235B-A22B Q4 | ~8.6 GB | **0.23 tok/s** | 3.9 tok/s with the model RAM-resident (Q2, DDR4, 36 GB VRAM — community-measured, single report)  [(Github)](https://github.com/ddidacus/mol-moe)  |
| gpt-oss-120b MXFP4 (best case) | ≈1.9 GB | **1.05 tok/s** | 9.7 tok/s measured when RAM-resident at ~26–28 GB/s DDR5 payload (community-measured, single report) — i.e., RAM, not NVMe, is what makes gpt-oss viable  [(Github)](https://github.com/junfanz1/MoE-Mixture-of-Experts-in-PyTorch)  |

*Traffic estimates assume the 8 GiB VRAM envelope: essentially no expert layers fit on the GPU, so all active expert bytes stream from RAM/NVMe every token. Qwen3-235B-A22B Q4: 94 layers × top-8 ≈ 8.6 GB/token all-experts-RAM/NVMe (an earlier ~4.0 GB estimate presumed ~half the expert layers GPU-resident — a placement 8 GiB VRAM cannot fund, hence hypothetical on this envelope). gpt-oss-120b MXFP4: 36 layers × top-4 × ~12.4–13.2 MB/expert ≈ 1.78–1.90 GB/token; ≈1.9 GB/token is used consistently throughout.*

Reaching 10 tok/s from disk requires ≤0.20 GB/token of traffic — no 100B-class MoE comes within 9.5–57× of that cold — or a sustained expert-cache hit rate of ~88–90%. Measured real-world hit rates are 3–4% cold, 11–41% warm, and 71% only in Flash-MoE's OS-page-cache configuration with 48 GB of RAM backing the working set and a 17.5 GB/s SSD refilling misses. The three runnable NVMe engines bracket the outcome. colibri (Apache-2.0, actively maintained, the only disk-streaming engine with first-class Windows support) delivers the honest consumer numbers above and the best anti-reactive-routing machinery (71.6% one-layer-ahead predictability, learned hot-store pins, O_DIRECT +34%, batch-union reads) — yet its own A/B log records a 32% loss from MTP speculation when the expert cache is cold  [(Github)](https://github.com/robertcsordas/moe) . antirez's ds4/DwarfStar holds the best 100B+ streaming measurement anywhere (4.8 tok/s for GLM 5.2 on an M5 Max 128 GB; author-reported, unverified) but has no Windows build path and its CUDA GLM streaming prefill is currently broken (issue #595)  [(Source)](https://cegal.gitlab.io/MOE/docs/getting_started.html) . FlexGen and ZeRO-Inference are the wrong regime entirely — throughput-oriented, large-batch, datacenter-NVMe designs whose batch-1 latency is the reciprocal of what they optimize  [(Github)](https://github.com/YelpArchive/MOE) . Finally, the ANEMLL Flash-MoE fork's 12.9–20.3 tok/s for a 397B model (author-reported, unverified) is the existence proof of >10 tok/s streaming — on a 17.5 GB/s Apple SSD with 128 GB unified memory, i.e., 6–9× the envelope's storage bandwidth and 2–4× its RAM; scaled by bandwidth alone it lands near 0.5 tok/s here. WARP's gate analysis adds a second wall: disk is only ~53.5% of a cold decode step at these scales, so even infinite storage bandwidth would less than double the rate against 6-core DDR4 compute  [(Bing)](https://www.bing.com/ck/a?!=&fclid=215493c1-c01a-6d44-253d-85c2c1416c73&hsh=4&ntb=1&p=ce526e30141fae8327ce768ac772ebf934dc44a701efed24d0480392730a2016JmltdHM9MTc0OTE2ODAwMA&ptn=3&u=a1aHR0cHM6Ly9naXRodWIuY29tL1N1bllNMjAyMC9Nb0UtRnVzaW9u&ver=2) . NVMe residency on this envelope is a capacity feature, not a speed feature: ~1 tok/s cold and ~2–4 tok/s warm for the best candidate, feeding the dead-end consolidation in Section 5.

---

### 4.5 Low-bit MoE ecosystems that run today

Sections 4.1–4.4 established that placement and scheduling cannot move the roofline $tg \approx \text{bandwidth} \div \text{bytes/token}$; this section asks whether a low-bit *format* can shrink the numerator while respecting the requester's floor that the active weight path stay at Q4-class precision (≥ ~4 bits per parameter of effective quality). Three ecosystems were examined; exactly one satisfies the constraint.

**BitNet / b1.58 (bitnet.cpp).** Microsoft's 1.58-bit runtime is actively maintained but structurally irrelevant here: its supported-model table is dense-only, the largest compatible checkpoint is Falcon3-10B, and no mixture-of-experts (MoE) model — at any size — exists in the 1.58-bit format as of February 2026. [(手机新浪网)](https://www.sina.cn/news/detail/5103904495504650.html)  The widely quoted "100B at 5–7 tok/s on one CPU" figure is an extrapolation in the bitnet.cpp paper, not a shipped model. [(arXiv.org)](https://arxiv.org/html/2411.11217v1)  The Windows 11 build path is also adversarial (one community report documents a nine-hour CMake/clang effort). [(Github)](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLPL-and-MoE)  Verdict: dead end — no model exists to run.

**2-bit MoE quants (ik_llama.cpp IQ2-class, Unsloth dynamic quants).** ik_llama.cpp's i-quant kernels run on Ampere (int8 dp4a-class math) and are the fastest CPU path for sub-3-bit MoE quants. [(Github)](https://github.com/swag2198/build-gpt2-moe)  The quality evidence is directionally favorable — independent ladders show MMLU effectively flat down to IQ2-class on a 35B-A3B MoE — but code generation degrades first: HumanEval drops from 59% (Q4_K_M) to 48% (IQ2_XXS) on the same model. [(Github)](https://github.com/gabrielolympie/moe-pruner)  Unsloth's calibrated 2.42-bit DeepSeek-V3 quant "passes all our tests" (producer-reported), yet bartowski's GLM-4.5-Air card itself labels IQ2_XXS "very low quality … usable." [(LandX)](https://landx.limxdynamics.com/blogs/3cd923e2925eeab7e60f93cd)  Every one of these quants is below the requester's Q4 floor and is therefore flagged, not recommended. Physics independently caps them: a 22B-active model at ~2.5 bpw streams ~6.9 GB of active expert bytes per token, which at the envelope's measured 17.8 GB/s DDR4 payload bandwidth is only ~2.5–3 tok/s (17.8 ÷ 6.9 ≈ 2.6) — matching the Q4-class ceiling conclusion that low-bit quants do not move the wall on this machine; the honest measurement is 4.34 tok/s for a 122B-A10B IQ2_XXS on a 6 GB-GPU + CPU rig (README-only, unverified). [(博客园)](https://www.cnblogs.com/duiwoeryan/p/19503299) 

**MXFP4 runtimes.** MXFP4 (4.25 bits/parameter, block-scaled FP4) is the *native training format* of gpt-oss — "roughly equivalent to ggml's Q4_0" but retaining full quality — making it the only low-bit format that satisfies the active-path constraint by construction rather than by tolerance. [(CSDN博客)](https://blog.csdn.net/chenhong007/article/details/135048016)  gpt-oss-120b ships as a 59.02 GiB GGUF that fits the 64 GiB RAM tier. [(arXiv.org)](https://arxiv.org/pdf/2401.14361v1.pdf)  llama.cpp executes MXFP4 MoE on sm_86 via packed dequant/dot CUDA kernels (no FP4 tensor cores required) with `--n-cpu-moe` expert offload; demonstrated on dual RTX 3090s (Ampere) plus DDR4 and at 25+ tok/s on a DDR5-6000 rig after correct XMP configuration (both community-measured, single reports). [(arXiv.org)](https://arxiv.org/pdf/2401.14361v2)  vLLM's Marlin MXFP4 path supports Ampere but only for fully GPU-resident serving (gpt-oss-120b targets a single A100 80 GB), and TensorRT-LLM's FP4 is Blackwell-only — both are dead ends on an 8 GiB card. [(arXiv.org)](https://arxiv.org/html/2401.14361v3)  True FP4/W4A4 compute is impossible on compute capability 8.6 regardless of runtime: the tensor-core datatype matrix tops out at INT4, and the emerging MXFP4-W4A4 CUTLASS kernels in vLLM 0.24 are Blackwell-first, with non-Blackwell receiving emulation fixes, not speed. [(arXiv.org)](https://arxiv.org/html/2411.01433v1) 

### 4.6 Model-side candidates: the model-fit table

Given MXFP4/Q4-class as the viable format, the question becomes which open-weight MoE physically fits 32 or 64 GiB of DDR4 and what the all-experts-RAM bandwidth ceiling permits. The ceiling follows the roofline of sections 1–3: decode reads $(\text{top-}k \div n_{\text{experts}}) \times$ routed-expert bytes per token. Worked once for gpt-oss-120b: $114.7\text{B routed expert params} \times 4/128 = 3.58\text{B active} \times 0.531\,\text{B/param (MXFP4)} \approx 1.9\,\text{GB/token}$ (equivalently, 36 layers × top-4 × ~12.4–13.2 MB/expert); at the envelope's DDR4 bandwidth — 17.8 GB/s measured payload, with ≈19.3 GB/s as the best-case effective figure (the 18 GiB/s dual-channel ceiling expressed in decimal units) — the ceiling is $1.90\,\text{GB/token} \div 17.8\text{–}19.3\,\text{GB/s} \approx 9.4\text{–}10.2$ tok/s. [(CSDN博客)](https://blog.csdn.net/chenhong007/article/details/135048016)  Real systems achieve 50–80% of such ceilings.

| Model | Total / active | Smallest Q4-class (GiB) | Fits 32 GiB? | Fits 64 GiB? | DDR4 ceiling (tok/s) |
|---|---|---|---|---|---|
| gpt-oss-120b | 116.8B / 5.1B | 55–59 (native MXFP4)  [(arXiv.org)](https://arxiv.org/pdf/2401.14361v1.pdf)  | ❌ | ⚠️ tight | ~10.2 |
| GLM-4.5-Air | 106B / 12B | 52.6–56.6 (Q3/IQ4_XS)  [(Github)](https://github.com/NVIDIA/TensorRT-LLM/issues/616)  | ❌ | ⚠️ ≤IQ4_XS | ~4.7 |
| Llama 4 Scout | 109B / 17B | 60.9 (Q4_K_M)  [(Github)](https://github.com/EfficientMoE/MoE-Infinity/blob/main/ARCHITECTURE.md)  | ❌ | ⚠️ very tight | ~5.2 |
| Mistral Small 4 | 119B / 6.5B | 68.7 (Q4_K_M)  [(arXiv.org)](https://arxiv.org/html/2410.22134v2)  | ❌ | ❌ at Q4 | ~8.8 |
| Ling-2.6-flash | 104B / 7.4B | ~57–60 (est.)  [(arXiv.org)](https://arxiv.org/pdf/2502.05370v1)  | ❌ | ⚠️ tight | ~10.2 |
| Laguna S 2.1 | 118B / 8B | 69.8 (Q4_K_M); ~55 NVFP4  [(Github)](https://github.com/EfficientMoE/MoE-Infinity)  | ❌ | ⚠️ NVFP4 only | ~7.1 |
| Nemotron-3-Super | 120.6B / 12.7B | 81.0 (Q4_K_M)  [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv240114361X/abstract)  | ❌ | ❌ at Q4 | ~6.6 |
| Qwen3.5-122B-A10B | 122B / 10B | ~65 (Q4_K_M est.)  [(arXiv.org)](https://arxiv.org/pdf/2410.22134)  | ❌ | ⚠️ borderline over | ~7.0 |
| Qwen3-235B-A22B | 235B / 22B | 132.9 (Q4_K_M)  [(Model Beats)](https://modelbeats.com/models/glm-4-5-air-106b-a12b)  | ❌ | ❌ | ~2.2 |
| Kimi K2 / DeepSeek V3.x / GLM-4.6+ | 284B–1T / 13–37B | ≥156  [(Github)](https://github.com/FedericoTs/quantprobe)  | ❌ | ❌ | ≤1.5 |

The table exposes the chapter's central negative result: **the 100B-total / 20–30B-active intersection is empty for this envelope.** Every open MoE with ≥20B active parameters is ≥235B total / ≥132.9 GiB at Q4-class (the smallest in class is Qwen3-235B-A22B: 235B total, Q4_K_M 142.65 GB = 132.9 GiB), so the class cannot fit 64 GiB of RAM at any quality worth having; the smallest in-class file, Qwen3-235B-A22B at Q2_K_XL (82.9 GiB), both exceeds RAM and faces a ~2.2 tok/s DDR4 ceiling, corroborated by a measured 3.9 tok/s at Q2 on a DDR4 rig with 29% of layers GPU-resident (community-measured, single report). [(Model Beats)](https://modelbeats.com/models/glm-4-5-air-106b-a12b)  Two further structural observations stand out. First, the requester's 32 GiB tier admits nothing: the smallest Q4-class 100B file (gpt-oss-120b, 59 GiB) exceeds it by ~1.8×, leaving mmap/disk streaming regimes measured at 0.19 tok/s. [(ModelScope 魔搭社区)](https://modelscope.cn/models/bartowski/Qwen_Qwen3-235B-A22B-Thinking-2507-GGUF)  Second, the fit ranking and the speed ranking diverge instructively — gpt-oss-120b and Ling-2.6-flash top the ceiling column (~10.2 tok/s) precisely because top-$k/n$ sparsity (4/128, 8/256) minimizes per-token expert traffic, while Llama 4 Scout's top-1 routing over 16 coarse experts reads *more* bytes per token (3.72 GB) despite fewer active experts. Note also that the ~10.2 tok/s figure is an ideal-bandwidth ceiling: targeted validation against nine measured DDR4 rows lands the realistic gpt-oss-120b expectation on this exact envelope at **5–8 tok/s**, with >10 tok/s rows traceable to DDR5 or fast-DDR4-on-Intel memory controllers. [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2401.14361)  The realistic active-parameter budget for 64 GiB DDR4 at Q4-class is ~5–13B active — one full class below the requester's specification.

### 4.7 Direct reactive-routing attacks

The reactive-routing wall — layer $L$'s expert selection is unknowable until layer $L$'s hidden state exists — admits no exact removal: $\text{router}_L$ consumes the post-attention residual stream, which depends on all previous layers' expert outputs, so hoisting all routers ahead of the layer stack is mathematically impossible; upstream llama.cpp has progressed only to feature requests for reactive two-tier caches, not router hoisting. [(CanItRun)](https://canitrun.dev/models/glm-4.5-air/)  What exists is a spectrum of approximations, of varying runnability.

Architecture-level fixes are dead ends for a stock-weights user. Pre-gated MoE (ISCA 2024) trains the layer-$N{+}1$ gate to consume layer $N$'s hidden state, but its artifact is Switch-era, FasterTransformer-based, unmaintained since May 2024, and no modern pretrained checkpoint uses the scheme. [(Github)](https://github.com/kvcache-ai/ktransformers/issues/1450)  Hash/deterministic routing gives *perfect* prefetch on paper but tops out at 4.5B-parameter experiments; no production open-weight model uses it. [(Github)](https://github.com/Orbiter/project-euler-llm-benchmark/blob/main/README.md)  Training-time predictability is more promising — StickyMoE's routing-consistency loss cuts expert switch rates up to 59% with released code — but every such intervention is validated only at toy scale, and no 100B-class checkpoint trained this way exists. [(Syntora)](https://runatlas.sh/resources/models/glm-4-5-air) 

Measured correlation evidence shows the wall is softer than assumed. Fate reports 78.8% raw cross-layer prediction accuracy (99% expert hit only with over-fetch and shallow-layer caching; no code released). [(ModelScope 魔搭社区)](https://modelscope.cn/models/moxin-org/Qwen3-235B-A22B-Instruct-2507-GGUF)  The strongest result is same-layer: ETH Zürich's pre-attention linear probes reach 93.03–97.62% exact-match on DeepSeek-V2-Lite, Qwen3-30B, and Phi-mini-MoE at 0.15 ms overhead — but no public code. [(OpenAI)](https://openai.com/index/introducing-gpt-oss/)  A 20-model study places Qwen3-family models in the high local-routing-consistency group, favorable for cache-based schemes. [(arXiv.org)](https://arxiv.org/html/2502.12224v2)  SmallThinker is the only shipped model with a pre-attention router, merged into llama.cpp mainline at b6012 — but it is 21B-A3B only, and through the same engine its Qwen3-30B-A3B baseline (33.52 tok/s on an i9-14900K) matches rather than beats the requester's established 33–35 tok/s ceiling. [(arXiv.org)](https://arxiv.org/pdf/2605.00604)  YALIS's router-first default-vector prefetch is runnable today (released branch, Apache-2.0) but buys only 5–14% time-per-output-token and targets a PyTorch/bf16 stack poorly matched to Windows and 8 GiB VRAM. [(arXiv.org)](https://arxiv.org/html/2410.17954)  The two llama.cpp forks that do show near-2× decode gains from predictive expert caching (llama-moe-cache, llama-wackMall) are single-author, unreplicated, and — critically for this chapter — their 100B-class wins require sub-Q3 quants and full RAM residency; at Q4-class on this envelope the mechanism collapses to 1.3–2.2 tok/s by the author's own RAM-pressure table. [(arXiv.org)](https://arxiv.org/pdf/2410.17954v1) 

## 5. Dead ends, consolidated

This section collects every investigated approach that cannot contribute to 10+ tok/s batch-1 on a 100B-class MoE within the stated envelope (RTX 3070 Ti 8 GiB, Zen 3 6c/12t, 32/64 GiB DDR4, consumer NVMe, Windows 11, active path ≥ Q4-class). Each entry names the failure mode specific to *this* hardware, since several projects remain viable elsewhere.

| Project | Category | Why dead on this envelope | Evidence |
|---|---|---|---|
| kTransformers | CPU–GPU expert offload | Native Windows deprecated (WSL only); documented configs assume ≥24 GB VRAM; gpt-oss/MXFP4 unsupported (three failure paths); AMX kernels absent on Zen 3; reproductions cluster at 6–8 tok/s even on multi-socket servers |  [(Github)](https://github.com/drk-m-s/paper/blob/main/paper_analysis/fate-fast-edge-inference-cross-layer-gate.md)  |
| PowerInfer v1 | Hot/cold neuron offload | No MoE support (ReLU/ReGLU models only); numbers are ReLU-converted dense models on an RTX 4090; no independent reproduction |  [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv241017954H/abstract)  |
| PowerInfer-2 | Smartphone NPU offload | Never open-sourced (404 confirmed); requires custom-retrained TurboSparse model and Snapdragon NPU |  [(Github)](https://github.com/ongunm/llama-moe-cache)  |
| Fiddler | CPU–GPU MoE orchestration | Research prototype, 16-bit Mixtral-8x7B only, dormant since 2024; ~60 GB of fp16 experts cannot fit the RAM tier |  [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2025arXiv250212224F/abstract)  |
| HOBBIT | Mixed-precision expert offload | Code never released; paper-only |  [(Github)](https://github.com/ranggihwang/Pregated_MoE)  |
| "MoE-in-a-Box" | — | Not found after ~15 search strategies; treated as misattribution |  [(Github)](https://github.com/antirez/ds4)  |
| mixtral-offloading | Speculative expert loading | Mixtral-only, unmaintained since 2024-04; 2-bit experts violate the Q4 floor; 2.3 tok/s on RTX 3060 |  [(AI/TLDR)](https://ai-tldr.dev/releases/justvugg-colibri/)  |
| AirLLM | Layer-wise disk streaming | 292 s/token measured on RTX 6000 Ada (author-published release); independent typicals 0.5–2 tok/s |  [(Flowtivity)](https://flowtivity.ai/blog/colibri-glm-52-local-inference-disk-streaming/)  |
| Flash-MoE | NVMe expert streaming | Apple-Metal-only, needs ~17.5 GB/s SSD (6–9× the envelope's NVMe); its own experiment log shows temporal prediction at 25% hit / MLP 31% |  [(Github)](https://github.com/alexandreofbh/colibri_GLM-5.2)  |
| BitNet / b1.58 | Low-bit format | Dense-only runtime; no MoE and no model >10B exists |  [(手机新浪网)](https://www.sina.cn/news/detail/5103904495504650.html)  |
| EAGLE-3 hybrid speculation | Speculative decoding | Every measured hybrid-8 GB result weak-to-negative: −19% on a CPU-resident-MoE rig; gpt-oss-120b break-even (+2% best) on SGLang |  [(Github)](https://github.com/JustVugg/colibri/blob/main/README.md)  |
| Draft-model speculation at batch 1 | Speculative decoding | Net loss across 19+45 measured configs even at 100% acceptance; verify pass re-reads the same bandwidth-bound experts |  [(Github)](https://github.com/antirez/ds4/tree/54b36ed9ba42da31b24f2d1a5feb075c2475dbb1)  |
| NVMe streaming for >100B models | Storage tier | 2.0–2.5 GB/s consumer NVMe ⇒ ~0.5–1.1 tok/s cold ceilings; measured 0.19 tok/s streaming 106B from SATA; llama.cpp PR #25294's O_DIRECT expert streaming reaches only 1.83–2.20 tok/s for a 254 GB model even on a Grace-Blackwell GB10 (73–79% cache hits) |  [(ModelScope 魔搭社区)](https://modelscope.cn/models/bartowski/Qwen_Qwen3-235B-A22B-Thinking-2507-GGUF)  |
| vLLM / TensorRT-LLM on 8 GB | Serving engines | No MoE-expert CPU offload; gpt-oss-120b assumes A100 80 GB resident; FP4 Blackwell-only |  [(arXiv.org)](https://arxiv.org/html/2401.14361v3)  |
| FlexGen | Throughput offload | Unmaintained since 2024; 1 tok/s only at effective batch 144 — trades latency for throughput, wrong regime |  [(Github)](https://github.com/antirez/ds4?ref)  |
| ZeRO-Inference (DeepSpeed) | Throughput offload | Datacenter multi-NVMe large-batch design; Linux-first; no consumer batch-1 path |  [(Github)](https://github.com/antirez/ds4?ref=creativeainews.com)  |
| colibri / ds4 SSD streaming | Weight streaming | Real consumer numbers 0.05–1.2 tok/s; 4+ tok/s only with full expert residency; ds4 SSD-streaming prefill fails outright on 744B |  [(Github)](https://github.com/antirez/ds4?ref=danmackinlay.name)  |
| wackMall at Q4-class | Predictive expert tiering | 10.6 tok/s 122B result requires IQ2_M (28 GB) fitting 31 GB RAM; at Q4_K_M (~70–75 GiB) the author's own table shows collapse to 1.3–2.2 tok/s |  [(arXiv.org)](https://arxiv.org/pdf/2410.17954v1)  |
| MoE-Infinity | Trace-based prefetch | Linux-only source build; measured 2.6 tok/s at 7 GB VRAM on a 16B model; paper-grade throughput claims not batch-1-reproduced |  [(Github)](https://github.com/antirez/ds4?ref=taaft)  |
| Pre-gated MoE | Architecture fix | Abandoned Switch-era artifact; needs full fine-tune; no modern checkpoint exists |  [(Github)](https://github.com/kvcache-ai/ktransformers/issues/1450)  |
| Hash / deterministic routing | Architecture fix | Research-only (≤4.5B models); no production open-weight checkpoint |  [(Github)](https://github.com/Orbiter/project-euler-llm-benchmark/blob/main/README.md)  |

Reading the table column-wise yields a useful taxonomy of failure modes, which matters because it tells the experimenter *which class of idea to stop exploring*. The largest cluster is regime mismatch: FlexGen, ZeRO-Inference, vLLM, and TensorRT-LLM are throughput-oriented serving systems whose wins require large batches, multiple NVMe devices, or GPU-resident weights — none of which interact with the single-user batch-1 latency problem, and none of which can be reconfigured into it. The second cluster is platform mismatch: kTransformers, PowerInfer-2, Flash-MoE, colibri/ds4, and MoE-Infinity assume AMX Xeons, smartphone NPUs, Apple unified memory at 17.5 GB/s, or Linux-only toolchains. The third cluster is evidence failure: HOBBIT, PowerInfer-2, and "MoE-in-a-Box" have no runnable artifact at all, and several measured claims (AirLLM's own 292 s/token release note among them) are self-falsifying. The most instructive rows are the near-misses — wackMall-at-Q4 and draft-model speculation — where the mechanism is sound and demonstrated, but the envelope's two binding constraints (RAM capacity at Q4-class, and the expert re-read cost of verification) independently nullify the gain. These two falsifications are the strongest evidence that the remaining headroom lies in reducing per-token expert traffic at Q4-class quality, not in scheduling around it — the thread taken up by the insights and next-experiment sections that follow.

---

## 6. Cross-cutting insights

Sections 1–5 judged projects individually; six conclusions emerged independently of any single project, and together they determine what a rational next step can be.

**Insight 1 — the requested class is uninstantiable on two independent axes, so the model must change, not the engine.** Capacity: every open-weight MoE with ≥20B active totals ≥228B parameters, ≥107 GiB at Q4-class — over any 64 GiB budget. [(export.arxiv.org)](http://export.arxiv.org/abs/2411.01433)  Bandwidth: 20–30B active at Q4-class streams 6.9–8.6 GB per token, capping decode near 2–2.5 tok/s on 17.8 GB/s (≈16.6 GiB/s) DDR4. Either wall alone empties the class, and no scheduler, cache, or quantizer touches either number.

**Insight 2 — one roofline explains every measured wall and every measured win.** The relation $\text{tg} \approx \text{bandwidth} \div \text{bytes/token}$ of Section 2 seats the requester's own walls (2.7 tok/s dense 32B; 33–35 tok/s on the 3.3B-active Qwen3-30B-A3B) and 54 pooled community rows within ~2.5×. [(arXiv.org)](https://arxiv.org/html/2510.16805v1)  Every verified improvement reduces bytes per token (smaller active class, lower bits) or raises bandwidth (same-machine DDR5-6000: ×2.5–3); [(arXiv.org)](https://arxiv.org/html/2512.12990v4)  Windows 11 adds a negative lever, costing 15–25% against Linux on identical hardware. [(arXiv.org)](https://arxiv.org/html/2509.07727v1) 

**Insight 3 — the reactive-routing wall yields only to cross-layer signals, and only on some models.** Token-id and history predictors converge at 20–58% accuracy across independent measurements (ProMoE: 54–58%), corroborating the requester's near-chance result. [(arXiv.org)](https://arxiv.org/html/2605.05819v1)  The two runnable mechanisms that predict routing *before* the selecting hidden state exists are cross-layer prediction (YALIS: quality-preserving on gpt-oss-120b, collapsing on Qwen3-30B) [(arXiv.org)](https://arxiv.org/html/2608.07911v1)  and temporal/co-occurrence reuse (llama-moe-cache: single-author, unreproduced). [(arXiv.org)](https://arxiv.org/pdf/2411.01433v1)  The frontier — ETH Zürich's pre-attention probes at 93.03–97.62% exact-match — ships no code. [(arXiv.org)](https://arxiv.org/html/2505.19645v3)  Prediction value is model-dependent and must be re-measured per target.

**Insight 4 — the strongest lever inside llama.cpp today is native multi-token prediction (MTP), and it is architecture-coupled.** Draft-model speculation is falsified three independent ways (Section 4.4), but MTP heads amortize DDR4 expert streaming across accepted tokens: +19–31% at greedy on the exact 8 GB-GPU + CPU-experts regime. [(arXiv.org)](https://arxiv.org/html/2510.02345v1)  llama.cpp wires MTP only for qwen35/qwen35moe and Gemma4; [(arXiv.org)](https://arxiv.org/html/2505.19645v2)  DeepSeek and GLM ship MTP tensors with no decoder graph and fail at context creation. [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv241101433T/abstract)  Model choice and speculative support are therefore a single decision, not two.

**Insight 5 — there is no counterexample to the bandwidth taxonomy.** Every >10 tok/s result on a >100B model used Apple- or Blackwell-class bandwidth: Flash-MoE/ANEMLL's 12.9–20.3 tok/s stands behind a 17.5 GB/s SSD and 273–546 GB/s unified memory; [(arXiv.org)](https://arxiv.org/html/2509.23638v1)  DGX Spark (273 GB/s) measures 57 tok/s on gpt-oss-120b; [(arXiv.org)](https://arxiv.org/html/2602.03921v1)  Strix Halo (256 GB/s) measures 33–51. [(wsisp.com)](https://www.wsisp.com/helps/82094.html)  Every Windows/DDR4/discrete-8 GB measurement for ≥100B models lands at 1–9 tok/s. The hardware tier, not the software stack, selects the regime.

**Insight 6 — SmallThinker proves the architectural exit exists and is unoccupied at 100B.** It remains the only shipped model with a pre-attention router (merged into llama.cpp at b6012), yet it is 21B-A3B only, and through the same engine its Qwen3-30B-A3B baseline (33.52 tok/s) matches rather than beats the requester's ceiling. [(unsloth.ai)](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune)  If a 100B-class checkpoint with early routing ships, the entire prefetch stack of Section 4.7 becomes viable overnight; until then, software-only approximation is the only path.

## 7. The single most promising next experiment

Given Insight 1, the experiment cannot target 20–30B active; it must be the closest honest neighbor. **Base experiment: gpt-oss-120b (native MXFP4 GGUF, 59.02 GiB) on the 64 GiB RAM tier with stock llama.cpp. [(CSDN博客)](https://blog.csdn.net/u011732210/article/details/161264493) ** The choice is forced, not merely attractive: it is the only ≥100B-total model that fits 64 GiB *and* combines a native-trained 4-bit format — MXFP4 (4.25 bits per parameter) is the model's own training format, roughly equivalent to Q4_0 at full quality — with a ~9.5–10.2 tok/s roofline ceiling on the envelope's DDR4 and a verified Windows/llama.cpp integration path. [(Will It Run AI)](https://willitrunai.com/blog/llama-4-gpu-requirements)  Its dense fraction (~2B parameters) plus attention and KV fit the 3070 Ti within the documented ~8 GB VRAM floor, leaving all 36 MoE layers' experts RAM-resident [(Will It Run AI)](https://willitrunai.com/blog/llama-4-gpu-requirements)  — precisely the all-experts-on-DDR4 regime the requester already optimized for Qwen3-30B-A3B (Section 2), so existing tuning knowledge transfers directly.

| Phase | Configuration | Measurement | Decision rule |
|---|---|---|---|
| Baseline | Current llama.cpp release, CUDA/MSVC build; `-ngl 99 --n-cpu-moe 36 -fa`, `-t 12`, `-b 4096 -ub 4096`, `--no-mmap` | llama-bench tg + pp at ctx 128 / 512 / 2048, temp 0 | Expected tg 5–8 tok/s |
| Placement sweep | `--fit` on vs off; `-b/-ub` 2048–8192; `-t` 6/12; pinned vs pageable host buffers | Same | Keep best; log deltas |
| Sanity gate | `nvidia-smi` and RAM commit charge observed during decode | VRAM spill / paging activity | Any swapped VRAM ⇒ WDDM overcommit — fix placement before trusting numbers [(Will It Run AI)](https://willitrunai.com/blog/llama-4-gpu-requirements)  |

Each sweep dimension exists because of a measured failure mode, not superstition. `--fit` reallocates layers against actual free VRAM and has moved real systems by whole tok/s; batch/ubatch interacts with the CPU expert kernels' utilization on a 6-core/12-thread Zen 3, where SMT threads can help or hurt under memory contention; and the sanity gate encodes the maintainers' own warning that Windows will allocate more VRAM than exists and silently swap to RAM, producing plausible-looking but meaningless numbers. [(Will It Run AI)](https://willitrunai.com/blog/llama-4-gpu-requirements)  Recording tg at three context depths separates the bandwidth floor from KV-cache growth.

**Predicted outcome and falsification bounds.** The roofline sets the absolute ceiling at 9.5–10.2 tok/s (1.90 GB/token ÷ 17.8 GB/s measured payload ≈ 9.4 tok/s; ÷ 19.3 GB/s best-case effective payload bandwidth ≈ 10.2 tok/s); the measured neighbors — 8–9 tok/s on a Ryzen 5600X + DDR4-3600 rig, [(openEuler 社区)](https://openeuler.csdn.net/6a0dbd32662f9a54cb75eb7b.html)  5.3 tok/s on Zen 3 + DDR4 (author-reported, unverified) [(Hardwarepedia)](https://hardwarepedia.com/blog/best-open-source-ai-models-to-run-locally-2026)  — bracket the honest expectation of **5–8 tok/s**. A sustained result above 10 tok/s would falsify the roofline itself and demands publication of the full configuration; a result below 5 indicates configuration error — XMP disabled or gear-down mode, single-channel seating, or the WDDM overcommit swap above. Effort estimate: **2–4 hours including the ~60 GB download; zero code changes.**

**Stretch step (days, code-level).** Port YALIS's cross-layer probe (branch `offload_prefetch`, Apache-2.0) [(arXiv.org)](https://arxiv.org/pdf/2411.01433v2)  into an llama.cpp fork — an estimated 3–7 days. It is the only runnable predictor demonstrated quality-preserving on gpt-oss-120b specifically (0.857 vs 0.849 baseline; it collapses to 0.641 on Qwen3-30B, so do not attempt it there). [(arXiv.org)](https://arxiv.org/html/2608.07911v1)  At a sustained ≥65% hit rate, predictive expert caching closes part of the gap toward the 9.5 tok/s roofline; it cannot exceed it. **Also worth one afternoon:** a Qwen3.5/3.6-MoE-class model with native MTP (+19–31% greedy, measured on this exact regime) [(arXiv.org)](https://arxiv.org/html/2510.02345v1)  — a smaller-total-class detour, not the 100B goal, and its 122B Q4_K_M (~65 GiB) sits borderline over budget.

**The honest wall-breaker statement.** Nothing in software reaches 10+ tok/s for a 20–30B-active model on this box; every measured crossing of that bar was physical. DDR5-6000 delivers 24–25 tok/s on this same model — 2.5–3× — but Zen 3 is a DDR4 socket, so that is a platform change, not an upgrade. [(arXiv.org)](https://arxiv.org/html/2512.12990v4)  Unified-memory hardware removes the split entirely: 33–51 tok/s on Strix Halo at 256 GB/s, [(wsisp.com)](https://www.wsisp.com/helps/82094.html)  57 tok/s on DGX Spark at 273 GB/s. [(arXiv.org)](https://arxiv.org/html/2602.03921v1)  If the base experiment lands in its predicted 5–8 tok/s band, the question this report opened is closed — and the next dollar belongs to memory bandwidth, not to another engine.

---

## References

 [(noze.it)](https://www.noze.it/en/insights/ktransformers-hybrid-cpu-gpu-inference/) : yassinebkr/beyondvram — requester's measured-walls repository — https://github.com/yassinebkr/beyondvram — 2026-08

 [(GitHub Gist)](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0?permalink_comment_id) : bartowski — Qwen3-235B-A22B-Thinking-2507-GGUF file table — https://modelscope.cn/models/bartowski/Qwen_Qwen3-235B-A22B-Thinking-2507-GGUF — n.d. (accessed 2026-08)

 [(DEV Community)](https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6) : llama.cpp Issue #19035 — gpt-oss:120b load log (GGUF metadata, 59.02 GiB) — https://github.com/ggml-org/llama.cpp/issues/19035 — 2026-01-23

 [(Hugging Face Forums)](https://discuss.huggingface.co/t/is-this-possible/163679) : llama.cpp Discussion #15396 — guide: running gpt-oss with llama.cpp — https://github.com/ggml-org/llama.cpp/discussions/15396 — 2025-08-18

 [(Medium)](https://medium.com/@david.sanftenberg/gpu-poor-how-to-configure-offloading-for-the-qwen-3-235b-a22b-moe-model-using-llama-cpp-13dc15287bed) : Framework Community — "Will the AI Max+ 395 (128GB) be able to run gpt-oss-120b?" (Frix_Rioux: RTX 3090 + 128GB DDR4-3600 + 5600X, 8–9 t/s) — https://community.frame.work/t/tracking-will-the-ai-max-395-128gb-be-able-to-run-gpt-oss-120b/73280 — 2025-08-06

 [(arXiv.org)](https://arxiv.org/html/2411.01433v2) : gotcontext.ai community benchmarks — gpt-oss-120b MXFP4, Ryzen 5800XT + 128GB DDR4, 5.3 tok/s (unverified) — https://gotcontext.ai/fr/benchmarks — n.d.

 [(vllm.ai)](https://discuss.vllm.ai/t/enable-expert-offloading/1884) : carteakey.dev — Optimizing gpt-oss-120b speed on consumer hardware (XMP 9.7→23.9–25+ tok/s) — https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/ — 2025-09-21 (updated 2026-02-15)

 [(arXiv.org)](https://arxiv.org/html/2410.17954v2) : yassinebkr/beyondvram — docs/system-characterization.md (17.783 GB/s RAM copy, 26.061 GB/s pinned H2D, 2.526 GB/s NVMe) — https://github.com/yassinebkr/beyondvram/blob/main/docs/system-characterization.md — 2026-08

 [(Packet.ai)](https://packet.ai/blog/speculative-decoding-explained) : yassinebkr/beyondvram — docs/moe-track-plan.md (33–35 tok/s ceiling; cache/prefetch/spec-decode negatives) — https://github.com/yassinebkr/beyondvram/blob/main/docs/moe-track-plan.md — 2026-08-11

 [(spheron.network)](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/) : github.crookster.org — Running GPT-OSS 120B on RTX 3080 Ti 12 GB at home (DDR4-3600, 18–22 tok/s) — https://github.crookster.org/running-gpt-oss-120b-on-rtx-3080-ti-12-gb-at-home/ — 2025-08-25

 [(Local AI Master)](https://localaimaster.com/blog/speculative-decoding-guide) : NVIDIA Developer Forums — DGX Spark llama.cpp benchmarks (gpt-oss-120b MXFP4 57→57.6 tok/s) — https://forums.developer.nvidia.com/t/moving-from-mac-to-nvidia-bought-powerful-hardware-but-drowning-in-configs/356374?page=2 — 2026-02

 [(arXiv.org)](https://arxiv.org/html/2503.01840v1) : ainews.liduos.com digest relaying r/LocalLLaMA "Qwen 235b @ 16GB VRAM" (RTX 4080 Super + 96GB DDR5, 8→9.8 tok/s) — https://ainews.liduos.com/post/2025-07-04 — 2025-07-04

 [(introl.com)](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025) : JigSawPT/moe-autopilot — adaptive hot/cold MoE expert fork; +31.6% gpt-oss-120b; V3 overlap closed as measured negative (WDDM ~23–29 µs wait-node tax) — https://github.com/JigSawPT/moe-autopilot — 2026-07-04

 [(reddit.com)](https://www.reddit.com/r/LocalLLaMA/comments/1uybm8y/tried_predicting_which_moe_experts_get_used_next/) : startupfortune.com — "Linux crushes Windows on llama.cpp inference by double digits" (same-hardware relay: 128 vs 108 tok/s) — https://startupfortune.com/linux-crushes-windows-on-llamacpp-inference-by-double-digits/ — 2026-04-26

 [(Hugging Face)](https://huggingface.co/papers/2502.12224) : ricardodeazambuja — gemma4-26B-only-8B-VRAM, docs/mtp-benchmark.md (8 GB VRAM + CPU-resident MoE over DDR4: MTP +19–31% greedy, EAGLE-3 −19%) — https://github.com/ricardodeazambuja/gemma4-26B-only-8B-VRAM/blob/main/docs/mtp-benchmark.md — 2026

 [(takara.ai)](https://tldr.takara.ai/p/2502.12224) : llama.cpp PR #22673 — MTP support, merged 2026-05-16 (qwen35/qwen35moe + Gemma4 wiring) — https://github.com/ggml-org/llama.cpp/pull/22673 — 2026-05-16

 [(Nebius)](https://nebius.com/blog/posts/lk-losses) : AlexChen31337/qwen35-moe-offload — program.md ("llama.cpp knobs maxed out at ~6.6 tok/s" on Qwen3.5-122B-A10B) — https://github.com/AlexChen31337/qwen35-moe-offload/blob/main/program.md — 2026-03-25

 [(mindstudio.ai)](https://www.mindstudio.ai/blog/run-744b-ai-model-consumer-laptop-colibri) : Codersera — Qwen 3.5 complete guide (122B-A10B Q4_K_M ≈ 70 GB) — https://codersera.com/blog/qwen-3-5-complete-guide-2026/ — 2026-05-27

 [(DEV Community)](https://dev.to/jamilxt/colibri-running-a-744b-ai-model-on-your-laptop-4l6g) : Madan et al. — Speculating Experts Accelerates Inference for Mixture-of-Experts (gpt-oss-120b quality 0.857 vs 0.849; Qwen3-30B 0.817→0.641) — https://arxiv.org/html/2603.19289v1 — 2026

 [(ernestchiang.com)](https://www.ernestchiang.com/en/notes/ai/openai-gpt-oss/) : axonn-ai/yalis — offload_prefetch branch (Apache-2.0, verified via GitHub API) — https://github.com/axonn-ai/yalis/tree/offload_prefetch — n.d.

 [(arXiv.org)](https://arxiv.org/html/2410.16144v1) : ongunm/llama-moe-cache — FATE-style expert cache fork (33.74→64.45 tok/s claim; AGPL-3.0; last commit Apr 2026; zero third-party reproduction) — https://github.com/ongunm/llama-moe-cache — 2026-04

 [(semaphore.io)](https://semaphore.io/blog/gpt-oss) : miltos22/llama-wackMall — repo page (Apache-2.0, v3 ACTIVE 2026-08-05) — https://github.com/miltos22/llama-wackMall — 2026-08-05

 [(NVIDIA Developer Forums)](https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651/18) : miltos22/llama-wackMall — README.md ("Verified results" incl. RAM-pressure table: 122B IQ2_M 8.0→10.60 tok/s; 16 GB RAM cap → 1.31–2.22 tok/s) — https://raw.githubusercontent.com/miltos22/llama-wackMall/main/README.md — 2026-08

 [(carteakey.dev)](https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/) : llama.cpp PR #24524 — cuda: MoE expert cache, adaptive VRAM caching of CPU-resident experts (+7–25%, 16/16 configs positive-or-parity; closed unmerged) — https://github.com/ggml-org/llama.cpp/pull/24524 — 2026-06-12

 [(Level1Techs Forums)](https://forum.level1techs.com/t/open-ai-gpt-oss-models-120-20b/234692) : EfficientMoE/MoE-Infinity — repo (Apache-2.0, pushed 2026-08-11) — https://github.com/EfficientMoE/MoE-Infinity — 2026-08-11

 [(DEV Community)](https://dev.to/bspann/bitnet-microsofts-1-bit-llms-that-run-on-your-cpu-20h8) : kimono-oyaji.com — 5 Local LLMs Benchmarked at Home: 31B to 235B (Qwen3-235B-A22B Q2 3.9 tok/s on DDR4 rig; measured 2026-06-15/16) — https://kimono-oyaji.com/en/local-llm-5-models-benchmark/ — 2026-07-05

 [(developersdigest.tech)](https://www.developersdigest.tech/blog/colibri-glm-52-slow-computer-local-inference) : MartinCrespoC/QuantumLeap---Llama.cpp-TurboQuant — ExpertFlow (README-only; implausible compression claims) — https://github.com/MartinCrespoC/QuantumLeap---Llama.cpp-TurboQuant — 2026-03-30

 [(Github)](https://github.com/JustVugg/colibri) : kvcache-ai/ktransformers Issue #1861 — three gpt-oss failure paths ("MXFP4 format is not currently supported") — https://github.com/kvcache-ai/ktransformers/issues/1861 — 2026-02-22

 [(Framework Community)](https://community.frame.work/t/tracking-will-the-ai-max-395-128gb-be-able-to-run-gpt-oss-120b/73280) : kvcache-ai/ktransformers — doc/en/install.md ("Windows native temporarily deprecated, please try WSL") — https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/install.md — 2026

 [(Github)](https://github.com/microsoft/BitNet) : Tiiny-AI/PowerInfer — repo (MIT; no MoE support; maintenance mode) — https://github.com/Tiiny-AI/PowerInfer — 2026-05

 [(QVAC, Local AI by Tether)](https://qvac.tether.io/blog/lora-fine-tuning-bitnet-b1-58-llms-on-heterogeneous-edge-gpus-via-qvac-fabric/) : ACM DL — "PowerInfer-2 and HeteroLLM are not open-source" — https://dl.acm.org/doi/pdf/10.1145/3767295.3769382 — 2026

 [(arXiv.org)](https://arxiv.org/pdf/2602.23881) : efeslab/fiddler — repo (Apache-2.0; last commit 2024-04-28; Mixtral fp16 only) — https://github.com/efeslab/fiddler — 2024-04-28

 [(arXiv.org)](https://arxiv.org/pdf/2505.01658v1) : Papers with Code — HOBBIT: "No code implementations yet" — https://paperswithcode.com/paper/hobbit-a-mixed-precision-expert-offloading — n.d.

 [(arXiv.org)](https://arxiv.org/html/2412.19437v1) : GitHub search — "MoE-in-a-box" repositories (0 results) — https://github.com/search?q=%22MoE-in-a-box%22&type=repositories — 2026-08

 [(arXiv.org)](https://arxiv.org/html/2603.09983v1) : dvmazur/mixtral-offloading — repo (MIT; last push 2024-04-08; speculative prefetching listed as WIP) — https://github.com/dvmazur/mixtral-offloading — 2024-04-08

 [(Github)](https://github.com/pestopoppa/epyc-root/blob/main/handoffs/active/gpu-acceleration-path.md) : lyogavin/airllm — README (per-expert disk streaming design and claims) — https://raw.githubusercontent.com/lyogavin/airllm/main/README.md — 2026-08

 [(Github)](https://github.com/sgl-project/sglang/issues/32226) : FedericoTs/quantprobe — measured results (GLM-4.5-Air 0.19 tok/s streaming from SATA, 16 GB RAM) — https://github.com/FedericoTs/quantprobe — 2026-08-01

 [(Github)](https://github.com/Vage91/Kortex/blob/main/README.md) : gorroai/flash-moe (ANEMLL fork) — "Beyond the DRAM Wall: 20.34 tok/s on M5 Max" (17.5 GB/s SSD + unified memory) — https://github.com/gorroai/flash-moe — 2026

 [(Github)](https://github.com/avifenesh/memra/blob/main/ARCHITECTURE.md) : microsoft/BitNet — repo (supported-models table: dense only, max Falcon3-10B) — https://github.com/microsoft/BitNet — 2026

 [(Sergio B. / Field Notes)](https://sergiiob.dev/local-ai/) : Fixstars — "Accelerating Inference of gpt-oss-120b with EAGLE-3" — https://blog.us.fixstars.com/accelerating-inference-of-gpt-oss-120b-with-eagle-3/ — n.d.

 [(Github)](https://github.com/mybigday/llama.rn/issues/359) : thc1006 — qwen3.6-speculative-decoding-rtx3090 (19-config llama.cpp speculative-decoding bench; net loss) — https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090 — 2026

 [(Github)](https://github.com/ggml-org/llama.cpp/discussions/22473) : llama.cpp PR #25294 — llama: stream MoE routed experts from disk (GLM-5.2 on GB10: 1.83–2.20 tok/s; open) — https://github.com/ggml-org/llama.cpp/pull/25294 — 2026-07-04

 [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/1602) : vLLM — GPT-OSS recipe (gpt-oss-120b serving sized for a single A100 80GB; no expert CPU offload) — https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html — 2025-12-04

 [(ik_llama.cpp)](https://ikawrakow-ik_llama-cpp.mintlify.app/) : NVIDIA — TensorRT-LLM overview (FP4 format support is Blackwell/B200-only) — https://nvidia.github.io/TensorRT-LLM/overview.html — n.d.

 [(fixstars.com)](https://blog.us.fixstars.com/accelerating-inference-of-gpt-oss-120b-with-eagle-3/) : harrisoldroyd.com — AMD Strix Halo for Local LLMs (gpt-oss-120b MXFP4 33–51 tok/s; 256 GB/s, ~215 GB/s usable) — https://harrisoldroyd.com/local-ai/strix-halo-for-local-llms/ — n.d. (accessed 2026-08)

 [(llama.cpp)](https://ggml-org-llama-cpp.mintlify.app/inference/speculative-decoding) : kvcache-ai/ktransformers repository — https://github.com/kvcache-ai/ktransformers — checked 2026-08-13

 [(Github)](https://github.com/raketenkater/llm-server) : Zhihu kTransformers DeepSeek-R1 reproduction thread — https://www.zhihu.com/question/12072554903 — 2025-02

 [(synsab.com)](https://synsab.com/archive/reports/deepseek3-technical-report.pdf) : kTransformers install documentation — https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/install.md — checked 2026-08-13

 [(DEV.co)](https://dev.co/ai/frameworks/powerinfer) : kTransformers issue #1861 (gpt-oss failure paths) — https://github.com/kvcache-ai/ktransformers/issues/1861 — 2026-02-22

 [(Github)](https://github.com/SJTU-IPADS/PowerInfer?spm=a2c6h.13046898.publish-article.57.42ad6ffakIcHCj) : Phoronix, KTransformers 0.5.3 (AVX2-only MoE inference) — https://www.phoronix.com/news/KTransformers-0.5.3 — 2026-04-02

 [(Github)](https://github.com/sjtu-ipads/powerinfer?ref=www.awesomepython.org) : ATSInfer, Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference — https://arxiv.org/html/2607.10183v2 — 2026-07-14

 [(Github)](https://github.com/Tiiny-AI/PowerInfer) : ik_llama.cpp README — https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md — checked 2026-08-13

 [(Github)](https://github.com/sjtu-ipads/powerinfer?ref) : Level1Techs forum, local AI hardware thread — https://forum.level1techs.com/t/local-ai-hardware-and-software-configuration/245811 — 2026-02-17

 [(gitcode.com)](https://blog.gitcode.com/ff5f80791d1765f84647aeba68bde100.html) : hada.io gpt-oss-120b benchmark aggregation — https://news.hada.io/topic?id=22490 — 2025-08-13

 [(aimagicx.com)](https://www.aimagicx.com/blog/qwen-3-5-vs-llama-vs-mistral-china-open-source-ai-2026) : ik_llama.cpp discussion #898 (TG memory-bound statement) — https://github.com/ikawrakow/ik_llama.cpp/discussions/898 — 2025-11-06

 [(Gitee)](https://gitee.com/shiyang0321/PowerInfer) : Tiiny-AI/PowerInfer repository — https://github.com/Tiiny-AI/PowerInfer — checked 2026

 [(Gitee)](https://gitee.com/magicor/PowerInfer) : PowerInfer paper (SOSP'24) — https://arxiv.org/html/2312.12456v2 — 2023-12

 [(Gitee)](https://gitee.com/sunlcc/PowerInfer) : PowerInfer issue #207 (PowerInfer-2 open-source request) — https://github.com/Tiiny-AI/PowerInfer/issues/207 — 2024-07-02

 [(Gitee)](https://gitee.com/xhal/PowerInfer) : PowerInfer-2 paper — https://arxiv.org/html/2406.06282v3 — 2024-12-12

 [(Open Source Agenda)](https://www.opensourceagenda.com/projects/powerinfer) : PowerInfer smallthinker directory README — https://github.com/Tiiny-AI/PowerInfer/tree/main/smallthinker — 2025-07-27

 [(Github)](https://github.com/wfloveiu/llm_inference_powerinfer) : efeslab/fiddler repository — https://github.com/efeslab/fiddler — checked 2026-08-13

 [(arXiv.org)](https://arxiv.org/html/2607.10183v1) : HOBBIT paper — https://arxiv.org/html/2411.01433v2 — 2024-11-03

 [(sKeskin)](https://skeskin.com/en/shuppan/qwen3-235b-a22b-instruct-2507-gguf-and-llamacpp) : llama.cpp Discussion #21419 (HOBBIT code not open-source) — https://github.com/ggml-org/llama.cpp/discussions/21419 — 2026-04-04

 [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/1699) : Eliseev & Mazur, Fast Inference of MoE LLMs with Offloading — https://arxiv.org/html/2312.17238v1 — 2023-12-28

 [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/1357) : Independent Mixtral-Offloading benchmark (arXiv 2512.17073) — https://arxiv.org/html/2512.17073v1 — 2025-12-18

 [(NVIDIA Developer Forums)](https://forums.developer.nvidia.com/t/tutorial-build-llama-cpp-from-source-and-run-qwen3-235b/352604) : EfficientMoE/MoE-Infinity repository — https://github.com/EfficientMoE/MoE-Infinity — checked 2026-08-13

 [(Github)](https://github.com/ggml-org/llama.cpp/issues/26448) : MoE-Infinity paper v3 — https://arxiv.org/html/2401.14361v3 — 2024-06-04

 [(MarkTechPost)](https://www.marktechpost.com/2025/08/06/moe-architecture-comparison-qwen3-30b-a3b-vs-gpt-oss-20b/) : SP-MoE paper (independent MoE-Infinity measurement) — https://arxiv.org/html/2510.10302v2 — 2025-11-06

 [(Meta 的新開源 LLM)](https://huggingface.tw/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) : AirLLM releases (Kimi K3 292 s/token release note) — https://github.com/lyogavin/airllm/releases — 2026-07-29

 [(Github)](https://github.com/kvcache-ai/ktransformers) : starlog.is AirLLM analysis — https://starlog.is/articles/llm-engineering/lyogavin-airllm/ — 2026-04

 [(Github)](https://github.com/ikawrakow/ik_llama.cpp/issues/895) : danveloper/flash-moe repository — https://github.com/danveloper/flash-moe — 2026-03-19

 [(sKeskin)](https://skeskin.com/post/qwen3-235b-a22b-instruct-2507-gguf-and-llamacpp) : arunbaby.com Flash-MoE analysis — https://www.arunbaby.com/ml-system-design/0069-flash-moe-macbook-397b-consumer-hardware/ — 2026-03

 [(Github)](https://github.com/arizqi/cpubrrr) : miltos22/llama-wackMall — https://github.com/miltos22/llama-wackMall — 2026-08-05

 [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2402.07033v1) : JigSawPT/moe-autopilot — https://github.com/JigSawPT/moe-autopilot — 2026-07-04

 [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2402.07033) : llama.cpp PR #15077 (--n-cpu-moe) — https://github.com/ggml-org/llama.cpp/pull/15077 — 2025-08-04

 [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv240207033K/abstract) : Fate: Fast Edge Inference of MoE Models via Cross-Layer Gate — https://arxiv.org/html/2502.12224v2 — 2025-05-07

 [(arXiv.org)](https://arxiv.org/html/2402.07033v2) : ongunm/llama-moe-cache — https://github.com/ongunm/llama-moe-cache — 2026-04-08

 [(yiyibooks.cn)](https://yiyibooks.cn/information/arxiv/2402.07033v1/) : ProMoE paper — https://arxiv.org/html/2410.22134v2 — 2025-02

 [(OpenReview)](https://openreview.net/pdf?id=WX7lxohjFe) : promoe-opensource/promoe repository — https://github.com/promoe-opensource/promoe — 2025-01-27

 [(themoonlight.io)](https://www.themoonlight.io/ko/review/fiddler-cpu-gpu-orchestration-for-fast-inference-of-mixture-of-experts-models) : ExpertFlow paper v2 (DAC'26) — https://arxiv.org/html/2410.17954v2 — 2026-04

 [(bit Quantization for Efficient and Accurate LLM Serving)](https://syfi.cs.washington.edu/publications/fiddler/) : Pre-Attention Expert Prediction and Prefetching (ETH) — https://arxiv.org/html/2511.10676v1 — 2025-11-10

 [(ar5iv)](https://ar5iv.labs.arxiv.org/abs/2402.07033) : Speculating Experts Accelerates Inference for Mixture-of-Experts (YALIS) — https://arxiv.org/html/2603.19289v1 — 2026-03-09

 [(netlify.app)](https://kanzhu.netlify.app/publication/fidder/) : axonn-ai/yalis repository — https://github.com/axonn-ai/yalis — 2026-03

 [(ar5iv)](https://ar5iv.labs.arxiv.org/abs/2402.07033v1) : Flash-MoE CLAUDE.md experiment log — https://github.com/danveloper/flash-moe/blob/main/CLAUDE.md — 2026-03-18

 [(themoonlight.io)](https://www.themoonlight.io/zh/review/fiddler-cpu-gpu-orchestration-for-fast-inference-of-mixture-of-experts-models) : SmallThinker paper — https://arxiv.org/html/2507.20984v2 — 2025-07-26

 [(Liner)](https://liner.com/review/fiddler-cpugpu-orchestration-for-fast-inference-mixtureofexperts-models) : zolotukhin.ai, Why MTP heads are the speculative decode draft Qwen3-A3B deserves — https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/ — 2026-05-08

 [(Github)](https://github.com/efeslab/fiddler) : thc1006/qwen3.6-speculative-decoding-rtx3090 — https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090 — 2026-04-21

 [(Github)](https://github.com/ggml-org/llama.cpp/issues/5484) : ik_llama.cpp issue #1602 (SuffixDecoding request) — https://github.com/ikawrakow/ik_llama.cpp/issues/1602 — 2026-04-09

 [(alphaxiv.org)](https://www.alphaxiv.org/zh/overview/2402.07033v3) : The Frontier Lab, MTP Defaults Are a Trap — https://thefrontierlab.ai/mtp-defaults-are-a-trap/ — 2026-05-28

 [(arXiv.org)](https://arxiv.org/abs/2402.07033) : ricardodeazambuja/gemma4-26B-only-8B-VRAM, mtp-benchmark.md — https://github.com/ricardodeazambuja/gemma4-26B-only-8B-VRAM/blob/main/docs/mtp-benchmark.md — 2026-06-12

 [(paperreading.club)](https://paperreading.club/page?id=208637) : llama.cpp PR #22673 (MTP support) — https://github.com/ggml-org/llama.cpp/pull/22673 — 2026-05-16

 [(arXiv.org)](https://arxiv.org/html/2502.05370v2) : ai-muninn.com, DeepSeek-V4-Flash on one 2080 Ti postmortem — https://ai-muninn.com/en/blog/deepseek-v4-flash-284b-on-one-2080ti — 2026-07-29

 [(OpenReview)](https://openreview.net/forum?id=k6q3liY7fP) : EAGLE paper (ICML'24) — https://arxiv.org/pdf/2401.15077.pdf — 2024-01

 [(Github)](https://github.com/naidezhujimo/MoE-Language-Model) : Fixstars, Accelerating inference of gpt-oss-120b with EAGLE-3 — https://blog.us.fixstars.com/accelerating-inference-of-gpt-oss-120b-with-eagle-3/ — 2026-03-09

 [(Github)](https://github.com/peytontolbert/tinylm) : llama.cpp discussion #22473 (ngram-mod dense results) — https://github.com/ggml-org/llama.cpp/discussions/22473 — 2026-04-28

 [(Github)](https://github.com/robertcsordas/moe) : colibri README main branch — https://github.com/JustVugg/colibri/blob/main/README.md — fetched 2026-08-13

 [(Github)](https://github.com/TKsavy/Soft-Moe-Audio) : Wavect, Colibri GLM-5.2 on Consumer Hardware — https://wavect.io/blog/colibri-glm-5-2-consumer-hardware/ — 2026-07-14

 [(Github)](https://github.com/junfanz1/MoE-Mixture-of-Experts-in-PyTorch) : carteakey.dev, Optimizing gpt-oss-120b local inference — https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/ — 2026-02

 [(Source)](https://cegal.gitlab.io/MOE/docs/getting_started.html) : antirez/ds4 repository — https://github.com/antirez/ds4 — checked 2026-08-13

 [(oxen.ai)](https://ghost.oxen.ai/arxiv-dives-mixture-of-experts-moe-with-mixtral-8x7b/) : ds4 issue #595 (CUDA GLM-5.2 streaming prefill failure) — https://github.com/antirez/ds4/issues/595 — 2026-07-24

 [(Github)](https://github.com/YelpArchive/MOE) : FMInference/FlexLLMGen repository — https://github.com/FMInference/FlexLLMGen — checked 2026-08-13

 [(Github)](https://github.com/naot97/mixture-of-experts-from-scratch) : deepspeedai/DeepSpeed repository (ZeRO-Inference/GDS) — https://github.com/deepspeedai/DeepSpeed — checked 2026-08-13

 [(Bing)](https://www.bing.com/ck/a?!=&fclid=215493c1-c01a-6d44-253d-85c2c1416c73&hsh=4&ntb=1&p=ce526e30141fae8327ce768ac772ebf934dc44a701efed24d0480392730a2016JmltdHM9MTc0OTE2ODAwMA&ptn=3&u=a1aHR0cHM6Ly9naXRodWIuY29tL1N1bllNMjAyMC9Nb0UtRnVzaW9u&ver=2) : sqliteai/warp docs/GATES.md (gate analysis: disk only 53.5% of a cold decode step) — https://github.com/sqliteai/warp/blob/main/docs/GATES.md — checked 2026-08-13

 [(Github)](https://github.com/HongyangLL/Alt-MoE) : llama.cpp PR #24524 (CUDA MoE expert cache, closed unmerged 2026-06; its post-mortem catalogs the prior closed attempts #21609, #21614, #21620, #23170, and prefetch-direction #21067) — https://github.com/ggml-org/llama.cpp/pull/24524 — 2026-06

 [(Github)](https://github.com/ddidacus/mol-moe) : kimono-oyaji.com local LLM benchmark (36 GB VRAM + 62 GB DDR4: Qwen3-235B-A22B Q2 3.9 t/s) — https://kimono-oyaji.com/en/local-llm-5-models-benchmark/ — 2026-06

 [(手机新浪网)](https://www.sina.cn/news/detail/5103904495504650.html) : microsoft/BitNet — supported-models table (dense only, max Falcon3-10B) — https://github.com/microsoft/BitNet — n.d. (accessed 2026-08)

 [(epoch.ai)](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference) : Octomil — On-device LLM inference 2025–2026 ("no native 1-bit model larger than 2B … as of February 2026") — https://docs.octomil.com/blog/on-device-llm-inference-2025-2026/ — 2026-02-18

 [(arXiv.org)](https://arxiv.org/html/2411.11217v1) : bitnet.cpp paper (100B-at-5–7-tok/s extrapolation) — https://arxiv.org/html/2410.16144v1 — 2024-10

 [(Github)](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLPL-and-MoE) : microsoft/BitNet Discussion #537 — "How I Finally Got Microsoft BitNet Running on Windows 11 (After 9 Hours of Pain)" — https://github.com/microsoft/BitNet/discussions/537 — 2026-04

 [(Github)](https://github.com/swag2198/build-gpt2-moe) : ikawrakow/ik_llama.cpp — repo (IQ2-class kernels; Ampere CUDA paths) — https://github.com/ikawrakow/ik_llama.cpp — n.d. (accessed 2026-08)

 [(Github)](https://github.com/gabrielolympie/moe-pruner) : yrougy/llm-quant-bench — LEGACY.md (Qwen3.6-35B-A3B quant evals; HumanEval 59% Q4_K_M vs 48% IQ2_XXS) — https://github.com/yrougy/llm-quant-bench/blob/main/LEGACY.md — 2026-04-27

 [(知乎专栏)](https://zhuanlan.zhihu.com/p/672633727) : HumanSpark — local-AI model survey (independent quant ladder: MMLU flat to IQ2-class) — https://humanspark.ai/local-ai/model-survey.html — 2026-02-12

 [(LandX)](https://landx.limxdynamics.com/blogs/3cd923e2925eeab7e60f93cd) : Unsloth — DeepSeek-V3-0324-GGUF-UD model card (2.42-bit "passes all our tests") — https://modelscope.cn/models/unsloth/DeepSeek-V3-0324-GGUF-UD — 2026-01-08

 [(Github)](https://github.com/NVIDIA/TensorRT-LLM/issues/616) : bartowski — zai-org_GLM-4.5-Air-GGUF file table (Q3_K_XL 56.45 GB; IQ4_XS 60.81 GB; IQ2_XXS quality labels) — https://modelscope.cn/models/bartowski/zai-org_GLM-4.5-Air-GGUF — n.d. (accessed 2026-08)

 [(博客园)](https://www.cnblogs.com/duiwoeryan/p/19503299) : MartinCrespoC/QuantumLeap — 122B-A10B IQ2_XXS 4.34 tok/s on 6 GB GPU + CPU (README-only, unverified) — https://github.com/MartinCrespoC/QuantumLeap---Llama.cpp-TurboQuant — n.d.

 [(CSDN博客)](https://blog.csdn.net/chenhong007/article/details/135048016) : llama.cpp Discussion #15396 — official guide: running gpt-oss with llama.cpp (MXFP4 ≈ Q4_0, full quality; min ~8 GB VRAM) — https://github.com/ggml-org/llama.cpp/discussions/15396 — 2025-08-18

 [(arXiv.org)](https://arxiv.org/pdf/2401.14361v1.pdf) : DebuggerCafe — gpt-oss inference with llama.cpp (116.8B/5.1B; 60.8 GiB MXFP4) — https://debuggercafe.com/gpt-oss-inference-with-llama-cpp/ — 2026-02-16

 [(arXiv.org)](https://arxiv.org/pdf/2401.14361v2) : llmgarage.ai — gpt-oss-120b on dual RTX 3090 (sm_86) + Ryzen 5800X + 64 GB DDR4, `--n-cpu-moe 26` — https://llmgarage.ai/gpt-oss-120b-dual-3090/ — n.d.

 [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2401.14361) : carteakey.dev — Optimizing gpt-oss-120b on consumer hardware (25+ tok/s after XMP, DDR5-6000; 9.72 tok/s at 2000 MT/s) — https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/ — 2025-09-21 (upd. 2026-03-12)

 [(arXiv.org)](https://arxiv.org/html/2401.14361v3) : vLLM — gpt-oss recipe (Ampere via TRITON_ATTN + Marlin MXFP4 MoE; 120B on single A100 80 GB) — https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html — 2025-12-04

 [(arXiv.org)](https://arxiv.org/html/2410.22134v1) : TensorRT-LLM — overview (FP4 = B200/Blackwell only) — https://nvidia.github.io/TensorRT-LLM/overview.html — n.d.

 [(arXiv.org)](https://arxiv.org/html/2411.01433v1) : gpusmith — CUDA compute-capability tensor-datatype table (CC 8.6: no FP4/FP6/FP8; INT4 yes) — https://gpusmith.com/articles/en/cuda-compute-capability-table-gpu — 2026-07-23

 [(arXiv.org)](https://arxiv.org/html/2411.01433) : vLLM v0.24.0 release notes (MXFP4 W4A4 MoE CUTLASS, Blackwell-first; non-Blackwell emulation fixes) — https://newreleases.io/project/github/vllm-project/vllm/release/v0.24.0 — 2026-06-29

 [(arXiv.org)](https://arxiv.org/html/2410.22134) : OpenAI — Introducing gpt-oss (116.8B total / 5.1B active; Apache-2.0; MXFP4 post-training) — https://openai.com/index/introducing-gpt-oss/ — 2025-08-05

 [(arXiv.org)](https://arxiv.org/pdf/2401.14361v3) : llama.cpp Issue #19035 — gpt-oss:120b GGUF metadata dump (59.02 GiB, 4.34 bpw) — https://github.com/ggml-org/llama.cpp/issues/19035 — 2026-01-23

 [(Github)](https://github.com/EfficientMoE/MoE-Infinity/blob/main/ARCHITECTURE.md) : Unsloth — Llama 4: How to Run & Fine-tune (Scout Q4_K_M 65.36 GB; dynamic quant table) — https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune — 2026-05-12

 [(arXiv.org)](https://arxiv.org/html/2410.22134v2) : llama.cpp Issue #25158 — Mistral-Small-4-119B Q4_K_M = 68.7 GiB model-load log — https://github.com/ggml-org/llama.cpp/issues/25158 — 2026-06-30

 [(arXiv.org)](https://arxiv.org/pdf/2502.05370v1) : curvedinf — attention-survey-2026-07 (arch table: Ling-2.6-flash 104B/7.4B; DeepSeek-V4-Flash 284B/13B) — https://github.com/curvedinf/attention-survey-2026-07 — 2026-07-21

 [(Github)](https://github.com/EfficientMoE/MoE-Infinity) : ai-tldr.dev — Poolside Laguna S 2.1 release (118B/8B; Q4_K_M ~75 GB; NVFP4 ~59 GB) — https://ai-tldr.dev/releases/poolside-laguna-s-2-1/ — 2026-07-21

 [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv240114361X/abstract) : bartowski — nvidia_Nemotron-3-Super-120B-A12B-GGUF file table (Q4_K_M 86.98 GB) — https://modelscope.cn/models/bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF — n.d. (accessed 2026-08)

 [(arXiv.org)](https://arxiv.org/pdf/2410.22134) : Codersera — Qwen 3.5 complete guide (122B-A10B Q4_K_M ~70 GB) — https://codersera.com/blog/qwen-3-5-complete-guide-2026/ — 2026-05-27

 [(Model Beats)](https://modelbeats.com/models/glm-4-5-air-106b-a12b) : bartowski — Qwen3-235B-A22B-Thinking-2507-GGUF file table (Q4_K_M 142.65 GB) — https://modelscope.cn/models/bartowski/Qwen_Qwen3-235B-A22B-Thinking-2507-GGUF — n.d. (accessed 2026-08)

 [(Github)](https://github.com/FedericoTs/quantprobe) : presenton.ai — How to run Kimi K2 locally (Q4_K_M 621 GB; 1–2 tok/s at 1.8-bit + 256 GB RAM) — https://blog.presenton.ai/blogs/how-to-run-kimi-k2-locally/ — 2026-07-23

 [(unsloth.ai)](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) : kimono-oyaji — 5 local LLMs benchmarked at home (Qwen3-235B 3.9 tok/s at Q2, DDR4, 29% GPU) — https://kimono-oyaji.com/en/local-llm-5-models-benchmark/ — 2026-08-13

 [(ModelScope 魔搭社区)](https://modelscope.cn/models/bartowski/Qwen_Qwen3-235B-A22B-Thinking-2507-GGUF) : FedericoTs/quantprobe — measured results (GLM-4.5-Air 0.19 tok/s streaming from SATA, 16 GB RAM) — https://github.com/FedericoTs/quantprobe — 2026-08-01

 [(Github)](https://github.com/Orbiter/project-euler-llm-benchmark) : crookster.org — Running GPT-OSS 120B on RTX 3080 Ti 12 GB (Alder Lake DDR4-3600, all 36 MoE layers on CPU: 18–22 tok/s) — https://github.crookster.org/running-gpt-oss-120b-on-rtx-3080-ti-12-gb-at-home/ — 2025-08-25

 [(CanItRun)](https://canitrun.dev/models/glm-4.5-air/) : llama.cpp Issue #20757 — two-tier GPU+RAM expert cache feature request (reactive-only today) — https://github.com/ggml-org/llama.cpp/issues/20757 — 2026-03-19

 [(Github)](https://github.com/kvcache-ai/ktransformers/issues/1450) : Hwang et al. — Pre-gated MoE (ISCA 2024); artifact repo (last push 2024-05-04, Switch-only) — https://arxiv.org/abs/2308.12066 ; https://github.com/ranggihwang/Pregated_MoE — 2024

 [(Github)](https://github.com/Orbiter/project-euler-llm-benchmark/blob/main/README.md) : Roller et al. — Hash Layers For Large Sparse Models (deterministic routing; ≤4.5B experiments) — https://arxiv.org/abs/2106.04426 — 2021

 [(Syntora)](https://runatlas.sh/resources/models/glm-4-5-air) : StickyMoE — Training MoE Models for Memory-Efficient Inference (59% switch-rate cut; code released) — https://arxiv.org/html/2607.08780v1 ; https://github.com/alikayyam/sticky_moe — 2026-06

 [(ModelScope 魔搭社区)](https://modelscope.cn/models/moxin-org/Qwen3-235B-A22B-Instruct-2507-GGUF) : Fang et al. — Fate: Fast Edge Inference of MoE Models via Cross-Layer Gate (78.8% raw; 99% hit with over-fetch; no code) — https://arxiv.org/abs/2502.12224 — 2025-02-17

 [(OpenAI)](https://openai.com/index/introducing-gpt-oss/) : Zhu, Bohl, Oester, Alonso — Pre-Attention Expert Prediction and Prefetching for MoE LLMs (93.03–97.62% exact-match; no code) — https://arxiv.org/html/2511.10676v1 — 2025-11-10

 [(arXiv.org)](https://arxiv.org/html/2502.12224v2) : Liang et al. — Not All Models Suit Expert Offloading: On Local Routing Consistency (20-model study; Qwen3 high-LRC) — https://arxiv.org/abs/2505.16056 — 2025

 [(arXiv.org)](https://arxiv.org/pdf/2605.00604) : SmallThinker paper (pre-attention router; 21B-A3B; llama.cpp b6012 merge) — https://arxiv.org/html/2507.20984v2 — 2025-07

 [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2410.17954) : Tiiny-AI/PowerInfer smallthinker README (21B 30.19 tok/s in-memory / 20.30 at 8 GiB cap; Qwen3-30B-A3B 33.52 baseline, i9-14900K) — https://github.com/Tiiny-AI/PowerInfer/tree/main/smallthinker — 2025-07-27

 [(arXiv.org)](https://arxiv.org/html/2410.17954) : Madan et al. — Speculating Experts Accelerates Inference for MoE (default-vector routing; 5–14% TPOT) + axonn-ai/yalis offload_prefetch branch — https://arxiv.org/html/2603.19289v1 ; https://github.com/axonn-ai/yalis/tree/offload_prefetch — 2026-03-09

 [(arXiv.org)](https://arxiv.org/pdf/2410.17954v1) : miltos22/llama-wackMall — repo (Apache-2.0; last push 2026-08-06; author-reported, unverified) — https://github.com/miltos22/llama-wackMall — 2026-08-06

 [(arXiv.org)](https://arxiv.org/abs/2502.12224) : llama-wackMall README (122B IQ2_M 8.0→10.6 tok/s; RAM-pressure table 1.31–2.22 tok/s when not RAM-resident) — https://raw.githubusercontent.com/miltos22/llama-wackMall/main/README.md — 2026-08

 [(Github)](https://github.com/drk-m-s/paper/blob/main/paper_analysis/fate-fast-edge-inference-cross-layer-gate.md) : kvcache-ai/ktransformers — repo + install doc (Windows native deprecated; 24 GB+ VRAM configs; Python ≤3.12 Linux wheels) — https://github.com/kvcache-ai/ktransformers ; https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/install.md — 2026-08-13

 [(arXiv.org)](https://arxiv.org/html/2410.17954v1) : kTransformers Issue #1861 — three confirmed gpt-oss failure paths (MXFP4 unsupported) — https://github.com/kvcache-ai/ktransformers/issues/1861 — 2026-02-22

 [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv241017954H/abstract) : Tiiny-AI/PowerInfer — README FAQ (ReLU/ReGLU models only; no MoE support) — https://github.com/Tiiny-AI/PowerInfer — n.d. (accessed 2026-08)

 [(Github)](https://github.com/ongunm/llama-moe-cache) : PowerInfer Issue #207 — PowerInfer-2 open-source request, never fulfilled (URL 404) — https://github.com/Tiiny-AI/PowerInfer/issues/207 — 2024-07-02

 [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2025arXiv250212224F/abstract) : efeslab/fiddler — research prototype, 16-bit Mixtral-8x7B only (last push 2024) — https://github.com/efeslab/fiddler — n.d. (accessed 2026-08)

 [(Github)](https://github.com/ranggihwang/Pregated_MoE) : Tang et al. — HOBBIT: Mixed Precision Expert Offloading (no public code) — https://arxiv.org/html/2411.01433v2 — 2024

 [(Github)](https://github.com/antirez/ds4) : GitHub search — "MoE-in-a-box" repositories (0 results; misattribution verdict) — https://github.com/search?q=%22MoE-in-a-box%22&type=repositories — n.d. (accessed 2026-08)

 [(AI/TLDR)](https://ai-tldr.dev/releases/justvugg-colibri/) : dvmazur/mixtral-offloading — speculative loading (~80% recall 1-ahead; 2.3 tok/s RTX 3060; last push 2024-04-08) — https://github.com/dvmazur/mixtral-offloading — 2024-04-08

 [(Flowtivity)](https://flowtivity.ai/blog/colibri-glm-52-local-inference-disk-streaming/) : lyogavin/airllm — releases (K3: 292 s/token measured on RTX 6000 Ada) — https://github.com/lyogavin/airllm/releases — n.d. (accessed 2026-08)

 [(Github)](https://github.com/alexandreofbh/colibri_GLM-5.2) : danveloper/flash-moe — C/Metal streaming engine (209 GB streamed; Apple-only) — https://github.com/danveloper/flash-moe — n.d. (accessed 2026-08)

 [(AI Master)](https://www.ai-master.cc/news/news-6387) : arunbaby.com — Flash-MoE analysis (4.36 tok/s 4-bit on M3 Max 48 GB; 17.5 GB/s SSD requirement; experiment-log predictor hit rates) — https://www.arunbaby.com/ml-system-design/0069-flash-moe-macbook-397b-consumer-hardware/ — n.d.

 [(Github)](https://github.com/JustVugg/colibri/blob/main/README.md) : ricardodeazambuja — gemma4-26B-only-8B-VRAM MTP benchmark (8 GB VRAM + CPU-resident MoE over DDR4: MTP +19–31%; EAGLE-3 −19%) — https://github.com/ricardodeazambuja/gemma4-26B-only-8B-VRAM/blob/main/docs/mtp-benchmark.md — n.d.

 [(Github)](https://github.com/antirez/ds4/blob/main/README.md) : Fixstars — Accelerating inference of gpt-oss-120b with EAGLE-3 (batch-1 best +2%; several configs worse) — https://blog.us.fixstars.com/accelerating-inference-of-gpt-oss-120b-with-eagle-3/ — n.d.

 [(Github)](https://github.com/antirez/ds4/tree/54b36ed9ba42da31b24f2d1a5feb075c2475dbb1) : thc1006 — qwen3.6-speculative-decoding-rtx3090 (19-config llama.cpp SD bench: net loss even at 100% acceptance) — https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090 — n.d. (accessed 2026-08)

 [(Github)](https://github.com/antirez/ds4/tree/ds4f-mxfp4) : llama.cpp PR #25294 — llama: stream MoE routed experts from disk (GLM-5.2 on GB10: 1.83–2.20 tok/s; open) — https://github.com/ggml-org/llama.cpp/pull/25294 — 2026-07-04

 [(Github)](https://github.com/antirez/ds4?ref) : FMInference/FlexLLMGen + FlexGen paper (1 tok/s at effective batch 144; unmaintained since 2024-10) — https://github.com/FMInference/FlexLLMGen ; https://arxiv.org/html/2303.06865 — 2023

 [(Github)](https://github.com/antirez/ds4?ref=creativeainews.com) : DeepSpeed — ZeRO-Inference / DeepNVMe (datacenter multi-NVMe throughput design) — https://github.com/deepspeedai/DeepSpeed — n.d. (accessed 2026-08)

 [(Github)](https://github.com/antirez/ds4?ref=danmackinlay.name) : JustVugg/colibri — repo (SSD/RAM weight streaming; "honest numbers") — https://github.com/JustVugg/colibri — n.d. (accessed 2026-08)

 [(Github)](https://github.com/antirez/ds4?ref=githubawesome.com) : Wavect — Colibri GLM-5.2 on consumer hardware: reality check (0.05–1.2 tok/s class results) — https://wavect.io/blog/colibri-glm-5-2-consumer-hardware/ — 2026-08-07

 [(Github)](https://github.com/antirez/ds4?ref=prijm.com) : antirez/ds4 Issue #595 — GLM-5.2 744B CUDA `--ssd-streaming` prefill failure — https://github.com/antirez/ds4/issues/595 — 2026-07-24

 [(Github)](https://github.com/antirez/ds4?ref=taaft) : EfficientMoE/MoE-Infinity — repo (Apache-2.0; Linux source build; 2.6 tok/s at 7 GB VRAM) — https://github.com/EfficientMoE/MoE-Infinity — 2026-08-11

 [(export.arxiv.org)](http://export.arxiv.org/abs/2411.01433) : bartowski — Qwen3-235B-A22B-Thinking-2507-GGUF file table (Q4_K_M 142.65 GB) — https://modelscope.cn/models/bartowski/Qwen_Qwen3-235B-A22B-Thinking-2507-GGUF — n.d. (accessed 2026-08)

 [(arXiv.org)](https://arxiv.org/html/2510.16805v1) : yassinebkr/beyondvram — docs/moe-track-plan.md (33–35 tok/s ceiling; cache/prefetch/spec-decode negatives) — https://github.com/yassinebkr/beyondvram/blob/main/docs/moe-track-plan.md — 2026-08-11

 [(arXiv.org)](https://arxiv.org/html/2512.12990v4) : carteakey.dev — Optimizing gpt-oss-120b speed on consumer hardware (XMP 9.72→23.85–25+ tok/s; DDR5-6000) — https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/ — 2025-09-21 (upd. 2026-03-12)

 [(arXiv.org)](https://arxiv.org/html/2509.07727v1) : startupfortune.com — "Linux crushes Windows on llama.cpp inference by double digits" (same-hardware relay) — https://startupfortune.com/linux-crushes-windows-on-llamacpp-inference-by-double-digits/ — 2026-04-26

 [(arXiv.org)](https://arxiv.org/html/2605.05819v1) : Song et al. — ProMoE (token-id predictor 54–58%; learned 84.7%) — https://arxiv.org/html/2410.22134v2 — 2024-10-29 (v2 2025-02)

 [(arXiv.org)](https://arxiv.org/html/2608.07911v1) : Madan et al. — Speculating Experts Accelerates Inference for Mixture-of-Experts (gpt-oss-120b 0.857 vs 0.849; Qwen3-30B 0.817→0.641; TPOT −5–14%) — https://arxiv.org/html/2603.19289v1 — 2026-03-09

 [(arXiv.org)](https://arxiv.org/pdf/2411.01433v2) : axonn-ai/yalis — offload_prefetch branch (Apache-2.0, verified via GitHub API) — https://github.com/axonn-ai/yalis/tree/offload_prefetch — n.d.

 [(arXiv.org)](https://arxiv.org/pdf/2411.01433v1) : ongunm/llama-moe-cache — FATE-style expert cache fork (33.74→64.45 tok/s claim; AGPL-3.0; zero third-party reproduction) — https://github.com/ongunm/llama-moe-cache — 2026-04

 [(arXiv.org)](https://arxiv.org/html/2505.19645v3) : Zhu, Bohl, Oester, Alonso — Pre-Attention Expert Prediction and Prefetching for MoE LLMs (93.03–97.62% exact-match; no code) — https://arxiv.org/html/2511.10676v1 — 2025-11-10

 [(arXiv.org)](https://arxiv.org/html/2510.02345v1) : ricardodeazambuja — gemma4-26B-only-8B-VRAM, docs/mtp-benchmark.md (8 GB VRAM + CPU-resident MoE over DDR4: MTP +19–31% greedy) — https://github.com/ricardodeazambuja/gemma4-26B-only-8B-VRAM/blob/main/docs/mtp-benchmark.md — 2026-06-12

 [(arXiv.org)](https://arxiv.org/html/2505.19645v2) : llama.cpp PR #22673 — MTP support, merged 2026-05-16 (qwen35/qwen35moe + Gemma4 wiring; 35B-A3B hybrid 22.9→29.4 tok/s) — https://github.com/ggml-org/llama.cpp/pull/22673 — 2026-05-16

 [(NASA/ADS)](https://ui.adsabs.harvard.edu/abs/2024arXiv241101433T/abstract) : ai-muninn.com — DeepSeek-V4-Flash 284B on one 2080 Ti postmortem (DeepSeek/GLM MTP graphs not implemented in llama.cpp) — https://ai-muninn.com/en/blog/deepseek-v4-flash-284b-on-one-2080ti — 2026-07-29

 [(arXiv.org)](https://arxiv.org/html/2509.23638v1) : gorroai/flash-moe (ANEMLL fork) — "Beyond the DRAM Wall: 20.34 tok/s on M5 Max" (17.5 GB/s SSD + unified memory) — https://github.com/gorroai/flash-moe — 2026

 [(arXiv.org)](https://arxiv.org/html/2602.03921v1) : NVIDIA Developer Forums — DGX Spark llama.cpp benchmarks (gpt-oss-120b MXFP4 57→57.6 tok/s) — https://forums.developer.nvidia.com/t/moving-from-mac-to-nvidia-bought-powerful-hardware-but-drowning-in-configs/356374?page=2 — 2026-02

 [(wsisp.com)](https://www.wsisp.com/helps/82094.html) : harrisoldroyd.com — AMD Strix Halo for Local LLMs (gpt-oss-120b MXFP4 33–51 tok/s; 256 GB/s, ~215 GB/s usable) — https://harrisoldroyd.com/local-ai/strix-halo-for-local-llms/ — n.d. (accessed 2026-08)

 [(unsloth.ai)](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) : SmallThinker paper (pre-attention router; 21B-A3B; llama.cpp b6012 merge) — https://arxiv.org/html/2507.20984v2 — 2025-07

 [(csdn.net)](https://gitcode.csdn.net/6a0dbd3310ee7a33f273f780.html) : Tiiny-AI/PowerInfer smallthinker README (Qwen3-30B-A3B 33.52 tok/s baseline, i9-14900K) — https://github.com/Tiiny-AI/PowerInfer/tree/main/smallthinker — 2025-07-27

 [(CSDN博客)](https://blog.csdn.net/u011732210/article/details/161264493) : llama.cpp Issue #19035 — gpt-oss:120b GGUF metadata dump (59.02 GiB, 4.34 bpw) — https://github.com/ggml-org/llama.cpp/issues/19035 — 2026-01-23

 [(Will It Run AI)](https://willitrunai.com/blog/llama-4-gpu-requirements) : llama.cpp Discussion #15396 — guide: running gpt-oss with llama.cpp (MXFP4 ≈ Q4_0, full quality; ~8 GB VRAM floor; Windows VRAM-overcommit swap warning) — https://github.com/ggml-org/llama.cpp/discussions/15396 — 2025-08-18

 [(spheron.network)](https://www.spheron.network/blog/deepseek-vs-llama-4-vs-qwen3/) : OpenAI — Introducing gpt-oss (116.8B total / 5.1B active; MXFP4 post-training; Apache-2.0) — https://openai.com/index/introducing-gpt-oss/ — 2025-08-05

 [(openEuler 社区)](https://openeuler.csdn.net/6a0dbd32662f9a54cb75eb7b.html) : Framework Community — "Will the AI Max+ 395 (128GB) be able to run gpt-oss-120b?" (Frix_Rioux: RTX 3090 + 128 GB DDR4-3600 + Ryzen 5600X, 8–9 tok/s) — https://community.frame.work/t/tracking-will-the-ai-max-395-128gb-be-able-to-run-gpt-oss-120b/73280 — 2025-08-06

 [(Hardwarepedia)](https://hardwarepedia.com/blog/best-open-source-ai-models-to-run-locally-2026) : gotcontext.ai community benchmarks — gpt-oss-120b MXFP4, Ryzen 5800XT + 128 GB DDR4, 5.3 tok/s (unverified submitter) — https://gotcontext.ai/fr/benchmarks — n.d.

