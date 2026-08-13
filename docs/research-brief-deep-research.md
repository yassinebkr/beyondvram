# Deep-research brief: concrete projects that could break the measured walls

Purpose: input prompt for an autonomous deep-research agent (Kimi desktop or
equivalent). Goal: identify concrete, runnable open-source projects — not only
papers — that could push local LLM inference past the walls measured and
recorded in this repository.

The text below the divider is written to be pasted verbatim into the research
agent.

---

## Research task

Survey open-source projects and repositories (papers only when they ship
runnable code) that could enable running a 100B-total-class MoE model with
20–30B active parameters at interactive speed (10+ tok/s generation, batch 1)
on the following consumer hardware envelope, without dropping the active
weight path below Q4-class precision:

- GPU: NVIDIA RTX 3070 Ti, 8 GiB VRAM (Ampere, compute capability 8.6)
- CPU: AMD Zen 3 desktop, 6 cores / 12 threads
- RAM: 64 GiB DDR4 (dual-channel, ~18 GiB/s measured payload copy bandwidth)
- Storage: consumer NVMe, measured direct-I/O sequential read 2.5 GB/s,
  random 1 MiB blocks 2.0 GB/s, random 4 KiB 0.055 GB/s
- OS: Windows 11; toolchains available: MSVC 2022, CUDA 13.3, Python 3.14

## Measured walls that any proposed solution must beat

All numbers below are recorded with raw artifacts; treat them as hard
constraints, not guesses:

1. Dense 32B Q4_K_M at the best GPU/CPU split: 2.7 tok/s, RAM-bandwidth-bound.
   Any dense-model approach at this size class is 12x slower than MoE on this
   machine.
2. Qwen3-30B-A3B (30B total, 3B active, 128 experts, top-8 routing) Q4_K_M
   with the best measured placement (48 layers on GPU, 33 CPU experts):
   33–35 tok/s. This is the current ceiling; raising GPU layers further or
   adding caching on top both measured negative.
3. Expert caching failed both ways at batch 1: a synchronous in-graph expert
   cache slowed 19.7 to 18.7/15.0 tok/s, and an asynchronous prefetch driven
   by a token-id-history predictor was vacuous — expert selection is a
   function of the hidden state, not the token id, and next-token expert
   prediction accuracy from history was near chance (top-8 of 128 experts).
   This reactive-routing property is the single most important wall: experts
   cannot be fetched before the hidden state that selects them is computed.
4. Static frequency-based expert residency is capped by roughly 1 GiB of VRAM
   headroom at the best placement — not enough to hold a meaningful share of
   the expert population.
5. Speculative decoding measured negative: llama-server with a draft model
   reached 29.0 tok/s vs 29.5 baseline — draft cost exceeded acceptance gain
   at batch 1 with CPU-resident experts.
6. Pinned host-to-device copy measured 26 GB/s (pageable 10 GB/s). RAM copy
   payload bandwidth 17.8 GB/s. PCIe is not the bottleneck at batch 1;
   DDR4 capacity and bandwidth are.
7. 1.58-bit BitNet GGUFs use a fork-only quant type that mainline llama.cpp
   cannot load; only the official bitnet.cpp fork runs them.

## What to find

For each category below, list concrete repositories with: URL, last-commit
recency, license, demonstrated hardware, measured tok/s (model + precision),
third-party reproduction evidence, and a concrete integration path (llama.cpp
GGUF compatibility, standalone runtime, or fork).

1. MoE expert-offload engines: kTransformers, PowerInfer / PowerInfer-2,
   MoE-in-a-Box, Flash-MoE, Mixtral-offloading, AirLLM, Fiddler, any llama.cpp
   fork or open PR doing expert-granularity offload beyond `--n-cpu-moe`.
2. Learned expert-prediction / prefetch: projects that predict next-token
   expert selections before the selecting hidden state exists (cross-layer
   predictors, token-conditioned predictors trained on routing traces), with
   code and measured prediction accuracy / cache hit rates. This is the
   specific wall that invalidated the in-repo async-prefetch attempt.
3. Speculative decoding stacks with measured batch-1 wins on consumer GPUs:
   EAGLE-2/EAGLE-3, Medusa, lookahead/n-gram caches; report acceptance rates
   and the draft overhead on 8 GiB-class cards.
4. NVMe-resident inference: engines streaming weights or experts directly
   from NVMe with async pipelines sustaining 2+ GB/s effective: FlexGen,
   DeepSpeed ZeRO-Inference, Fiddler, llama.cpp mmap paths, anything with
   measured numbers on consumer NVMe rather than enterprise Optane/RAID.
5. Low-bit MoE ecosystems that run today: BitNet official fork (b1.58),
   2-bit MoE quants with working Ampere kernels, MXFP4 runtimes (gpt-oss).
6. Model-side candidates: verify which open-weight models actually exist in
   the 100B-total / 20–30B-active class (gpt-oss-120b, DeepSeek, Kimi, Qwen
   MoE lines), their smallest Q4-class quant sizes, and whether any fit a
   64 GiB RAM envelope with the expert layers RAM-resident.
7. Anything that breaks the reactive-routing wall directly: routers exposing
   expert selection earlier in the graph, graph rewrites hoisting selection,
   models trained with predictable or token-id-dependent routing.

## Required output format

A ranked table by expected tok/s on the stated hardware, each row justified
with the evidence links. Flag dead ends explicitly. Where a claim rests only
on the project's own README, mark it as unverified. End with the single most
promising concrete next experiment and the estimated effort to reproduce it
on the stated envelope.

Context repository with all raw measurements:
https://github.com/yassinebkr/beyondvram
