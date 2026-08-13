# Roadmap: toward 100B-class MoE on consumer hardware

Status: planning document, 2026-08-13. Hardware change pending: +32 GiB RAM (64 GiB total), arriving in 2–3 weeks.

## Where the measurements leave the project

Every software angle measured so far says the same thing: on a RAM-bound consumer machine, practical speed comes from sparse activation, not from moving dense weights faster. The measured frontier on this machine is Qwen3-30B-A3B at ~33–35 tok/s (`-ngl 48 --n-cpu-moe 33`, llama.cpp b10355). Beyond-stock software speedups (sync cache, async prefetch, static residency, speculative decoding) were all measured and none pays off — see `docs/moe-track-plan.md`.

The binding constraint for bigger models is **RAM capacity**, not the GPU. MoE inference needs the full expert pool resident in RAM (mmap'd GGUF); only the active few GB per token do real work.

## What 64 GiB RAM unlocks

Rule of thumb from the measured baselines: sustained usable configs need model size ≤ ~60% of RAM (leaving headroom for OS, KV, pinned buffers).

| Candidate | Total | Active | Q4-class size | Fits 64 GiB? | Expected class |
|---|---:|---:|---:|---|---|
| Qwen3-30B-A3B (current) | 30.5B | 3.3B | 18.6 GB | already running | ~34 tok/s measured |
| gpt-oss-20b (MXFP4) | 21B | 3.6B | ~12 GB | already fits in 32 GiB | second MoE arch, methodology check |
| gpt-oss-120b (MXFP4) | 117B | 5.1B | ~60 GB | tight but plausible | the realistic big-rung target |
| Qwen3-235B-A22B | 235B | 22B | ~130 GB | no | needs ~256 GB RAM or NVMe track |
| DeepSeek-V3/R1 class | 671B | 37B | ~380 GB | no | far outside 64 GiB |
| Kimi-K2 class | ~1T | 32B | ~500+ GB | no | far outside 64 GiB |

Honest boundary: the flagship 20–30B-active MoE models (DeepSeek, Kimi) are an order of magnitude beyond 64 GiB RAM at any usable quantization. The 100B-class, ~5B-active rung (gpt-oss-120b) is the largest credible target for this machine after the upgrade. A future NVMe-resident expert track would be required for anything bigger — and the archived dense-streaming result plus the Track-1 cache negatives say that path needs a predictive component that does not exist yet.

## Work plan when the RAM arrives

1. Re-run system characterization (B03 RAM bandwidth may change with population/channel config; record the new system state alongside).
2. Placement grid method (Track 1, two-knob `-ngl` × `--n-cpu-moe`) applied to gpt-oss-120b — first 100B-class measurement.
3. Optional preliminary: gpt-oss-20b today (fits current 32 GiB) to validate the method on a second MoE architecture (MXFP4 experts, different router) before the big download.
4. Perplexity anchor on the fixed corpus for quality comparison across rungs.

## Trust in the method

The methodology is the transferable asset: pinned llama.cpp revision, byte-exact parity gating, placement grids, raw-repetition retention. Each new model/rung is a measurement campaign, not a research restart.
