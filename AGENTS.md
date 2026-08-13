# AGENTS.md

## Project overview

BeyondVRAM is a measurement-first consumer-hardware LLM research workspace. Its first dense `NVMe -> RAM -> pinned staging -> VRAM` streaming track is archived: it established correctness but was storage-bound and not competitive with a practical llama.cpp Q4 baseline on this hardware. See `docs/archive-dense-streaming.md` before proposing work in that archived direction.

Track 1 (MoE expert locality) is **closed**: expert-routing locality measured (strong), and all four beyond-stock speedups measured **negative** — sync expert cache (19.67 baseline vs 18.73/15.04 tok/s), async predictive prefetch (predictor structurally vacuous), frequency-static residency (VRAM-capped), and speculative decoding (29.0 vs 29.5 tok/s). Practical optimum: `-ngl 48 --n-cpu-moe 33` at ~33–35 tok/s. A post-closure external-claim check (ongunm/llama-moe-cache FATE fork, claiming 33.74→64.45 tok/s) was measured **non-reproducible**: corrupt output as shipped (staging race), and 4.05 vs 27.08 tok/s at 54% pool-size-invariant hit once corrected — the same reactive-routing wall on an independent implementation. See `docs/moe-track-plan.md` before proposing MoE work.

Track 2 (dense RAM-resident partial offload) has its **baseline**: dense Qwen3-32B Q4_K_M peaks at ~2.7 generation tok/s (ngl 22) — ~12× slower than the Track-1 MoE at similar total size, confirming sparse activation beats dense weights by an order of magnitude on this RAM-bound machine. See `docs/track2-dense-offload.md`.

Track 3 (low-bit/ternary) has its **baseline**: Q3_K_M is marginal (3.48 tok/s, +2% PPL), IQ2_XXS reaches 10.62 tok/s because 9 GB nearly fits VRAM (+30% PPL), and the official BitNet i2_s GGUF is incompatible with mainline b10355 (fork-only quant type 36). The MoE Q4_K_M remains the quality/speed frontier. See `docs/track3-low-bit.md`.

Track 4 (gpt-oss MXFP4, 100B-class path) is **open**: gpt-oss-20b validated MXFP4 on sm_86 (29.52 tok/s smoke) and re-measured the placement optimum on a second architecture — `-ngl 24 --n-cpu-moe 10` ≈ 44.8 tok/s, plateau K=8–12, same interior-optimum shape as Qwen3. The roofline model was refined: the ceiling is min(DDR4 bytes ÷ expert-bytes/token, CPU expert-kernel throughput) — gpt-oss all-CPU sits at 37% of its byte ceiling (kernel-bound), Qwen3 at ~90% (bandwidth-bound). gpt-oss-120b (59 GiB MXFP4, the only ≥100B Q4-class file fitting 64 GiB) is gated on the RAM upgrade; predicted 5–8 tok/s. See `docs/track4-gpt-oss.md`.

The project remains **measurement-first**. No custom inference engine is selected or planned at this point. Read these documents before proposing architecture changes:

- `docs/implementation-path-analysis.md` — comparative analysis of implementation paths (Python/PyTorch vs llama.cpp fork vs hybrid) and the provisional work sequence.
- `docs/system-characterization.md` — what the B01–B06 system benchmarks measure, results so far, and interpretation rules.
- `docs/model-selection.md` — the Qwen3-8B baseline decision (Q09), candidates considered, and derived layer byte sizes for the overlap experiment.
- `docs/archive-dense-streaming.md` — closure rationale and preserved evidence for the archived first track.
- `docs/moe-track-plan.md` — Track 1 closure: expert-locality measurements, negative cache PoC, best MoE placement, FATE-fork claim non-reproduction.
- `docs/track2-dense-offload.md` — Track 2 baseline: dense Qwen3-32B sweep and the dense-vs-MoE comparison.
- `docs/track3-low-bit.md` — Track 3 baseline: low-bit quants, BitNet compatibility boundary, quality/speed frontier.
- `docs/track4-gpt-oss.md` — Track 4 open: gpt-oss MXFP4 validation, second-architecture roofline, 120b plan.
- `benchmarks/system/README.md` — how to run the characterization suite.

The working code includes preserved system characterization, OVR, and RLS artifacts. They are evidence, not the active implementation path. The 7B-class control model was **Qwen3-8B**; the measured active models are now **Qwen3-30B-A3B** (MoE, Track 1) and **Qwen3-32B** (dense, Track 2).

## Repository layout

```
benchmarks/system/     System characterization scripts
  run_all.py           Runs every characterization script in order
  collect_system_info.py  Environment snapshot -> results/system/system-info.json
  benchmark_storage.py    B01/B02: Windows direct-I/O sequential and random reads
  benchmark_ram.py        B03: host RAM memcpy bandwidth (NumPy)
  benchmark_cuda.py       B04–B06: pageable/pinned H2D and FP16 GEMM (PyTorch CUDA)
  plot_results.py         Renders plots/system_memory_hierarchy.png with Pillow
  common.py               Shared CSV/JSON result-writing helpers
docs/                  Research reports (implementation-path analysis, system characterization)
results/system/        Raw per-repetition measurements and the environment snapshot
results/               system_characterization.csv (medians, ranges, standard deviations)
plots/                 Generated figures (measured values only)
experiments/overlap/   Synthetic layer-streaming experiment (fixture, direct-I/O staging,
                       variants.py, pipeline.py, README.md)
experiments/real_layer/ Real Qwen3-8B layer package, streamed execution, and correctness check
experiments/moe_trace/  MoE expert-locality track: run_traces.py (10-prompt capture),
                        analyze_locality.py (frequency/overlap/reuse/LRU), parity_check.py,
                        cache_cost_model.py (hit curves + GGUF bytes + measured BW -> PoC sizing),
                        measure_cache_poc.py (cache on/off perf comparison),
                        placement_grid.py (decoupled -ngl x --n-cpu-moe sweep),
                        prefetch_simulation.py (predictor hit-rate sim),
                        speculative_bench.py (llama-server draft vs no-draft),
                        fate_repro.py (FATE-fork claim check; --reparse rebuilds JSON from raw logs)
experiments/dense_offload/ Track 2: placement_sweep.py (dense Qwen3-32B -ngl sweep)
experiments/low_bit/    Track 3: bench_quants.py, perplexity_quants.py
experiments/gpt_oss/    Track 4: placement_grid.py (gpt-oss-20b -ngl x --n-cpu-moe; --refine)
results/track2-dense/   Dense 32B placement sweep results
results/track3-low-bit/ Low-bit bench + perplexity results and the fixed corpus
results/gpt-oss/        gpt-oss-20b placement grid + refinement results
models/                 Local checkpoints (Qwen3-8B, Qwen3-30B-A3B, Qwen3-32B, BitNet, gpt-oss-20b)
results/moe-locality/   Raw expert traces (trace-*.jsonl), generated text, parity outputs,
                        locality-analysis.json, cache-cost-model.json, moe-cache-poc.json,
                        cache-stats-*.json, moe-cache-poc.patch + moe-cache-new-files/,
                        fate-repro/ (FATE-fork matrix, per-rep logs, fork-patches.patch)
tools/build-scripts/    Native build helpers (configure-trace-build.bat, build-trace.bat,
                        build-fate.bat, kill-stale-llama.ps1)
tools/llama.cpp-source/ llama.cpp pinned at dd1ea52 (b10355) + additive moe-trace example,
                        moe-cache PoC (env-gated, LLAMA_MOE_CACHE unset = stock behavior)
tools/llama.cpp-b10355/ Unmodified b10355 release binaries (baseline + parity reference)
tools/llama-moe-cache/  Gitignored FATE-fork clone (commit 77c8767, AGPL-3.0) for the claim
                        check; local patches at results/moe-locality/fate-repro/fork-patches.patch;
                        rebuild with MSYS2_ARG_CONV_EXCL="*" cmd /c "tools\build-scripts\build-fate.bat"
tests/                 Pytest suite: fixture, rows, timeline, checksums, Windows I/O adapter
src/                   Empty placeholder for future work
```

## Technology stack and environment

- Pure Python scripts; there is **no package manifest** (no `pyproject.toml`, `requirements.txt`, etc.) and no build system. Scripts run directly with the active Python interpreter.
- Target host: Windows 11, NVIDIA GeForce RTX 3070 Ti (8 GiB VRAM, compute capability 8.6), Python 3.14. Scripts are invoked from the repository root.
- A project-local virtual environment `.venv` (Python 3.14, torch 2.13.0+cu126, NumPy, Pillow, Transformers, Safetensors, Hugging Face Hub, hf_xet, gguf) holds Python experiment dependencies. The PyPI Windows wheel for torch is CPU-only — the CUDA build comes from `https://download.pytorch.org/whl/cu126`.
- Native trace-build prerequisites were installed on 2026-08-11: CMake 4.4.2, Visual Studio 2022 Build Tools (MSVC 19.44), and CUDA Toolkit 13.3 (`nvcc` 13.3.73). The llama.cpp source checkout in `tools/llama.cpp-source/` is pinned to `dd1ea5243` (tag b10355); its only modifications are the additive `examples/moe-trace/` executable and one `add_subdirectory` line. Rebuild with `MSYS2_ARG_CONV_EXCL="*" cmd /c "tools\build-scripts\build-trace.bat"` (Ninja + vcvars64 + CUDA 13.3; the VS generator fails to resolve CudaToolkitDir). Do not silently move the checkout to a different revision.
- Dependencies: NumPy (required by `benchmark_ram.py`) and Pillow (required by `plot_results.py`). PyTorch with CUDA is an **optional** dependency of `benchmark_cuda.py`; when absent, B04–B06 are recorded as *skipped* rows rather than fabricated. Run `benchmark_cuda.py` and `plot_results.py` with `.venv\Scripts\python.exe`.
- `benchmark_storage.py` is **Windows-only**: it uses `ctypes` against `kernel32` (`CreateFileW` with `FILE_FLAG_NO_BUFFERING`, `VirtualAlloc`, `ReadFile`) for direct I/O. Do not "port" it silently; POSIX support would be a new design decision.
- The repository is not currently under version control; `.venv` and other local artifacts live inside the project tree.

## Build and run commands

There is nothing to build. Run everything from the repository root:

```powershell
python benchmarks/system/run_all.py            # full suite (creates a temporary 2 GiB fixture)
```

Individual scripts and smoke tests:

```powershell
python benchmarks/system/benchmark_storage.py --file-size-mib 512 --random-total-mib 64 --repetitions 3
python benchmarks/system/benchmark_ram.py --size-mib 256 --repetitions 3
.venv\Scripts\python.exe benchmarks/system/benchmark_cuda.py --transfer-mib 128 --transfer-repetitions 3 --matmul-repetitions 5
.venv\Scripts\python.exe benchmarks/system/plot_results.py

# Checksum-verified synthetic overlap benchmark (temporary 13.94 GB fixture by default)
.venv\Scripts\python.exe experiments/overlap/pipeline.py --sizes fp16 int4 --blocks 36 --repetitions 3

# Real Qwen3-8B layer experiment (requires local checkpoint in models/Qwen3-8B)
.venv\Scripts\python.exe experiments/real_layer/qwen_layer_stream.py --model-dir models/Qwen3-8B --layer 0 --repetitions 3

# MoE expert-locality track (trace + cache PoC done; see docs/moe-track-plan.md)
.venv\Scripts\python.exe experiments/moe_trace/parity_check.py      # output parity vs b10355
.venv\Scripts\python.exe experiments/moe_trace/run_traces.py        # 10-prompt trace capture
.venv\Scripts\python.exe experiments/moe_trace/analyze_locality.py  # locality report
.venv\Scripts\python.exe experiments/moe_trace/measure_cache_poc.py # cache on/off perf (slow)
```

Notes on running benchmarks:

- The default storage run creates a temporary 2 GiB fixture (deleted afterward unless `--keep-file` is passed) and performs five direct-I/O repetitions. Point `--file` at a path on the volume being characterized.
- Close storage-heavy applications and keep the machine on AC power. Record power plan, background workload, and thermal state alongside serious comparison runs; do not compare runs made under different GPU power policies without recording that fact.
- All sizes and offsets must be 4096-byte aligned (enforced by the storage script).
- b10355 `llama-cli` is unusable from scripts: it stays in an interactive `> ` input loop even with `-no-cnv` and stdin closed. Use `llama-completion.exe -no-cnv` for non-interactive reference runs. If a llama process is orphaned (it holds several GiB of VRAM), kill all of them with `powershell -NoProfile -ExecutionPolicy Bypass -File tools/build-scripts/kill-stale-llama.ps1`.

## Testing

`tests/test_overlap_logic.py` covers fixture invariants, row validation, timeline writing, checksum identification, and the Windows direct-I/O staging adapter; it requires no CUDA work. `tests/conftest.py` puts the repo root and `experiments/overlap/` on `sys.path`, and experiment modules import `bootstrap` to reach `benchmarks/system/common.py`. Run with `.venv\Scripts\python.exe -m pytest tests -v`. The benchmarks remain the verification mechanism for hardware claims: each script appends raw repetitions to `results/system/raw_measurements.csv` and rebuilds `results/system_characterization.csv` via `common.rebuild_summary()`. The raw CSV is the full history (including old `skipped` and `error` rows); the summary drops a benchmark's skip records once that benchmark has measured rows.

## Conventions and measurement discipline

These rules come from the project's own documents; follow them when changing benchmarks or interpreting results:

- **Record, don't invent.** A benchmark whose dependencies are missing must write a `status="skipped"` row with the reason — never estimate or fabricate a value. Skipped means missing evidence, not zero bandwidth.
- **Keep every repetition.** Raw rows go to `results/system/raw_measurements.csv` (append); the summary CSV reports medians plus min/max and standard deviation. Never select the fastest repetition.
- **Payload vs physical traffic.** `GB/s_payload` is useful bytes divided by elapsed time. RAM copy moves roughly 2× the payload physically (one read plus one write). Keep this distinction in notes and interpretation.
- **Plots contain measured values only**; missing tests are labeled as not measured, never extrapolated.
- **Direct I/O only for storage numbers.** B01/B02 use `FILE_FLAG_NO_BUFFERING` so results are not silently filesystem-cache numbers. The current reader uses queue depth 1; overlapped-I/O benchmarks are a deliberate later step.
- **A high standalone bandwidth does not imply pipeline overlap.** Overlap claims require end-to-end makespan plus per-stage CUDA/I/O event timing.

## Code style

- Standard-library-first Python; third-party imports (NumPy, Pillow, PyTorch) are used only where the script genuinely needs them, and optional ones are imported lazily with graceful skip behavior (see `benchmark_cuda.py`).
- Type hints on function signatures, `from __future__ import annotations`, module docstring stating purpose on the first line, `argparse` CLIs with MiB/KiB-suffixed flags, `pathlib.Path` for paths, `if __name__ == "__main__": main()` entry points.
- Shared output logic lives in `benchmarks/system/common.py`; new benchmarks should reuse `write_rows`, `skipped_row`, `rebuild_summary`, and the fixed CSV field list rather than inventing a new output format.
- Documentation and comments are written in English, in a measured, evidence-first tone: state what is measured versus hypothesized, cite sources for external claims, and mark provisional decisions as provisional.

## Security and operational considerations

- Benchmarks write only inside `results/` (and the temporary storage fixture). The storage fixture can be up to 2 GiB by default — ensure free disk space before full runs.
- `benchmark_storage.py` and `collect_system_info.py` call Windows APIs via `ctypes`; treat changes to handle/buffer management carefully (leaked handles or misaligned sizes fail loudly, which is intended).
- `collect_system_info.py` shells out to `nvidia-smi` and `nvcc`; missing tools are tolerated and recorded as `null`, not errors.
- No secrets, credentials, or network access are involved anywhere in the current codebase.
