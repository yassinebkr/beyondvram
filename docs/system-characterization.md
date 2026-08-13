# System characterization

Status: B01–B06 measured; dense-streaming track archived 2026-08-11. These measurements remain inputs to successor experiments.

## Purpose

These tests characterize the memory hierarchy that a streamed inference design would actually use. They do not benchmark a model and they do not prove that stages overlap. Raw repetitions live in `results/system/raw_measurements.csv`; medians and variation are in `results/system_characterization.csv`. The plot contains measured values only.

## Known environment snapshot

Initial inspection detected Windows 11, an NVIDIA GeForce RTX 3070 Ti with 8192 MiB reported VRAM, NVIDIA driver 610.74 (CUDA UMD 13.3), compute capability 8.6, and Python 3.14.4. B01–B03 were measured with the system Python (NumPy 2.4.4) before PyTorch existed on this machine. B04–B06 were measured from a project-local virtual environment (`.venv`, Python 3.14.4) containing PyTorch 2.13.0+cu126 and NumPy 2.5.2 — note that the PyPI Windows wheel for torch 2.13 is CPU-only, so the CUDA build was installed from `https://download.pytorch.org/whl/cu126`. No CUDA toolkit compiler (`nvcc`) is installed; none of the current scripts need one. At the B04–B06 capture `nvidia-smi` reported 6553 MiB free with ordinary desktop applications resident on the GPU; that is a transient observation, not a capacity guarantee. The reproducible metadata capture is `collect_system_info.py`; its JSON output is authoritative for the benchmark run.

## Test matrix

| ID | Measurement | Method | Key confounders |
|---|---|---|---|
| B01 | NVMe sequential reads | aligned Windows unbuffered reads, full fixture, 5 repetitions | volume/device identity, thermal state, other I/O |
| B02 | random/chunked reads | deterministic random offsets at 4 KiB–16 MiB, 5 repetitions | queue depth is 1; real pipeline may use overlapped I/O |
| B03 | RAM memcpy | pre-faulted NumPy copy, 7 repetitions | one thread/library implementation; payload vs bus traffic |
| B04 | pageable H2D | PyTorch copy with CUDA-event timing, 10 repetitions | pageable staging behavior is implementation-dependent |
| B05 | pinned H2D | pinned tensor and non-blocking copy, 10 repetitions | pinned allocation excluded; desktop GPU contention |
| B06 | GPU/VRAM | free/total memory plus repeated FP16 matrix multiply | synthetic GEMM is not a transformer layer |

## Results from this machine

These are medians from the canonical run, not expected specifications:

| ID | Workload | Median | Range | Repetitions |
|---|---|---:|---:|---:|
| B01 | 2 GiB direct sequential read, 16 MiB chunks | 2.526 GB/s | 2.500–2.561 GB/s | 5 |
| B02 | random direct read, 4 KiB chunks | 0.055 GB/s | 0.055–0.055 GB/s | 5 |
| B02 | random direct read, 64 KiB chunks | 0.531 GB/s | 0.521–0.537 GB/s | 5 |
| B02 | random direct read, 1 MiB chunks | 1.989 GB/s | 1.876–2.098 GB/s | 5 |
| B02 | random direct read, 16 MiB chunks | 2.473 GB/s | 1.916–3.085 GB/s | 5 |
| B03 | 512 MiB NumPy copy payload | 17.783 GB/s | 15.579–18.564 GB/s | 7 |
| B04 | 256 MiB pageable H2D copy | 11.104 GB/s | 10.402–12.272 GB/s | 10 |
| B05 | 256 MiB pinned H2D copy | 26.061 GB/s | 25.069–26.167 GB/s | 10 |
| B06 | VRAM free at capture (of 8 GiB) | 6.692 GiB | n=1, transient | 1 |
| B06 | 4096×4096 FP16 GEMM | 35.355 TFLOP/s | 27.414–41.825 TFLOP/s | 20 |

The B02 result is the first concrete architectural signal: fine-grained storage access is extremely costly at queue depth one, while layer-scale chunks approach the sequential rate. The 16 MiB result has a standard deviation of 0.470 GB/s and only 16 operations per repetition, so it is evidence for testing coarse reads—not yet a reliable bandwidth constant. B03 is roughly seven times B01 in payload terms, but this comparison does not include H2D and does not by itself determine the value of a RAM cache.

B05 adds the second signal: pinned H2D reaches 26.06 GB/s with low variance, roughly ten times the B01 sequential storage rate, so for weights streamed from NVMe the storage read stage — not the H2D stage — bounds throughput at these access sizes. An idle `nvidia-smi` query (P8 power state) showed the PCIe link at gen 1 ×16 against a gen 4 ×16 maximum; 26 GB/s is not achievable at gen 1, so the link evidently ramps up under transfer load, but link state during the actual repetitions was not recorded and stays an explicit metadata gap. B04 shows the pageable penalty directly: pageable copies reach only about 0.43× the pinned rate on this machine, consistent with the CUDA requirement for page-locked buffers for genuinely asynchronous transfers — the overlap experiment must therefore use pinned staging, as the path analysis assumed.

B06 is the weakest of the six as evidence. The FP16 GEMM median of 35.4 TFLOP/s comes with a standard deviation of 5.4 TFLOP/s and visibly oscillating repetitions (roughly 27 / 35 / 41 TFLOP/s); plausible confounders are clock ramping from the idle P8 state (255 MHz observed against a 2100 MHz maximum) and desktop GPU contention, but neither was tracked per repetition. It establishes that CUDA compute works and gives a coarse compute scale; it is not a transformer-layer measurement. The 6.69 GiB free-VRAM figure was captured with desktop applications resident and must not be treated as a stable inference budget.

Workload sizes are command-line inputs, not expected performance numbers. B01/B02 use `FILE_FLAG_NO_BUFFERING` so the reported storage numbers do not silently become filesystem-cache numbers. This first reader uses queue depth one; an overlapped-I/O benchmark comes later because concurrency and access size should be chosen from these baseline results.

## Reproduction

```powershell
python benchmarks/system/run_all.py
```

`run_all.py` invokes each script with whatever Python launches it; to include B04–B06, launch it with the project virtual environment (`.venv\Scripts\python.exe`) that holds the CUDA-enabled PyTorch build. For a short smoke test:

```powershell
python benchmarks/system/benchmark_storage.py --file-size-mib 512 --random-total-mib 64 --repetitions 3
python benchmarks/system/benchmark_ram.py --size-mib 256 --repetitions 3
.venv\Scripts\python.exe benchmarks/system/benchmark_cuda.py --transfer-mib 128 --transfer-repetitions 3 --matmul-repetitions 5
.venv\Scripts\python.exe benchmarks/system/plot_results.py
```

The storage fixture is deleted after a successful run unless `--keep-file` is given. Point `--file` at a path on the volume being characterized; the workspace path measures only its current volume. Record power plan, background workload, and any thermal throttling alongside serious comparison runs.

## Interpretation rules

- Report medians plus min/max and standard deviation; do not select the fastest repetition.
- Treat a skipped test as missing evidence, not zero bandwidth.
- Distinguish payload GB/s from physical traffic. RAM copy reads and writes approximately twice the payload.
- A high standalone bandwidth does not imply pipeline overlap. The next experiment must time end-to-end makespan and individual CUDA/I/O events.
- A 64 GiB RAM upgrade should be evaluated by rerunning the same suite and a controlled cache-capacity trace, not by extrapolating B03.

## Smallest useful follow-up experiment

Implemented as `experiments/overlap/` on 2026-08-11. It uses Qwen3-8B-sized FP16 blocks (387,203,072 bytes) and an INT4-sized proxy (118,489,088 bytes), direct I/O, explicit pinned staging, GPU checksum verification, and three schedules: serial, double-buffered, and three-stage. The canonical 36-block/three-repetition result is documented in `experiments/overlap/README.md`: the pipelined schedules reduced FP16 median makespan from 18.487 s to about 16.31 s and INT4 from 2.890 s to about 2.13 s, leaving storage plus the explicit staging copy as the practical bound. This is a measured synthetic-pipeline result, not a transformer inference claim.

## RLS follow-up

`experiments/real_layer/` now validates one real Qwen3-8B BF16 decoder layer through Windows direct I/O, pinned staging, H2D, and official-layer execution. The retained layer-0 runs had exact hidden-state agreement with an independently loaded reference layer. See that experiment's README for timings and boundaries.

## Current limitations

The exact physical NVMe model could not be queried through the restricted Windows management interface; it remains an explicit metadata gap rather than an assumption. PCIe link state during the B04/B05 repetitions was not recorded (only an idle gen 1 reading exists). B06 compute timings carry no per-repetition clock or contention record. OVR answers only the synthetic, queue-depth-one pipeline question under WDDM desktop contention; it does not validate real Qwen3 weights, quantized kernels, attention/KV-cache behavior, asynchronous NVMe queue depth, or bounded inference beyond system RAM.
