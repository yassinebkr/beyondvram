# Archived: layer-streaming overlap experiment (OVR)

OVR is an archived overlap gate for the former path-C dense-streaming investigation. It measures a synthetic, checksum-verified path through NVMe -> aligned RAM buffer -> pinned host tensor -> VRAM -> GPU compute using Qwen3-8B-sized blocks. It is not transformer inference and does not make claims about real-model token throughput. The overall dense-streaming decision is recorded in `docs/archive-dense-streaming.md`.

## Design

The fixture contains 4096-byte cells. Cell `c` contains only byte `c % 251`, so the expected sum of any aligned block is computed analytically. After each GPU compute operation, the experiment sums the transferred device block and compares it with that expected value. This catches wrong-block and torn-buffer reuse errors. It does not detect every possible corruption that preserves a byte sum.

There are three schedules:

- `sync`: direct read, RAM-to-pinned staging copy, H2D, then GEMM for each block.
- `double_buffer`: storage remains synchronous while H2D(N+1) shares a copy stream with compute(N). A pinned slot is reused only after its previous H2D finishes; a device slot is reused only after its previous compute finishes.
- `three_stage`: a worker reads storage(N+2), the copy stream transfers H2D(N+1), and the default stream computes(N). The three-slot pinned ring waits for its previous H2D; the two-slot device ring waits for its previous compute.

The direct-I/O reader has an aligned kernel buffer, so copying that buffer into a pinned tensor is deliberately explicit and separately timed as `stage_copy`. A native direct-to-pinned reader is a possible later path-C component only if this measured cost warrants it.

## Measured canonical result - 2026-08-11

Canonical configuration: 36 blocks, 4096-by-4096 FP16 GEMM per block, three repetitions, RTX 3070 Ti in WDDM desktop use. The raw rows and per-block timeline are the evidence of record.

| Block size | Variant | Median makespan | Payload throughput | Storage + staging-copy | Interpretation |
|---|---|---:|---:|---:|---|
| FP16, 387,203,072 B | sync | 18.487 s | 0.754 GB/s | 16.236 s | stage sum agrees with serial execution |
| FP16, 387,203,072 B | double_buffer | 16.324 s | 0.854 GB/s | 16.199 s | most H2D and compute are hidden |
| FP16, 387,203,072 B | three_stage | 16.311 s | 0.855 GB/s | 16.196 s | essentially tied with double buffering |
| INT4 proxy, 118,489,088 B | sync | 2.890 s | 1.476 GB/s | 2.095 s | serial stage overhead remains visible |
| INT4 proxy, 118,489,088 B | double_buffer | 2.141 s | 1.992 GB/s | 2.093 s | most H2D and compute are hidden |
| INT4 proxy, 118,489,088 B | three_stage | 2.130 s | 2.003 GB/s | 2.097 s | marginally faster than double buffering |

The FP16 staging copy median was 0.775 s per 36-block pass (about 21.5 ms/block); the INT4 proxy was 0.244 s (about 6.8 ms/block). This is a measured cost on the current Python/NumPy staging implementation, not a lower bound for native I/O.

## Run

Run from the repository root with the CUDA-enabled virtual environment:

```powershell
# Six-block smoke test; exercises three-slot pinned-buffer recycling.
.venv\Scripts\python.exe experiments\overlap\pipeline.py --sizes fp16 --blocks 6 --repetitions 1 --matmul-size 1024

# Canonical run.
.venv\Scripts\python.exe experiments\overlap\pipeline.py --sizes fp16 int4 --blocks 36 --repetitions 3
```

The fixture is deleted by default. Pass `--keep-file` to retain it. Results append to `results/system/raw_measurements.csv`, are summarized in `results/system_characterization.csv`, and add per-block stage durations to `results/system/overlap_timeline.csv`. The workload label includes blocks and matrix size so smoke runs do not contaminate canonical summaries.

## Confounders and boundaries

- The GPU runs under WDDM with desktop applications active; scheduling and clock behavior can vary.
- Storage has one direct-I/O reader and queue depth one. This tests the stated baseline, not asynchronous multi-request NVMe capability.
- The first run after fixture creation may differ from later runs even with direct I/O; retain all repetitions.
- The GEMM is synthetic and does not model attention, activation transfer, KV cache, allocator behavior, or real quantized kernels.
- Per-block durations are a timeline CSV, not a Chrome/Nsight trace. A profiler trace is a later diagnostic if these measured timings cease to explain a result.
