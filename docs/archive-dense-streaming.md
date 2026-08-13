# Archive decision: dense NVMe weight streaming

Status: archived on 2026-08-11. This document closes the first research track; it does not claim a fundamental impossibility result.

## Question tested

Can an explicit `NVMe -> RAM -> pinned host memory -> VRAM -> GPU` pipeline make dense transformer models larger than 8 GiB VRAM practically executable on this RTX 3070 Ti system?

## What was demonstrated

- Windows unbuffered/direct reads, pinned staging, asynchronous H2D copies, CUDA events, and buffer-lifetime safety can be exercised from Python.
- A synthetic Qwen3-8B-sized pipeline reduced makespan through scheduling, but remained storage-bound.
- A real Qwen3-8B BF16 layer package produced exact hidden-state agreement with an independently loaded official Transformers layer after direct I/O, staging, and H2D.
- A three-real-layer reference comparison also passed exactly. Its corrected storage-worker schedule did not beat serial execution meaningfully.
- Unmodified llama.cpp b10355 with Qwen3-8B Q4_K_M is a practical fully GPU-resident baseline on this same machine.

## Why the track stops here

The measured real BF16 layer read was about 0.570 s, while the corresponding one-token layer compute was about 0.003 s. Storage therefore cannot be hidden by compute at this granularity. Better storage, asynchronous queue depth, and native staging may improve the constant, but the current evidence does not justify building an entire dense streamed-inference engine before testing alternatives with better locality.

The result applies to this hardware, driver, direct-I/O implementation, block size, and decode-shaped workload. It does not prove that all storage-backed inference is impractical or that future faster hardware cannot alter the boundary.

## Preserved artifacts

- `benchmarks/system/`: B01--B06 host characterization.
- `experiments/overlap/`: synthetic checksum-verified scheduling experiment.
- `experiments/real_layer/`: real Qwen3 BF16 layer correctness and timing experiments.
- `results/system/raw_measurements.csv`: full repetition history, including instrumentation errors and corrected runs.
- `results/llama-bench-b10355-q4km.json`: pinned llama.cpp baseline output.

## Successor research tracks

1. **MoE expert locality:** total parameters can exceed GPU memory while each token activates only a subset. Measure routing locality and cache policies before designing expert streaming.
2. **RAM-resident partial offload:** measure llama.cpp CPU/GPU placement for models that exceed VRAM but fit in current RAM; revisit after a 64 GiB upgrade if one occurs.
3. **Low-bit models:** measure actual quality, compatibility, VRAM use, and token rates for models designed for aggressive quantization rather than treating storage as a substitute for memory.
