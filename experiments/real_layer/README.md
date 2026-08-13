# Archived: real Qwen3-8B layer streaming experiment (RLS)

RLS is the archived real-weight correctness experiment from the dense NVMe-streaming track. It is deliberately one decoder layer, not an inference engine and not a token-throughput benchmark. Its conclusion is summarized in `docs/archive-dense-streaming.md`.

## What it tests

The harness reads only `model.layers.0.*` from the official BF16 Safetensors checkpoint and creates a derived, aligned package. Each tensor begins at a 4096-byte offset. For every repetition it:

```text
derived layer package --Windows direct I/O--> aligned RAM buffer
  -> explicit NumPy copy -> pinned uint8 tensor -> non-blocking H2D
  -> one GPU byte buffer -> views assigned to the official Qwen3DecoderLayer
  -> official decoder-layer forward
```

The reference is a separate `Qwen3DecoderLayer` with the same tensors loaded directly from Safetensors. Both receive the same deterministic BF16 hidden state and RoPE positions. The output check is exact (`max_abs_error == 0`), which is appropriate because the same BF16 bytes and official operations are used. It validates the layer’s hidden state, not final logits or generated text.

The package is a test artifact, not a proposed final model format. It makes the storage granularity and alignment explicit, whereas Safetensors shards interleave unrelated model tensors. A later experiment must compare this derived package with direct original-format access before any storage-format decision.

## Measured layer-0 result — 2026-08-11

Configuration: Qwen3-8B BF16 layer 0, 387,203,072-byte payload, 387,203,072-byte aligned package, one-token decode-shaped input, RTX 3070 Ti under WDDM. Five retained successful repetitions (a smoke, a three-repetition run, and final verification) are summarized below; raw rows are the evidence of record.

| Stage | Median | Interpretation |
|---|---:|---|
| Direct read | 0.570 s | 0.679 GB/s payload for this 369 MiB package; considerably below B01’s 16 MiB-chunk sequential result, so the one-read behavior needs investigation rather than extrapolation. |
| RAM to pinned staging | 0.0212 s | Explicit Python/NumPy copy; this is visible engineering overhead. |
| Pinned H2D | 0.0147 s | Consistent in scale with B05; not the bottleneck. |
| Qwen3 decoder-layer compute | 0.0031 s | One-token, no-KV-cache execution only; it does not represent full prompt prefill or a production attention kernel. |
| Streamed/reference maximum absolute error | 0 | Correctness gate passed for every retained run. |

For this exact un-overlapped decode-shaped experiment, direct storage plus staging is far larger than H2D and compute. That supports—not proves—the synthetic experiment’s storage-bound diagnosis. It is not a statement about full-model token rate: embeddings, all 36 layers, KV cache, output projection, scheduling overlap, quantization, and allocator behavior are still outside this experiment.

## Run

The checkpoint must already be available locally. Package a layer once:

```powershell
.venv\Scripts\python.exe experiments\real_layer\qwen_layer_stream.py `
  --model-dir models\Qwen3-8B --package-dir results\real_layer\layer00 --layer 0 --pack-only
```

Then measure it:

```powershell
.venv\Scripts\python.exe experiments\real_layer\qwen_layer_stream.py `
  --model-dir models\Qwen3-8B --package-dir results\real_layer\layer00 --layer 0 --repetitions 3
```

Rows use benchmark id `RLS` in `results/system/raw_measurements.csv` and are summarized in `results/system_characterization.csv`. The checkpoint and derived package are retained deliberately so the measured run is reproducible.

## Three-layer scheduler follow-up — 2026-08-11

`three_layer_pipeline.py` packages and executes layers 0--2 in sequence, retains GPU hidden states, and compares the final output with a normal three-layer reference. The corrected `scheduler=v2` run retained three repetitions per variant; both variants had exact final hidden-state agreement. Median serial makespan was 1.839 s and the storage-worker variant was 1.843 s. This is within short-run variation, so it does **not** demonstrate a useful overlap speedup. Direct reads remained dominant.

The initial smoke included module construction inside its timed scope. Its raw rows are preserved under the earlier workload label; `scheduler=v2` is the corrected comparison. Run it with `.venv\Scripts\python.exe experiments\real_layer\three_layer_pipeline.py --model-dir models\Qwen3-8B --repetitions 3`.

## llama.cpp measured baseline — 2026-08-11

Unmodified llama.cpp b10355 (`dd1ea5243`) with the official CUDA 13.3 runtime bundle detected the RTX 3070 Ti. With Qwen’s official Qwen3-8B Q4_K_M GGUF (5,021,827,072 bytes), all layers requested on GPU, and mmap loading, `llama-bench` measured 2516 tokens/s for a 128-token prompt (high variation) and 85.33 tokens/s for 32-token generation (three samples). Raw JSON: `results/llama-bench-b10355-q4km.json`. This is a fully GPU-resident quantized baseline, not a streamed-model result.

## llama.cpp source inventory — inspected 2026-08-11

The current llama.cpp loader parses GGUF metadata and tensor locations, supports split files and quantization types, and supplies tensor data through model-loading/backend allocation machinery. Its backend model is persistent placement in backend buffers, not an exposed policy that evicts every dense decoder-layer weight after each token. This is why a true streamed-residency fork would cross loader, allocation, scheduler, and copy-lifetime code rather than being a single loading flag.

Useful existing pieces remain substantial: GGUF tensor offsets/metadata, quantization layouts and CUDA kernels, CPU/GPU placement, memory mapping/direct-I/O modes, scheduler infrastructure, and a trusted decoder baseline. The next llama.cpp step is therefore to pin a revision, build an unmodified CUDA baseline, and trace one block’s allocation and graph path—not to fork it yet. Source reviewed: [loader](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp), [backend scheduler/allocation implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp), and [CUDA backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu).
