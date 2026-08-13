# Implementation-path analysis

Status: archived dense-streaming research report, updated 2026-08-11. It remains a record of the alternatives considered, not a current implementation plan. See `docs/archive-dense-streaming.md`.

## Executive conclusion

The first phase followed path C: a small explicit Python/PyTorch correctness harness plus llama.cpp as a measured baseline. Its evidence does **not** justify a dense NVMe-streaming runtime or a llama.cpp fork on this machine. The successor track is MoE locality, RAM-resident partial offload, then low-bit models.

This is not a conclusion that Python will be the production runtime, that GGUF will be the final format, or that custom storage is necessary. It is a sequencing decision intended to maximize what each experiment teaches while retaining a credible route to low-level control.

## What source research can answer

PyTorch already exposes pinned host allocations, non-blocking tensor copies, CUDA events, and user-controlled streams. NVIDIA specifies an important boundary: CPU/GPU copies need page-locked host buffers to be genuinely asynchronous, and overlap additionally depends on device copy-engine support and the absence of accidental synchronization ([CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html), [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)). This makes PyTorch suitable for testing residency schedules and overlap semantics without writing a CUDA allocator first.

llama.cpp is more than a reference decoder. It already provides GGUF metadata and tensor offsets, many low-bit quantizations, CUDA kernels, CPU/GPU hybrid execution, memory mapping, a direct-I/O load mode, and a backend scheduler. Its current loader explicitly distinguishes mmap and direct I/O, builds an index of tensor locations, and checks whether weights are supported by a target backend ([loader source](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp), [GGUF layout](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h)). Its project scope explicitly includes quantized CUDA inference and partial GPU acceleration for models larger than VRAM ([llama.cpp README](https://github.com/ggml-org/llama.cpp)). These are strong reasons not to reimplement format parsing, quantization kernels, or baseline inference merely for novelty.

What the existing runtime does not establish is the proposed dense-model policy: evicting and reloading transient weight blocks every decoding step under an explicitly controlled NVMe → RAM → pinned staging → VRAM pipeline. mmap provides demand paging, not an application-controlled cache policy. Existing CPU/GPU offload normally assigns tensors to persistent backend buffers; changing that lifecycle cuts across loading, graph scheduling, allocation, and CUDA execution. That is possible in a fork, but it is not a small first experiment.

## What only experiments on this machine can answer

The following remain measured questions, not architectural facts:

- sequential and chunked direct-read bandwidth of the workspace NVMe;
- latency and throughput as access granularity approaches a tensor or layer;
- effective single-buffer and pipelined pageable/pinned H2D bandwidth;
- whether storage, H2D, and the target GEMMs overlap on this Windows driver and GPU;
- usable VRAM after the display/desktop, CUDA context, allocator, KV cache, and workspaces;
- RAM-copy cost, page-cache behavior, and the memory-pressure cliff near 32 GiB;
- whether Python dispatch and allocator overhead matter at layer-sized granularity;
- whether a 64 GiB upgrade changes the best policy, rather than merely increasing cache hit rate.

## Comparative evaluation

Scores are qualitative hypotheses: 5 is favorable for this research phase. They are not benchmark results.

| Criterion | A: Python/PyTorch | B: llama.cpp fork | C: hybrid |
|---|---:|---:|---:|
| Startup complexity | 5 | 2 | 4 |
| Learning transformer internals | 5 | 3 | 5 |
| Control over layer residency | 4 | 4 | 5 |
| Control over NVMe reads | 3 | 5 | 5 |
| Control over RAM caching | 3 | 5 | 5 |
| Pinned-memory support | 5 | 5 | 5 |
| Asynchronous H2D support | 5 | 5 | 5 |
| CUDA stream control | 5 | 4 | 5 |
| Quantized-weight compatibility | 2 | 5 | 4 |
| Reference validation | 5 | 4 | 5 |
| Expected steady-state overhead | 2 | 5 | 4 |
| Eventually exceeding RAM | 3 | 5 | 5 |

### A — Python/PyTorch

Advantages: the forward pass can be written in the same block structure used in papers; intermediate hidden states and logits are easy to compare; hooks and profilers shorten the correctness loop; pinned tensors, streams, and events are accessible; storage policies can remain plain, inspectable code. Safetensors supports named tensor access and slices and is a reasonable experimental input format ([Safetensors documentation](https://huggingface.co/docs/safetensors/index)).

Disadvantages: standard module/device APIs are designed around persistent tensors, not continual weight rematerialization. General low-bit weights often depend on third-party kernels and layouts. Python object overhead, the GIL around orchestration, caching allocators, and framework-inserted conversions can obscure the path being measured. Stock file APIs do not guarantee Windows direct I/O or explicit cache eviction.

Engineering consequence: use A for correctness and schedule experiments, not as evidence that a Python production engine is fast enough. Make transfers explicit and instrument every boundary. Avoid high-level automatic device maps in the core experiment because they hide residency decisions.

### B — llama.cpp fork

Advantages: mature GGUF and quantization support, efficient CPU and CUDA kernels, a known-good decoder, existing CPU/GPU placement, tensor metadata with file offsets, mmap/direct-I/O loader modes, and benchmark tools. It offers the shortest route to testing streamed quantized weights in an optimized runtime once the desired policy is known.

Disadvantages: startup learning cost is substantially higher. Weight lifetime is connected to backend buffer allocation and graph scheduling, so a true transient-residency design is not a localized loader patch. Debugging numerical differences is less interactive. Carrying a long-lived fork has a continuing merge and test burden. Current upstream behavior can also change, so experiments must pin a commit rather than cite `master` alone.

Engineering consequence: compile and benchmark an unmodified, pinned llama.cpp revision as a baseline. Before any fork, trace one transformer block from GGUF offset through backend allocation, graph construction, scheduler splits, tensor copies, and kernel launch. Only fork after a minimal scheduling experiment identifies a capability that cannot be tested cleanly from Python or an extension.

### C — hybrid

Advantages: preserves Python's transparent correctness loop while permitting direct Windows I/O, registered/pinned ring buffers, custom packed formats, and CUDA kernels in focused native components. A single storage/transfer component can be benchmarked independently before it is coupled to transformer execution. Reference logits can come from an ordinary framework implementation and llama.cpp.

Disadvantages: two language/tooling layers, ABI and build complexity, duplicated tensor metadata unless interfaces are disciplined, and a risk that the experimental harness and native component disagree about ownership or synchronization. Poorly chosen boundaries can introduce extra copies.

Engineering consequence: keep the interface narrow: tensor/block identifier and byte range in; a lifetime-tracked device buffer/event out. Python owns the experiment and validation initially. Native code owns only direct I/O, pinned buffer pools, or kernels after evidence shows they are needed.

## Storage and quantization are not yet decisions

GGUF is the best baseline for llama.cpp compatibility and quantized experiments. Safetensors is the clearest baseline for inspecting conventional model tensors in Python. Both already store tensor metadata plus contiguous byte regions, which may be enough for layer-oriented access. A custom layer container is justified only if traces show material losses from alignment, excessive seeks, tensor ordering, decompression, or metadata overhead.

Start correctness validation with the model's reference floating-point representation (FP16 or BF16 as supported) and a tiny input, even if the full representation must stream. Then add one widely supported 4-bit GGUF quantization as a performance/footprint point. INT8 can be added if it answers a specific accuracy or bandwidth question. Comparing formats is more informative than declaring one winner before I/O and kernel measurements.

## Conceptual architectures

Normal inference (conceptual; not a measured implementation):

```text
model file -> load once -> RAM and/or VRAM resident weights
                                  |
tokens -> embeddings -> block 0 -> block 1 -> ... -> logits
                 activations and KV cache persist across the pass
```

Candidate streamed inference (conceptual; not implemented):

```text
NVMe --read N+2--> RAM cache --stage N+1--> pinned ring --H2D--> VRAM slot
                                                                  |
activations + KV cache --------------------------------------> block N compute
```

The arrows are hypotheses about ownership and timing, not proof of overlap. Pinned memory is a scarce staging resource, not a synonym for the whole RAM cache. The model may ultimately favor grouped layers, sublayers, or persistent attention/embedding/output tensors.

## Provisional work sequence

1. Run B01–B06 and retain every repetition.
2. Use their results to choose realistic block sizes for a three-stage synthetic pipeline.
3. Build one explicit transformer-block correctness experiment in Python using ordinary reference weights. **Completed for Qwen3-8B layer 0 on 2026-08-11:** `experiments/real_layer/` streams a 387,203,072-byte BF16 layer package through Windows direct I/O, pinned staging, and H2D, then exactly matches the official Transformers layer output. See its README for retained timings and boundaries.
4. Measure synchronous load → H2D → compute, then double-buffering; verify overlap with CUDA events and a profiler trace.
5. Benchmark an unmodified pinned llama.cpp revision and inventory which loader/backend pieces can be reused.
6. Revisit the A/B/C decision with measured bottlenecks and engineering cost.

## Open-question status after research

Q01–Q03 and Q05 remain open; path C is the provisional sequence, not a frozen runtime. Q04 is partially answered: llama.cpp already supplies valuable format, quantization, backend, offload, mmap/direct-I/O, and reference-inference machinery, but its exact reuse boundary needs a pinned-source trace. Q06–Q08 remain open; the initial experiments should deliberately retain both Safetensors/reference floating point and GGUF/quantized comparison points. Q09–Q15 require model selection work or measurements and are not silently resolved here.
