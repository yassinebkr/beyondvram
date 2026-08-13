project:
  name: consumer-hardware-streamed-llm
  stage: research-and-poc
  status: architecture-not-yet-frozen

mission: >
  Investigate, design, implement and benchmark a proof-of-concept inference
  system capable of executing transformer models that exceed available GPU VRAM,
  using an RTX 3070 Ti 8 GB, system DDR4 RAM and NVMe storage.

  Begin around the 7B scale so experiments are practical and correctness can
  be verified.

  The long-term research question is whether smarter memory hierarchy usage,
  streaming, caching, quantization, scheduling and asynchronous execution can
  substantially extend the model sizes practically executable on ordinary
  consumer hardware.

user_goal: >
  I want to learn this field from fundamentals while building a real system.
  Explain important decisions, measurements and failures rather than hiding
  complexity behind framework calls.

hardware:
  os: Windows 11
  gpu:
    model: NVIDIA RTX 3070 Ti
    vram_gb: 8
  ram:
    installed_gb: 32
    upgrade_possible: true
    possible_target_gb: 64
  storage:
    type: NVMe
    exact_model: unknown
    measured_bandwidth: unknown

known_constraints[4]:
  - GPU VRAM is limited to 8 GB.
  - System RAM is currently 32 GB.
  - Storage and RAM performance must be measured rather than assumed.
  - The final architecture must be based on experimental evidence.

open_questions[15]{id,question,status}:
  Q01,"Should the first implementation use Python/PyTorch, modify llama.cpp, or use another runtime?",OPEN
  Q02,"Would a Python prototype teach us enough before moving to C++/CUDA?",OPEN
  Q03,"Would modifying llama.cpp give a materially better starting architecture?",OPEN
  Q04,"Can llama.cpp already provide parts of the desired behavior that should not be reimplemented?",OPEN
  Q05,"Should the project eventually fork llama.cpp or remain an independent runtime?",OPEN
  Q06,"Should weights initially remain in GGUF, Safetensors, or another existing format?",OPEN
  Q07,"Is a custom layer-oriented storage format actually necessary?",OPEN
  Q08,"Should initial experiments use FP16, INT8, INT4, or multiple formats?",OPEN
  Q09,"Which specific 7B-class model gives the cleanest experimental baseline?",ANSWERED-WITH-USER-2026-08-10: Qwen3-8B (docs/model-selection.md)
  Q10,"What is the real sequential and random-read performance of the user's NVMe?",OPEN
  Q11,"What host-to-device PCIe bandwidth is achievable on this machine?",OPEN
  Q12,"How effectively can storage IO, H2D transfer and CUDA computation overlap?",OPEN
  Q13,"What should remain permanently resident in VRAM?",OPEN
  Q14,"What should remain cached in RAM versus streamed from storage?",OPEN
  Q15,"How much would increasing RAM from 32 GB to approximately 64 GB improve measured performance?",OPEN

important_rule: >
  Do not silently resolve OPEN questions. Investigate them.

decision_process:
  steps[6]:
    - Identify architectural alternatives.
    - Inspect relevant existing implementations and documentation.
    - Build minimal experiments where uncertainty cannot be resolved analytically.
    - Measure each alternative.
    - Record advantages, disadvantages and engineering cost.
    - Recommend a path with evidence before committing to major architecture.

candidate_implementation_paths[3]{id,name,description}:
  A,Python/PyTorch,"Fast experimental implementation using explicit layer movement, PyTorch CUDA streams and instrumentation."
  B,llama.cpp fork,"Modify llama.cpp loading/execution machinery to experiment with true streamed weight residency and GGUF."
  C,hybrid,"Use Python for experiments and instrumentation while implementing performance-critical storage or CUDA components in C++/CUDA."

first_task:
  objective: >
    Do NOT begin by implementing the entire inference engine.

    First create an engineering research report comparing candidate paths A, B
    and C specifically for this project and this hardware.

  investigate[12]:
    - startup complexity
    - ease of understanding transformer internals
    - control over layer residency
    - control over NVMe reads
    - control over RAM caching
    - pinned-memory support
    - asynchronous H2D support
    - CUDA stream control
    - compatibility with quantized weights
    - ability to validate against reference inference
    - expected runtime overhead
    - difficulty of eventually exceeding system RAM

  output:
    file: docs/implementation-path-analysis.md
    requirement: >
      End with a recommendation, but explicitly mark it as provisional until
      hardware microbenchmarks are completed.

second_task:
  objective: characterize-the-machine-before-designing-around-assumptions

  benchmarks[6]{id,test}:
    B01,NVMe sequential read bandwidth
    B02,NVMe random/chunked read behavior
    B03,RAM memcpy bandwidth
    B04,pageable RAM to VRAM bandwidth
    B05,pinned RAM to VRAM bandwidth
    B06,basic GPU compute and VRAM availability

  requirements[5]:
    - Produce reproducible scripts.
    - Record raw measurements.
    - Record hardware and software versions.
    - Run enough repetitions to expose variance.
    - Do not hardcode expected performance numbers.

  outputs[4]:
    - benchmarks/system/
    - results/system_characterization.csv
    - plots/system_memory_hierarchy.png
    - docs/system-characterization.md

research_principles[9]:
  - Correctness before optimization.
  - Measure instead of guessing.
  - Keep hypotheses separate from measured facts.
  - Reuse existing high-quality implementations where beneficial.
  - Do not reinvent functionality merely for novelty.
  - Do not treat current framework limitations as fundamental hardware limits.
  - Make each optimization independently switchable when practical.
  - Preserve reproducibility.
  - Document failed experiments because they are useful research results.

core_hypothesis:
  statement: >
    Transformer inference does not fundamentally require all weights to reside
    simultaneously in GPU VRAM.

  possible_memory_hierarchy: NVMe -> DDR4 -> pinned host buffer -> VRAM -> GPU

  status: hypothesis-to-test

possible_execution_model:
  status: candidate-not-final

  idea: >
    Maintain activations and required runtime state while making transformer
    weight blocks transient residents of VRAM.

  candidate_pipeline[3]:
    - Read future weight block from NVMe into RAM.
    - Transfer upcoming block from RAM into VRAM.
    - Compute current block on GPU.

  desired_overlap: >
    Eventually investigate whether IO(N+2), H2D(N+1), and COMPUTE(N) can execute
    concurrently.

  warning: >
    Do not assume this pipeline is optimal. Measurements may indicate a different
    granularity such as tensors, sublayers, grouped layers or persistent subsets.

correctness_strategy:
  reference: trusted-normal-inference
  compare[4]:
    - hidden states where practical
    - final logits
    - selected token probabilities
    - generated token sequence

  requirement: >
    Streaming execution must first reproduce reference results within an
    appropriate numerical tolerance before performance claims are considered.

model_strategy:
  initial_scale: approximately-7B
  exact_model: undecided

  selection_criteria[7]:
    - accessible weights
    - suitable license
    - well understood architecture
    - compatible tokenizer
    - reference implementation available
    - useful quantization ecosystem
    - enough size to exceed 8 GB VRAM in at least one representation

memory_questions:
  investigate[8]:
    - layer weight footprint
    - activation footprint
    - KV-cache footprint
    - CUDA workspace
    - allocator overhead
    - persistent tensors
    - RAM filesystem cache behavior
    - practical staging-buffer size

future_research:
  status: not-for-initial-poc

  topics[10]:
    - double buffering
    - triple buffering
    - asynchronous direct IO
    - Windows memory-mapped files
    - CUDA streams
    - CUDA graphs
    - weight-only quantization
    - KV-cache quantization
    - MoE expert streaming
    - speculative decoding

beyond_ram_goal:
  status: later-milestone

  definition: >
    Demonstrate bounded-memory inference where the complete model representation
    exceeds intentionally available system RAM and weights are fetched from
    backing storage during inference.

  success_requires[4]:
    - process memory remains bounded
    - model executes correctly
    - storage traffic is measured
    - performance limitations are characterized

documentation:
  audience: beginner-growing-into-systems-ml-research

  explain_from_scratch[12]:
    - transformer inference
    - tensors and matrix multiplication
    - model parameters
    - quantization
    - VRAM
    - system RAM
    - virtual memory
    - memory mapping
    - NVMe
    - PCIe
    - pinned memory
    - CUDA streams

  diagrams[6]:
    - normal inference memory architecture
    - candidate streamed architecture
    - transformer block anatomy
    - memory hierarchy
    - synchronous execution timeline
    - asynchronous pipeline timeline

  rule: >
    Diagrams describing unimplemented designs must be labeled conceptual.
    Measured plots must clearly distinguish measured data from estimates.

repository:
  initial_structure[9]:
    - README.md
    - docs/
    - experiments/
    - benchmarks/
    - results/
    - plots/
    - src/
    - tests/
    - tools/

agent_behavior:
  rules[12]:
    - Work incrementally.
    - Inspect the existing repository before creating files.
    - Explain major architectural decisions.
    - Do not claim a benchmark without running it.
    - Do not fabricate plots or results.
    - Do not assume Python is the chosen implementation.
    - Do not assume llama.cpp is the chosen implementation.
    - Do not assume GGUF is the chosen storage format.
    - Do not assume a custom format is required.
    - Do not prematurely optimize.
    - Preserve experiments even when they fail.
    - Ask me before locking in a major architecture when alternatives remain genuinely competitive.

start_here: >
  Begin with the implementation-path analysis and hardware characterization plan.

  Before writing the inference engine, tell me what you intend to inspect,
  which assumptions can be answered from source-code research, which require
  experiments on my machine, and what the smallest useful first experiment is.