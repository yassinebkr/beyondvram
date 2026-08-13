# Local model checkpoints (not tracked in git)

Download commands used on this machine (huggingface_hub, `hf download`):

```
# Track 0 control model
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models/Qwen3-8B-GGUF
# (Qwen3-8B BF16 HF checkpoint for the archived real-layer experiment: Qwen/Qwen3-8B -> models/Qwen3-8B)

# Track 1 MoE
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models/Qwen3-30B-A3B-GGUF

# Track 2 dense
hf download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf --local-dir models/Qwen3-32B-GGUF

# Track 3 low-bit
hf download bartowski/Qwen_Qwen3-32B-GGUF Qwen_Qwen3-32B-Q3_K_M.gguf --local-dir models/Qwen3-32B-GGUF
hf download bartowski/Qwen_Qwen3-32B-GGUF Qwen_Qwen3-32B-IQ2_XXS.gguf --local-dir models/Qwen3-32B-GGUF
hf download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf --local-dir models/bitnet-b1.58-2B-4T

# Speculative-decoding draft
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models/Qwen3-0.6B-GGUF

# Track 4 validation: MXFP4 on sm_86 + second-architecture roofline check
hf download ggml-org/gpt-oss-20b-GGUF --local-dir models/gpt-oss-20b-GGUF
```
