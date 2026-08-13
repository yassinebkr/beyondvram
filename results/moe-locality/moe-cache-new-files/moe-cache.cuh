#pragma once

// MoE expert VRAM cache (BeyondVRAM proof of concept).
//
// A per-tensor LRU cache that lets a fused MoE expert tensor (ne[2] = cap slot
// count, living in a CUDA buffer) stand in for the full n_experts weight data
// held in a pinned host buffer. Enabled via LLAMA_MOE_CACHE=<cap>:<first_layer>.
// When the env var is unset every function here is inert and stock behavior is
// preserved.

#include "common.cuh"

#include <string>

// Returns true when LLAMA_MOE_CACHE is set and valid; fills cap/first_layer.
bool ggml_cuda_moe_cache_config(int & cap, int & first_layer);

// Same, also fills last_layer (-1 = unbounded).
bool ggml_cuda_moe_cache_config_range(int & cap, int & first_layer, int & last_layer);

// If src0 is a registered cache slot tensor, executes the whole mul_mat_id:
// per token, resolves the requested global expert ids to slots (fetching
// missing expert slices from the pinned host buffer into VRAM slots via LRU
// eviction) and launches the batch-1 mmvq kernel with remapped slot ids.
// The per-token split is required: a single kernel launch over more requested
// experts than cap slots would read slot contents already invalidated by
// evictions from the same call. Returns false when src0 is not registered
// (stock path).
bool ggml_cuda_moe_cache_forward(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor * dst);

// Returns true when the cache is enabled and t is a registered slot tensor.
bool ggml_cuda_moe_cache_registered(const ggml_tensor * t);

std::string ggml_cuda_moe_cache_stats_string();
