#include "moe-cache.cuh"

#include "mmvq.cuh"

#include "ggml-cuda.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// MoE expert VRAM cache (BeyondVRAM proof of concept). See moe-cache.cuh.
//
// Config: LLAMA_MOE_CACHE=<cap>:<first_layer> (first_layer is informational on
// this side; the llama-side loader decides which tensors become slot tensors).
// Callers that trigger cache fetches must set GGML_CUDA_DISABLE_GRAPHS=1 because
// resolution requires a stream synchronization (same constraint as the fallback
// path in ggml_cuda_mul_mat_id).

namespace {

struct moe_cache_state {
    ggml_tensor * tensor     = nullptr;  // slot tensor, ne[2] = cap, data in a CUDA buffer
    char *        dev_base   = nullptr;  // = tensor->data
    void *        host_base  = nullptr;  // pinned, full n_experts * slice_bytes
    int64_t       n_experts  = 0;
    int64_t       cap        = 0;
    size_t        slice_bytes = 0;       // = tensor->nb[2]

    std::vector<int> expert_slot;        // n_experts -> slot or -1
    std::vector<int> slot_expert;        // cap -> expert or -1

    // LRU over slots: front = most recently used
    std::list<int>                          lru;
    std::vector<std::list<int>::iterator>   lru_pos;  // slot -> position in lru

    int64_t hits   = 0;
    int64_t misses = 0;
    int64_t bytes  = 0;
};

struct moe_cache_config {
    bool parsed       = false;
    bool enabled      = false;
    int  cap          = 0;
    int  first_layer  = 0;
    int  last_layer   = -1; // -1: unbounded (all layers >= first_layer)
};

moe_cache_config get_config() {
    static moe_cache_config cfg = [] {
        moe_cache_config c;
        c.parsed = true;
        const char * env = getenv("LLAMA_MOE_CACHE");
        if (env == nullptr || env[0] == '\0') {
            return c;
        }
        int cap = 0, first_layer = 0, last_layer = -1;
        const int n_fields = sscanf(env, "%d:%d:%d", &cap, &first_layer, &last_layer);
        if (n_fields < 2 || cap <= 0 || first_layer < 0 || (n_fields == 3 && last_layer < first_layer)) {
            GGML_ABORT("LLAMA_MOE_CACHE must have the form <cap>:<first_layer>[:<last_layer>] (e.g. 8:18 or 8:0:8), got '%s'", env);
        }
        c.enabled     = true;
        c.cap         = cap;
        c.first_layer = first_layer;
        c.last_layer  = last_layer;
        return c;
    }();
    return cfg;
}

std::unordered_map<const ggml_tensor *, std::unique_ptr<moe_cache_state>> & registry() {
    static std::unordered_map<const ggml_tensor *, std::unique_ptr<moe_cache_state>> r;
    return r;
}

} // namespace

bool ggml_cuda_moe_cache_config(int & cap, int & first_layer) {
    const moe_cache_config cfg = get_config();
    cap         = cfg.cap;
    first_layer = cfg.first_layer;
    return cfg.enabled;
}

bool ggml_cuda_moe_cache_config_range(int & cap, int & first_layer, int & last_layer) {
    const moe_cache_config cfg = get_config();
    cap         = cfg.cap;
    first_layer = cfg.first_layer;
    last_layer  = cfg.last_layer;
    return cfg.enabled;
}

// resolve one requested expert to a slot, fetching from the pinned host buffer on miss
static int moe_cache_resolve(moe_cache_state & st, int32_t expert, cudaStream_t stream) {
    GGML_ASSERT(expert >= 0 && expert < st.n_experts);

    int slot = st.expert_slot[expert];
    if (slot >= 0) {
        // hit: mark slot as most recently used
        st.lru.erase(st.lru_pos[slot]);
        st.lru.push_front(slot);
        st.lru_pos[slot] = st.lru.begin();
        st.hits++;
        return slot;
    }

    // miss: evict the least recently used slot
    slot = st.lru.back();
    const int evicted = st.slot_expert[slot];
    if (evicted >= 0) {
        st.expert_slot[evicted] = -1;
    }
    st.lru.erase(st.lru_pos[slot]);
    st.lru.push_front(slot);
    st.lru_pos[slot] = st.lru.begin();

    CUDA_CHECK(cudaMemcpyAsync(st.dev_base + (size_t) slot*st.slice_bytes,
                               (const char *) st.host_base + (size_t) expert*st.slice_bytes,
                               st.slice_bytes, cudaMemcpyHostToDevice, stream));
    st.expert_slot[expert] = slot;
    st.slot_expert[slot]   = expert;
    st.misses++;
    st.bytes += (int64_t) st.slice_bytes;
    return slot;
}

bool ggml_cuda_moe_cache_forward(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
    if (!get_config().enabled) {
        return false;
    }
    const auto it = registry().find(src0);
    if (it == registry().end()) {
        return false;
    }
    moe_cache_state & st = *it->second;

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->nb[0] == (int64_t) sizeof(int32_t));
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(ids->ne[0] % src1->ne[1] == 0 && src1->ne[3] == 1); // src1 tokens are in ne[2]; ne[1] is 1 (gate/up) or n_expert_used (down)
    GGML_ASSERT(dst->ne[1] == ids->ne[0] && dst->ne[2] == ids->ne[1]);
    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT((int64_t) get_config().cap >= ids->ne[0]);

    cudaStream_t stream = ctx.stream();

    // download the requested global expert ids (<= 8 * n_tokens)
    // note: ids can be a strided view (nb[1] > ne[0]*nb[0]), so the host copy
    // must cover ggml_nbytes(ids), not just nelements*sizeof(int32_t)
    std::vector<int32_t> ids_host(ggml_nbytes(ids)/sizeof(int32_t));
    if (ids->buffer != nullptr && ggml_backend_buffer_is_host(ids->buffer)) {
        memcpy(ids_host.data(), ids->data, ggml_nbytes(ids));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
        // requires GGML_CUDA_DISABLE_GRAPHS=1 (precedent: fallback path in ggml_cuda_mul_mat_id)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const int64_t n_used  = ids->ne[0];
    const int64_t stride_i = ids->nb[0] / (int64_t) sizeof(int32_t);
    const int64_t stride_t = ids->nb[1] / (int64_t) sizeof(int32_t);

    ggml_cuda_pool_alloc<int32_t> slot_ids_dev(ctx.pool(), n_used);

    // one batch-1 mmvq launch per token: a single launch over more requested
    // experts than cap slots could read slots invalidated by evictions from the
    // same call; per token at most n_used <= cap experts are live, and the H2D
    // fetches of the next token are stream-ordered after this token's kernel
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        std::vector<int32_t> slot_ids_host(n_used);
        for (int64_t i = 0; i < n_used; ++i) {
            slot_ids_host[i] = moe_cache_resolve(st, ids_host[t*stride_t + i*stride_i], stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev.ptr, slot_ids_host.data(), n_used*sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        ggml_tensor ids_t  = *ids;
        ids_t.ne[1] = 1;
        ids_t.nb[0] = sizeof(int32_t);
        ids_t.nb[1] = n_used*ids_t.nb[0];
        ids_t.nb[2] = ids_t.nb[1];
        ids_t.nb[3] = ids_t.nb[2];
        ids_t.data  = slot_ids_dev.ptr;

        ggml_tensor src1_t = *src1;
        src1_t.ne[2] = 1;
        src1_t.nb[3] = src1_t.nb[2];
        src1_t.data  = (char *) src1->data + t*src1->nb[2];

        ggml_tensor dst_t  = *dst;
        dst_t.ne[2] = 1;
        dst_t.nb[3] = dst_t.nb[2];
        dst_t.data  = (char *) dst->data + t*dst->nb[2];

        ggml_cuda_mul_mat_vec_q(ctx, src0, &src1_t, &ids_t, &dst_t);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

bool ggml_cuda_moe_cache_registered(const ggml_tensor * t) {
    return get_config().enabled && registry().find(t) != registry().end();
}

std::string ggml_cuda_moe_cache_stats_string() {
    std::string out;
    int64_t hits = 0, misses = 0, bytes = 0;
    for (const auto & kv : registry()) {
        hits   += kv.second->hits;
        misses += kv.second->misses;
        bytes  += kv.second->bytes;
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"enabled\": %s, \"cap\": %d, \"first_layer\": %d, \"last_layer\": %d, \"tensors\": %zu, \"hits\": %lld, \"misses\": %lld, \"bytes\": %lld, \"per_tensor\": [",
             get_config().enabled ? "true" : "false",
             get_config().cap, get_config().first_layer, get_config().last_layer, registry().size(),
             (long long) hits, (long long) misses, (long long) bytes);
    out += buf;

    bool first = true;
    for (const auto & kv : registry()) {
        const moe_cache_state & st = *kv.second;
        snprintf(buf, sizeof(buf),
                 "%s{\"name\": \"%s\", \"cap\": %lld, \"n_experts\": %lld, \"hits\": %lld, \"misses\": %lld, \"bytes\": %lld}",
                 first ? "" : ", ", ggml_get_name(st.tensor),
                 (long long) st.cap, (long long) st.n_experts,
                 (long long) st.hits, (long long) st.misses, (long long) st.bytes);
        out += buf;
        first = false;
    }
    out += "]}";
    return out;
}

// public C API (ggml/include/ggml-cuda.h), called from the llama-side loader

extern "C" bool ggml_cuda_moe_cache_get_config(int * cap, int * first_layer) {
    int c = 0, fl = 0;
    const bool enabled = ggml_cuda_moe_cache_config(c, fl);
    if (cap) {
        *cap = c;
    }
    if (first_layer) {
        *first_layer = fl;
    }
    return enabled;
}

extern "C" bool ggml_cuda_moe_cache_get_config_range(int * cap, int * first_layer, int * last_layer) {
    int c = 0, fl = 0, ll = -1;
    const bool enabled = ggml_cuda_moe_cache_config_range(c, fl, ll);
    if (cap) {
        *cap = c;
    }
    if (first_layer) {
        *first_layer = fl;
    }
    if (last_layer) {
        *last_layer = ll;
    }
    return enabled;
}

extern "C" void * ggml_cuda_moe_cache_register(struct ggml_tensor * slot_tensor, int64_t n_experts) {
    GGML_ASSERT(slot_tensor != nullptr);
    GGML_ASSERT(get_config().enabled);
    GGML_ASSERT(slot_tensor->ne[2] == get_config().cap);
    GGML_ASSERT(n_experts > slot_tensor->ne[2]);
    GGML_ASSERT(slot_tensor->data != nullptr);

    auto st = std::make_unique<moe_cache_state>();
    st->tensor      = slot_tensor;
    st->dev_base    = (char *) slot_tensor->data;
    st->n_experts   = n_experts;
    st->cap         = slot_tensor->ne[2];
    st->slice_bytes = (size_t) slot_tensor->nb[2];
    st->expert_slot.assign(n_experts, -1);
    st->slot_expert.assign(st->cap, -1);
    st->lru_pos.resize(st->cap);
    for (int64_t s = 0; s < st->cap; ++s) {
        st->lru.push_front((int) s);
        st->lru_pos[s] = st->lru.begin();
    }

    const size_t host_bytes = (size_t) n_experts*st->slice_bytes;
    cudaError_t err = cudaMallocHost(&st->host_base, host_bytes);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to allocate %.2f MiB of pinned host memory for %s: %s\n",
                       __func__, host_bytes/1024.0/1024.0, ggml_get_name(slot_tensor), cudaGetErrorString(err));
        return nullptr;
    }

    void * host_base = st->host_base;
    const int64_t cap = st->cap;
    registry()[slot_tensor] = std::move(st);

    GGML_LOG_INFO("%s: registered %s: cap = %lld slots, n_experts = %lld, slice = %.2f MiB, host buffer = %.2f MiB\n",
                  __func__, ggml_get_name(slot_tensor), (long long) cap, (long long) n_experts,
                  (double) (host_bytes/n_experts)/1024.0/1024.0, host_bytes/1024.0/1024.0);

    return host_base;
}

extern "C" const char * ggml_cuda_moe_cache_stats_json(void) {
    static std::string json;
    json = ggml_cuda_moe_cache_stats_string();
    return json.c_str();
}
