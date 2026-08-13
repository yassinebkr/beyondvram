// moe-trace: observational expert-routing tracer for MoE models (BeyondVRAM Track 1).
//
// Captures the tensors named "ffn_moe_topk-<layer>" produced by
// llm_graph_context::build_moe_ffn (src/llama-graph.cpp) through the existing
// scheduler eval-callback API. It changes nothing about routing, weights,
// scheduling, expert placement, or model outputs; it only reads the selected
// expert IDs after graph evaluation and appends them as JSONL records:
//
//   {"pos": <absolute token position>, "layer": <0-based layer>, "experts": [ids]}
//
// One record per (layer, token column). The output path comes from the
// MOE_TRACE_OUT environment variable (default: moe-trace.jsonl).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct moe_trace_state {
    FILE *  out = nullptr;
    int64_t call_pos_base = 0;   // absolute position of the first column of the current llama_decode call
    int64_t graph_offset  = 0;   // columns consumed by earlier micro-batches within the current call
    int64_t prev_cols     = 0;   // token columns of the previous record (same micro-batch graph)
    int     last_layer    = -1;  // layer of the previous record; a drop signals a new graph
    int64_t n_records     = 0;
};

bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (moe_trace_state *) user_data;

    static const char prefix[] = "ffn_moe_topk";
    if (strncmp(t->name, prefix, sizeof(prefix) - 1) != 0) {
        return false;  // not interested in this node
    }
    if (ask) {
        return true;   // request the data after evaluation
    }

    // parse the layer index from "ffn_moe_topk-<il>"
    const char * dash = strrchr(t->name, '-');
    const int layer = dash != nullptr ? atoi(dash + 1) : -1;

    if (t->type != GGML_TYPE_I32) {
        LOG_ERR("%s: unexpected tensor type %s for %s\n", __func__, ggml_type_name(t->type), t->name);
        return true;
    }

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];

    // copy to host if the tensor lives on a device
    std::vector<uint8_t> host_data;
    const uint8_t * data = nullptr;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        host_data.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, host_data.data(), 0, host_data.size());
        data = host_data.data();
    }

    // a layer index that does not advance means a new micro-batch graph started
    if (layer <= state->last_layer) {
        state->graph_offset += state->prev_cols;
    }
    state->last_layer = layer;
    state->prev_cols  = n_tokens;

    for (int64_t j = 0; j < n_tokens; ++j) {
        const int64_t pos = state->call_pos_base + state->graph_offset + j;
        fprintf(state->out, "{\"pos\": %lld, \"layer\": %d, \"experts\": [",
                (long long) pos, layer);
        for (int64_t i = 0; i < n_expert_used; ++i) {
            const size_t offset = (size_t) (j * t->nb[1] + i * t->nb[0]);
            const int32_t expert = *(const int32_t *) (data + offset);
            fprintf(state->out, "%s%d", i == 0 ? "" : ", ", expert);
        }
        fprintf(state->out, "]}\n");
        state->n_records += 1;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * out_path = getenv("MOE_TRACE_OUT");
    if (out_path == nullptr || out_path[0] == '\0') {
        out_path = "moe-trace.jsonl";
    }

    moe_trace_state trace;
    trace.out = fopen(out_path, "w");
    if (trace.out == nullptr) {
        LOG_ERR("%s: cannot open trace output '%s'\n", __func__, out_path);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &trace;
    params.warmup            = false;  // do not pollute the trace with warmup decodes

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto * smpl = common_sampler_init(model, params.sampling);
    if (smpl == nullptr) {
        LOG_ERR("%s: failed to initialize sampler\n", __func__);
        return 1;
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (prompt_tokens.empty()) {
        LOG_ERR("%s: no input tokens (try providing a prompt with -p)\n", __func__);
        return 1;
    }
    const int n_prompt = (int) prompt_tokens.size();
    LOG_INF("%s: prompt tokens = %d, predict = %d, trace out = %s\n",
            __func__, n_prompt, params.n_predict, out_path);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    int64_t n_pos = 0;
    llama_token new_token_id = LLAMA_TOKEN_NULL;

    for ( ; n_pos + batch.n_tokens < n_prompt + params.n_predict; ) {
        trace.call_pos_base = n_pos;
        trace.graph_offset  = 0;
        trace.prev_cols     = 0;
        trace.last_layer    = -1;

        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }
        fflush(trace.out);

        n_pos += batch.n_tokens;

        new_token_id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token_id, true);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];
        const int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            LOG_ERR("%s: failed to convert token to piece\n", __func__);
            return 1;
        }
        fwrite(buf, 1, n, stdout);
        fflush(stdout);

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    printf("\n");
    LOG_INF("%s: wrote %lld trace records to %s\n", __func__, (long long) trace.n_records, out_path);

    common_sampler_free(smpl);
    fclose(trace.out);
    llama_backend_free();

    return 0;
}
