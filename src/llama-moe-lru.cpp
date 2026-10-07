#include "llama-moe-lru.h"
#include "llama.h"

#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>

bool llama_moe_lru::init_model_overrides(llama_model_loader & ml, const std::vector<llama_device> & devices,
                                     std::vector<llama_model_tensor_buft_override> & storage, int n_slot) {
    if (n_slot <= 0) {
        return false; // LRU disabled, keep the overrides as-is
    }

    ggml_backend_buffer_type_t pinned_buft = nullptr;
    for (const auto & d : devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(d.dev, &props);
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(d.dev);
        if (host && host != ggml_backend_cpu_buffer_type()) {
            pinned_buft = host;
            break;
        }
    }
    if (!pinned_buft) {
        return false; // no GPU backend with a pinned host buffer in this build
    }

    const llama_model_tensor_buft_override * cur = ml.tensor_buft_overrides;
    if (!cur) {
        return false;
    }

    std::vector<llama_model_tensor_buft_override> & transformed = storage;
    transformed.clear();
    int n_pinned = 0;
    for (const auto * o = cur; o->pattern != nullptr; ++o) {
        transformed.push_back(*o);
        if (o->buft == ggml_backend_cpu_buffer_type() && strstr(o->pattern, "exps") != nullptr) {
            transformed.back().buft = pinned_buft;
            n_pinned++;
        }
    }
    if (n_pinned == 0) {
        return false;
    }

    transformed.push_back({ nullptr, nullptr });
    ml.tensor_buft_overrides = transformed.data();
    LLAMA_LOG_INFO("llama_moe_lru: %d expert weight override(s) redirected to the pinned host buffer\n", n_pinned);
    return true;
}

bool llama_moe_lru::backend_supports(ggml_backend_dev_t dev) {
    // capability probe: excludes GPU backends without LRU kernels
    // (e.g. Vulkan)
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) {
        return false;
    }
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    ggml_tensor * st  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8);
    ggml_tensor * dec = ggml_lru_decide(ctx.get(), ids, st, 1, 1);
    return ggml_backend_dev_supports_op(dev, dec);
}

llama_moe_lru::llama_moe_lru(const llama_model & model, const std::vector<ggml_backend_t> & gpu_backends, int n_slot) {
    std::unordered_map<ggml_backend_dev_t, ggml_backend_t> backend_of;
    for (ggml_backend_t b : gpu_backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b);
        if (dev) {
            backend_of[dev] = b;
        }
    }
    if (backend_of.empty()) {
        throw std::runtime_error("llama_moe_lru: no GPU backends given");
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(backend_of.begin()->first);

    const int64_t n_expert = model.hparams.n_expert;
    const int64_t n_used_h = model.hparams.n_expert_used();
    const int64_t n_layer_tot = model.hparams.n_layer();

    const int max_batch = (int) (n_slot / n_used_h);
    if (max_batch < 1) {
        throw std::runtime_error("llama_moe_lru: pool too small for one token's experts");
    }

    std::vector<int> pooled_ids;
    std::vector<ggml_backend_buffer_type_t> pending_buft;
    for (int il = 0; il < (int) n_layer_tot; il++) {
        const ggml_tensor * up = model.layers[il].ffn_up_exps;
        const ggml_tensor * down = model.layers[il].ffn_down_exps;
        if (!up || !down) continue;
        if (model.layers[il].ffn_gate_up_exps) {
            throw std::runtime_error("llama_moe_lru: merged gate_up experts not supported");
        }
        if (up->buffer && ggml_backend_buffer_is_host(up->buffer)) {
            if (!up->data) {
                throw std::runtime_error("llama_moe_lru: expert weights not mapped");
            }
            if (ggml_backend_buffer_get_type(up->buffer) != host_buft) {
                throw std::runtime_error("llama_moe_lru: expert weights are host but not pinned "
                                         "(use --n-cpu-moe or -ot with --moe-lru-slots and a non-lazy load mode)");
            }
            pooled_ids.push_back(il);
            ggml_backend_dev_t cdev = nullptr;
            const ggml_tensor * gate_inp = model.layers[il].ffn_gate_inp;
            if (gate_inp && gate_inp->buffer) {
                cdev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(gate_inp->buffer));
            }
            if (!backend_of.count(cdev)) {
                cdev = backend_of.begin()->first;
            }
            pending_buft.push_back(ggml_backend_dev_buffer_type(cdev));
        }
    }
    if (pooled_ids.empty()) {
        throw std::runtime_error("llama_moe_lru: no layers with host-side expert weights");
    }

    this->n_layers = (int) pooled_ids.size();
    this->n_slot = n_slot;
    this->n_expert = n_expert;

    layer_to_pool.assign(n_layer_tot, -1);
    for (int li = 0; li < this->n_layers; li++) {
        layer_to_pool[pooled_ids[li]] = li;
    }

    layers.resize(this->n_layers);

    for (int li = 0; li < this->n_layers; li++) {
        const int il = pooled_ids[li];
        per_layer & L = layers[li];
        L.buft = pending_buft[li];

        ggml_tensor * exps[N_KIND] = {
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_down_exps,
        };
        for (int k = 0; k < N_KIND; k++) {
            if (!exps[k] || !exps[k]->data) {
                throw std::runtime_error("llama_moe_lru: expert tensor missing or unmapped");
            }
            GGML_ASSERT(ggml_is_contiguous(exps[k]));
            L.exps[k] = exps[k];
        }

        ggml_context_ptr pctx(ggml_init({ (N_KIND + 2) * ggml_tensor_overhead(), nullptr, true }));
        for (int k = 0; k < N_KIND; k++) {
            ggml_tensor * t = ggml_new_tensor_3d(pctx.get(), exps[k]->type,
                                                 exps[k]->ne[0], exps[k]->ne[1], n_slot);
            L.pool[k] = t;
        }
        L.state = ggml_new_tensor_1d(pctx.get(), GGML_TYPE_I32,
                                     n_expert + 2 * (int64_t) n_slot + 3);

        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(pctx.get(), L.buft);
        if (!b) {
            throw std::runtime_error("llama_moe_lru: pool alloc failed");
        }
        bufs.emplace_back(b);
        pool_ctxs.emplace_back(std::move(pctx));

        // owner = -1, slot owner = -1, last_use = 0, tick = 1
        const int64_t n_state = n_expert + 2 * (int64_t) n_slot + 3;
        std::vector<int32_t> h(n_state, -1);
        for (int64_t i = n_expert + n_slot; i < n_expert + 2 * n_slot; i++) h[i] = 0;
        h[n_expert + 2 * n_slot] = 1;
        ggml_backend_tensor_set(L.state, h.data(), 0, h.size() * 4);
    }

    size_t pool_bytes = 0;
    for (const auto & b : bufs) {
        pool_bytes += ggml_backend_buffer_get_size(b.get());
    }
    LLAMA_LOG_INFO("llama_moe_lru: %d pooled layers x %d slots, %d experts, topk %lld, "
                   "max verification batch %d, %.2f GiB\n",
                   this->n_layers, n_slot, (int) n_expert, (long long) n_used_h, max_batch,
                   pool_bytes / 1073741824.0);
}


void llama_moe_lru::log_final_stats() {
    int64_t total_ids = 0, total_fills = 0;

    std::vector<int> pool_to_layer(n_layers);
    for (int il = 0; il < (int) layer_to_pool.size(); il++) {
        if (layer_to_pool[il] >= 0) {
            pool_to_layer[layer_to_pool[il]] = il;
        }
    }

    for (int li = 0; li < n_layers; li++) {
        std::vector<int32_t> h(3);
        const int64_t off = (n_expert + 2 * (int64_t) n_slot) * 4;
        ggml_backend_tensor_get(layers[li].state, h.data(), off, 3 * 4);
        total_ids += h[1];
        total_fills += h[2];
        if (h[1] > 0) {
            LLAMA_LOG_INFO("llama_moe_lru: layer %d: ids %lld fills %lld hit %.1f%%",
                           pool_to_layer[li], (long long) h[1], (long long) h[2],
                           100.0 * (1.0 - (double) h[2] / h[1]));
        }
    }
    if (total_ids > 0) {
        LLAMA_LOG_INFO("llama_moe_lru: final: %d layers, %lld ids, %lld fills, hit %.2f%%",
                       n_layers, (long long) total_ids, (long long) total_fills,
                       100.0 * (1.0 - (double) total_fills / total_ids));
    }
}
