#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <vector>
#include <cstdint>

struct llama_model;
struct llama_device;
struct llama_model_tensor_buft_override;

// GPU-resident LRU expert pools, activated by --moe-lru-slots
struct llama_moe_lru {
    static constexpr int N_KIND = 3;
    static constexpr int NODES_PER_LAYER = 6;
    enum kind { KIND_GATE = 0, KIND_UP = 1, KIND_DOWN = 2 };

    // call before the expert tensors are created; storage must outlive the
    // loading phase
    static bool init_model_overrides(struct llama_model_loader & ml, const std::vector<llama_device> & devices,
                                     std::vector<llama_model_tensor_buft_override> & storage, int n_slot);

    static bool backend_supports(ggml_backend_dev_t dev);

    llama_moe_lru(const llama_model & model, const std::vector<ggml_backend_t> & gpu_backends, int n_slot);

    // il is the model layer number
    bool pooled(int il) const { return il >= 0 && il < (int) layer_to_pool.size() && layer_to_pool[il] >= 0; }
    int n_pooled() const { return n_layers; }
    int n_slots() const { return n_slot; }

    void log_final_stats();

    // call only when pooled(il)
    ggml_tensor * state_get(int il) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].state; }
    ggml_tensor * pool_get(int il, int kind) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].pool[kind]; }
    ggml_tensor * exps_get(int il, int kind) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].exps[kind]; }

private:
    struct per_layer {
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_tensor * pool[N_KIND] = {};   // [ne0, ne1, n_slot] in VRAM
        ggml_tensor * exps[N_KIND] = {};  // expert weights in pinned host memory
        ggml_tensor * state = nullptr;     // i32, layout in ggml.h
    };

    int n_layers = 0, n_slot = 0;
    int64_t n_expert = 0;

    std::vector<int> layer_to_pool;

    std::vector<ggml_context_ptr> pool_ctxs; // per-layer pool/state tensor objects
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<per_layer> layers;
};
