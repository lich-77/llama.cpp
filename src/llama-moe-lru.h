#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <vector>
#include <cstdint>

struct llama_model;
struct llama_device;
struct llama_model_tensor_buft_override;

// GPU-resident LRU expert pools: the decode expert matmuls read VRAM
// pools, the LRU bookkeeping and fills run as graph nodes. activated by
// --moe-lru-slots; the pooled layer set follows the placement parameters
struct llama_moe_lru {
    static constexpr int N_KIND = 3; // 0 = gate, 1 = up, 2 = down
    // graph nodes the pooled path adds per layer: decide, 2 views, 3 fills
    static constexpr int NODES_PER_LAYER = 6;
    enum kind { KIND_GATE = 0, KIND_UP = 1, KIND_DOWN = 2 };

    // rewrite the expert overrides to the pinned host buffer; call before
    // the expert tensors are created. storage must outlive the loading phase
    static bool init_model_overrides(struct llama_model_loader & ml, const std::vector<llama_device> & devices,
                                     std::vector<llama_model_tensor_buft_override> & storage, int n_slot);

    // true if the device can run the LRU ops; capability probe instead of
    // device-type filtering, so meta devices pass through (tensor split)
    static bool backend_supports(ggml_backend_dev_t dev);

    llama_moe_lru(const llama_model & model, const std::vector<ggml_backend_t> & gpu_backends, int n_slot);

    // il is the model layer number; the pooled set may be any subset (-ot)
    bool pooled(int il) const { return il >= 0 && il < (int) layer_to_pool.size() && layer_to_pool[il] >= 0; }
    int n_pooled() const { return n_layers; }
    int n_slots() const { return n_slot; }

    // per-layer and total counters at teardown (hit rates)
    void log_final_stats();

    // call only when pooled(il)
    ggml_tensor * state_get(int il) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].state; }
    ggml_tensor * pool_get(int il, int kind) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].pool[kind]; }
    ggml_tensor * exps_get(int il, int kind) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].exps[kind]; }

private:
    struct per_layer {
        ggml_backend_buffer_type_t buft = nullptr; // this layer's device buft
        ggml_tensor * pool[N_KIND] = {};   // [ne0, ne1, n_slot] in VRAM
        ggml_tensor * exps[N_KIND] = {};  // the expert weights in pinned host memory
        ggml_tensor * state = nullptr;     // i32 [n_expert + 2*n_slot + 3]
                                           // (owner | slot_of | last_use | tick | cum_ids | cum_fills)
    };

    int n_layers = 0, n_slot = 0;
    int64_t n_expert = 0;

    // model layer -> index into layers, -1 when not pooled
    std::vector<int> layer_to_pool;

    std::vector<ggml_context_ptr> pool_ctxs; // per-layer pool/state tensor objects
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<per_layer> layers;
};
