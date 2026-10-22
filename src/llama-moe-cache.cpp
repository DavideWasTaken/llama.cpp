#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // hybrid mode: decayed use count of each expert, so the cache keeps the experts of the current text
    static constexpr float score_decay = 0.99f;
    std::vector<float> score; // [n_layer*n_expert]
    std::vector<int32_t> misses;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        score.assign((size_t) n_layer*n_expert, 0.0f);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }

    // hybrid mode: count the hits and misses of layer il and update the scores of its experts
    // up to max_fills misses, highest score first, replace the cached expert with the lowest score if it is lower
    // never evict an expert selected in this step of the layer, the device reads it
    void admit(int32_t il, const int32_t * ids, size_t n_ids, int32_t max_fills, std::vector<fill> & fills, size_t & n_hit, size_t & n_miss) {
        fills.clear();
        n_hit  = 0;
        n_miss = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }

        // each cached layer runs once per step, so all the scores decay at the same rate
        int32_t * slots = slot_map[il];
        float   * sc    = score.data() + (size_t) il*n_expert;
        for (int32_t e = 0; e < n_expert; ++e) {
            sc[e] *= score_decay;
        }

        misses.clear();
        for (int32_t e : uniq) {
            sc[e] += 1.0f;
            if (slots[e] >= 0) {
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }
        n_miss = misses.size();
        std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) { return sc[a] > sc[b]; });

        for (int32_t e : misses) {
            if ((int32_t) fills.size() >= max_fills) {
                break;
            }
            int32_t victim       = -1;
            float   victim_score = INFINITY;
            for (int32_t s = 0; s < n_slots; ++s) {
                const int32_t k = key_of[s];
                if (k < 0) {
                    victim = s;
                    victim_score = -INFINITY;
                    break;
                }
                if (k / n_expert == il && seen[k % n_expert] == seen_gen) {
                    continue;
                }
                if (score[k] < victim_score) {
                    victim = s;
                    victim_score = score[k];
                }
            }
            if (victim < 0 || victim_score >= sc[e]) {
                break; // the other misses have lower scores
            }
            if (key_of[victim] >= 0) {
                slot_map[key_of[victim] / n_expert][key_of[victim] % n_expert] = -1;
            }
            key_of[victim] = il*n_expert + e;
            slots[e] = victim;
            fills.push_back({ e, victim });
        }
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the SLRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks

        // hybrid mode
        ggml_tensor * cpu_ids   = nullptr;  // I32 [n_expert_used*max_batch_hybrid] in host memory, -1 for the cached experts
        ggml_tensor * slot_rows = nullptr;  // I32 [n_expert_used, n_expert], copied to the device in place of slot_map
        std::vector<int32_t> device_rows;   // data of slot_rows: row e has the slot of expert e for each position in a token
        double credit = 0.0;                // unspent upload budget in bytes
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    // hybrid mode: larger batches may make the scheduler offload the CPU experts to the GPU, which cannot read negative ids
    static constexpr int64_t max_batch_hybrid = 16;

    int32_t n_expert_used;

    // hybrid mode: the cached experts run on the device and the others on the CPU, nothing is uploaded before the layer runs
    bool   hybrid;
    double budget_per_layer = 0.0; // bytes of missed experts added to the cache per layer and decode step

    // hybrid mode: scores of the experts, loaded at start to fill the cache and saved at exit
    std::string profile;
    bool    no_alloc       = false;
    int32_t n_expert_model = 0;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    stats stats_hybrid;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            bool hybrid, size_t upload_budget, const char * profile_path) :
            n_expert_used(model.hparams.n_expert_used_max()), hybrid(hybrid),
            profile(hybrid && profile_path ? profile_path : ""), no_alloc(model.hparams.no_alloc), n_expert_model((int32_t) model.hparams.n_expert),
            layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        // hybrid mode: before it, one zero slot per position in a token (never written), as mul_mat_id does not allow duplicate experts in a token
        const int32_t n_extra_slots = hybrid ? n_expert_used + 1 : 1;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_extra_slots), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(hybrid ? 3 : 1);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }
        if (hybrid) {
            // a decode step runs every cached layer once, so the budget of a step is shared by all of them
            size_t n_cached_layers = 0;
            for (const group & g : groups) {
                n_cached_layers += g.n_slots > 0 ? g.layers.size() : 0;
            }
            budget_per_layer = (double) upload_budget/n_cached_layers;
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_extra_slots);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + (hybrid ? n_expert_used : 0), bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (hybrid) {
                    l.cpu_ids = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, n_expert_used*max_batch_hybrid);
                    ggml_format_name(l.cpu_ids, "moe_cache.cpu_ids-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_ids), alignment_host);

                    l.slot_rows = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, n_expert_used, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_rows, "moe_cache.slot_rows-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_rows), alignment_host);
                }
                layer_of[hybrid ? l.slot_rows : l.slot_map] = il;
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
            if (!profile.empty()) {
                load_profile();
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
        if (hybrid) {
            LLAMA_LOG_INFO("%s: hybrid mode: cached experts on the device, the others on the CPU, %.2f MiB of misses added per step\n",
                __func__, upload_budget/1024.0/1024.0);
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        if (!profile.empty() && !no_alloc) {
            save_profile();
        }
        log_stats();
    }

    // profile file: magic, version, n_layer, n_expert (u32), then the scores (f32 [n_layer][n_expert])
    static constexpr uint32_t profile_magic   = 0x50434d4c; // "LMCP"
    static constexpr uint32_t profile_version = 1;

    // fill the cache with the experts of the highest scores, as if a previous session had warmed it up
    void load_profile() {
        const size_t n_layer = layers.size();
        std::vector<float> scores(n_layer*n_expert_model);
        try {
            llama_file file(profile.c_str(), "rb");
            if (file.size() != 4*sizeof(uint32_t) + scores.size()*sizeof(float) ||
                    file.read_u32() != profile_magic || file.read_u32() != profile_version ||
                    file.read_u32() != n_layer || file.read_u32() != (uint32_t) n_expert_model) {
                LLAMA_LOG_WARN("%s: MoE cache profile %s does not match the model, starting with an empty cache\n", __func__, profile.c_str());
                return;
            }
            file.read_raw(scores.data(), scores.size()*sizeof(float));
        } catch (const std::exception & err) {
            LLAMA_LOG_INFO("%s: no MoE cache profile loaded (%s), starting with an empty cache\n", __func__, err.what());
            return;
        }

        size_t n_filled = 0;
        size_t bytes    = 0;
        std::vector<std::pair<float, int32_t>> cand;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            cand.clear();
            for (int32_t il : g.layers) {
                for (int32_t e = 0; e < n_expert_model; ++e) {
                    const int32_t key = il*n_expert_model + e;
                    const float   sc  = scores[key];
                    g.slru.score[key] = std::isfinite(sc) && sc > 0.0f ? sc : 0.0f;
                    if (g.slru.score[key] > 0.0f) {
                        cand.push_back({ g.slru.score[key], key });
                    }
                }
            }
            const size_t n = std::min(cand.size(), (size_t) g.n_slots);
            std::partial_sort(cand.begin(), cand.begin() + n, cand.end(), std::greater<>());
            for (size_t s = 0; s < n; ++s) {
                const int32_t key = cand[s].second;
                const int32_t il  = key / n_expert_model;
                const int32_t e   = key % n_expert_model;
                g.slru.key_of[s]       = key;
                g.slru.slot_map[il][e] = (int32_t) s;
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t expert_size = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) l.experts[ip]->data + e*expert_size, s*expert_size, expert_size);
                    bytes += expert_size;
                }
            }
            n_filled += n;
        }
        LLAMA_LOG_INFO("%s: MoE cache filled from %s: %zu experts, %.2f MiB\n", __func__, profile.c_str(), n_filled, bytes/1024.0/1024.0);
    }

    void save_profile() const {
        const size_t n_layer = layers.size();
        std::vector<float> scores(n_layer*n_expert_model, 0.0f);
        for (const group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                std::copy_n(g.slru.score.begin() + (size_t) il*n_expert_model, n_expert_model, scores.begin() + (size_t) il*n_expert_model);
            }
        }
        try {
            llama_file file(profile.c_str(), "wb");
            file.write_u32(profile_magic);
            file.write_u32(profile_version);
            file.write_u32((uint32_t) n_layer);
            file.write_u32((uint32_t) n_expert_model);
            file.write_raw(scores.data(), scores.size()*sizeof(float));
            LLAMA_LOG_INFO("%s: MoE cache profile saved to %s\n", __func__, profile.c_str());
        } catch (const std::exception & err) {
            LLAMA_LOG_WARN("%s: failed to save the MoE cache profile: %s\n", __func__, err.what());
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // the misses of the hybrid mode run on the CPU, so the selected experts do not have to fit in the cache
        if (hybrid) {
            return n_tokens > 0 && n_tokens <= max_batch_hybrid ? l.slot_rows : nullptr;
        }

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_cpu_ids(int32_t il) const {
        if (!hybrid || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        return layers[il].cpu_ids;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (hybrid) {
            return copy_hybrid(backend, il, dst);
        }

        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        const size_t bytes = upload_fills(backend, l, g);

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    // upload the experts in fills to their slots, consecutive experts going to consecutive slots are uploaded together
    size_t upload_fills(ggml_backend_t backend, const layer & l, const group & g) const {
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        return bytes;
    }

    // hybrid mode: the device reads the cached experts, the zero slots give a zero output for the others
    // the CPU computes the others from cpu_ids; within the budget, some misses are uploaded for the next steps
    bool copy_hybrid(ggml_backend_t backend, int32_t il, ggml_tensor * dst) {
        layer & l = layers[il];
        group & g = groups[l.ig];

        const int32_t * slots    = g.slru.slot_map[il];
        const int32_t   n_expert = g.slru.n_expert;

        GGML_ASSERT(ids.size() <= (size_t) ggml_nelements(l.cpu_ids));
        GGML_ASSERT(ggml_nbytes(dst) == ggml_nbytes(l.slot_rows));

        // both maps describe the cache before this step adds experts to it
        // a miss at position k of a token reads zero slot n_slots + k, so no slot repeats in a token
        l.device_rows.resize(n_expert*n_expert_used);
        for (int32_t e = 0; e < n_expert; ++e) {
            for (int32_t k = 0; k < n_expert_used; ++k) {
                l.device_rows[e*n_expert_used + k] = slots[e] >= 0 ? slots[e] : g.n_slots + k;
            }
        }
        int32_t * cpu_ids = (int32_t *) l.cpu_ids->data;
        for (size_t i = 0; i < ids.size(); ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            cpu_ids[i] = slots[ids[i]] >= 0 ? -1 : ids[i];
        }

        size_t expert_size = 0;
        for (const ggml_tensor * w : l.experts) {
            expert_size += w->nb[2];
        }
        l.credit = std::min(l.credit + budget_per_layer, std::max(budget_per_layer, 2.0*expert_size));
        const int32_t max_fills = (int32_t) (l.credit/expert_size);

        size_t n_hit  = 0;
        size_t n_miss = 0;
        g.slru.admit(il, ids.data(), ids.size(), max_fills, fills, n_hit, n_miss);
        l.credit -= (double) fills.size()*expert_size;

        // the slots taken by the uploads are not read in this step: they held experts not selected here
        const size_t bytes = upload_fills(backend, l, g);

        stats_hybrid.hits   += n_hit;
        stats_hybrid.misses += n_miss;
        stats_hybrid.bytes  += bytes;

        // l.device_rows stays alive until the next copy of this layer, which synchronizes the backend first
        ggml_backend_tensor_set_async(backend, dst, l.device_rows.data(), 0, ggml_nbytes(dst));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        log("hybrid (misses on the CPU)", stats_hybrid);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        bool hybrid, size_t upload_budget, const char * profile) :
    pimpl(new impl(model, backends, bufts, size, hybrid, upload_budget, profile)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_cpu_ids(int32_t il) const {
    return pimpl->get_cpu_ids(il);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
