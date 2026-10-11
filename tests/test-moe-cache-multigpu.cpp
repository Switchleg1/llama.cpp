#include "../src/llama-context.h"
#include "../src/llama-graph.h"
#include "ggml-cuda/moe-cache-host.cuh"
#include "test-moe-cache.h"
#include "ggml-quants.h"

namespace {

constexpr int n_dim     = 256;
constexpr int n_experts = 8;
constexpr int n_used    = 2;
constexpr int n_slots   = 4;
constexpr size_t bounded_host_budget = 2 * 1024 * 1024;
constexpr uint32_t all_cached_layers = (1u << 2) - 1;

ggml_graph_execution_certificate layer_certificate(bool required = false) {
    ggml_graph_execution_certificate certificate{};
    certificate.magic            = GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC;
    certificate.abi_version      = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;
    certificate.struct_size      = sizeof(certificate);
    certificate.flags            = required ? GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED : 0;
    certificate.domain           = required ? GGML_GRAPH_EXECUTION_DOMAIN_DRAFT : GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
    certificate.row_semantics    = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    certificate.owner_namespace  = 723;
    certificate.owner_generation = 1;
    certificate.n_rows           = 1;
    certificate.n_sequences      = 1;
    return certificate;
}

struct layer_fixture {
    ggml_context_ptr                                  weights;
    ggml_backend_buffer_ptr                           weight_buffer;
    std::vector<ggml_backend_buffer_ptr>              ordinary_buffers;
    ggml_backend_buffer_type_t                        buft = nullptr;
    uint32_t                                           cached_layers = all_cached_layers;
    uint32_t                                           cache_slots   = n_slots;
    llm_graph_result                                  result{ 256 };
    ggml_tensor *                                     input      = nullptr;
    ggml_tensor *                                     logits[2]  = {};
    ggml_tensor *                                     ids[2]     = {};
    ggml_tensor *                                     up[2]      = {};
    ggml_tensor *                                     gate[2]    = {};
    ggml_tensor *                                     gate_up[2] = {};
    ggml_tensor *                                     down[2]    = {};
    ggml_tensor *                                     down_bias[2] = {};
    ggml_tensor *                                     output[2]  = {};
    std::vector<ggml_backend_moe_candidate_group_v2>  groups;
    std::vector<ggml_backend_moe_candidate_tensor_v2> tensors;

    explicit layer_fixture(size_t host_budget, uint32_t layout, ggml_type type, bool pageable = false,
                           const std::array<int, 2> & ordinary_devices = { 0, 0 },
                           uint32_t cache_layer_mask = all_cached_layers, uint32_t slots = n_slots, bool named_weights = false,
                           ggml_type down_type = GGML_TYPE_COUNT) :
        cached_layers(cache_layer_mask), cache_slots(slots) {
        weights.reset(ggml_init({ 32 * ggml_tensor_overhead(), nullptr, true }));
        CHECK(weights != nullptr);
        buft = pageable ? pageable_cached_buffer_type() : ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
        CHECK(buft != nullptr);
        for (int layer = 0; layer < 2; ++layer) {
            const bool cached = (cached_layers & (1u << layer)) != 0;
            const auto ordinary = [&](ggml_tensor * tensor) {
                if (cached) {
                    return;
                }
                auto * ordinary_buft = ggml_backend_cuda_buffer_type(ordinary_devices[layer]);
                auto buffer = ggml_backend_buft_alloc_buffer(ordinary_buft, ggml_backend_buft_get_alloc_size(ordinary_buft, tensor));
                CHECK(buffer != nullptr);
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                CHECK(ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
                ordinary_buffers.emplace_back(buffer);
            };
            groups.push_back({ layout, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0 });
            const auto bank = [&](uint32_t role, int rows) {
                const auto bank_type = role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && down_type != GGML_TYPE_COUNT ? down_type : type;
                auto * tensor = ggml_new_tensor_3d(weights.get(), bank_type, n_dim, rows, n_experts);
                if (named_weights) {
                    const char * name = role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT ? "gate_up" :
                        role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT ? "gate" :
                        role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT ? "up" : "down";
                    ggml_format_name(tensor, "blk.%d.ffn_%s_exps.weight", layer, name);
                }
                tensors.push_back({ tensor, (uint32_t) layer, role, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
                                    cached ? static_cast<uint32_t>(GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) : 0u, 0 });
                ordinary(tensor);
                return tensor;
            };
            if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
                gate_up[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 2 * n_dim);
            } else {
                if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
                    gate[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, n_dim);
                }
                up[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, n_dim);
            }
            down[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, n_dim);
            if (pageable) {
                down_bias[layer] = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, n_dim, n_experts);
                tensors.push_back({down_bias[layer], (uint32_t) layer, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS,
                    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS,
                    cached ? static_cast<uint32_t>(GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) : 0u, 0});
                ordinary(down_bias[layer]);
            }
        }
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
        if (weight_buffer != nullptr) {
            ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        CHECK(weight_buffer != nullptr || !ordinary_buffers.empty());
        size_t salt = 101;
        for (auto * tensor = ggml_get_first_tensor(weights.get()); tensor;
             tensor        = ggml_get_next_tensor(weights.get(), tensor)) {
            const auto bytes = cached_fusion_test_data(tensor, salt++);
            ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
        }
        if (host_budget != 0) {
            auto snapshot = manifest();
            CHECK(ggml_backend_cuda_moe_cached_configure_sources(buft, &snapshot));
        }
        build_graph();
    }

    void build_graph(llm_ffn_op_type activation = LLM_FFN_SILU, float clamp = 0.0f, bool scaled = false, bool external = false, int n_tokens = 1) {
        result.reset();
        auto * ctx = result.get_ctx();
        input      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_dim, n_tokens);
        ggml_set_input(input);
        auto *              cur = input;
        llama_adapter_loras loras;
        llm_graph_params    params{};
        params.arch                = LLM_ARCH_QWEN3MOE;
        params.hparams.n_embd      = n_dim;
        params.hparams.n_layer_all = 2;
        params.hparams.n_expert    = n_experts;
        params.hparams.n_expert_used_arr.fill(n_used);
        params.hparams.n_head_arr.fill(1);
        params.hparams.n_head_kv_arr.fill(1);
        params.hparams.swiglu_clamp_exp.fill(clamp);
        params.ubatch.n_tokens = n_tokens;
        params.loras           = &loras;
        params.res             = &result;
        llm_graph_context builder(params);
        for (int layer = 0; layer < 2; ++layer) {
            logits[layer] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, n_tokens);
            ggml_set_input(logits[layer]);
            auto * scale = scaled ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_experts) : nullptr;
            auto * route = external ? ggml_argsort_top_k(ctx, logits[layer], n_used) : nullptr;
            cur = output[layer] =
                builder.build_moe_ffn(cur, nullptr, nullptr, up[layer], nullptr, gate[layer], nullptr, down[layer],
                                      down_bias[layer], nullptr, n_experts, n_used, activation, true, 1.0f,
                                      LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, layer, logits[layer], gate_up[layer],
                                      nullptr, scale, nullptr, nullptr, route);
            auto & region         = result.get_moe_regions().back();
            region.semantic_group = layer;
            region.domain         = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
            ids[layer]            = region.route;
            const bool simple = !down_bias[layer] && !scaled && !external &&
                (activation == LLM_FFN_SILU || activation == LLM_FFN_GELU) &&
                !(activation == LLM_FFN_SILU && gate[layer] && clamp > 1e-6f);
            const auto & hybrid = region.hybrid_ffn;
            CHECK((hybrid.input != nullptr) == simple);
            if (simple) {
                CHECK(hybrid.up == up[layer] && hybrid.gate == gate[layer] && hybrid.gate_up == gate_up[layer]);
                CHECK(hybrid.down == down[layer] && hybrid.type_op == activation);
                CHECK(hybrid.down_output->op == GGML_OP_MUL_MAT_ID && hybrid.down_output->src[0] == down[layer]);
                CHECK(hybrid.down_output->src[1] == hybrid.activation && hybrid.down_output->src[2] == region.route);
                CHECK(hybrid.input->ne[0] == n_dim && hybrid.input->ne[1] == 1 && hybrid.input->ne[2] == n_tokens);
            }
            ggml_set_output(cur);
            ggml_build_forward_expand(result.get_gf(), cur);
        }
    }

    ~layer_fixture() {
        weight_buffer.reset();
        ggml_backend_cuda_moe_cached_free_buffer_type(buft);
    }

    ggml_backend_moe_candidate_snapshot_v2 manifest() const {
        return candidate_snapshot_v2(cache_slots, groups.data(), groups.size(), tensors.data(), tensors.size());
    }

    void set_inputs(int iteration) {
        std::vector<float> values(ggml_nelements(input));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.001f * (static_cast<int>((i + iteration) % 17) - 8);
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        for (int layer = 0; layer < 2; ++layer) {
            std::vector<float> scores(ggml_nelements(logits[layer]));
            for (size_t i = 0; i < scores.size(); ++i) {
                scores[i] = (i % n_experts + iteration + 3 * layer + 2 * ((i / n_experts) % 2)) % n_experts;
            }
            ggml_backend_tensor_set(logits[layer], scores.data(), 0, scores.size() * sizeof(float));
        }
    }
};

struct split_probe {
    layer_fixture *                                             fixture;
    std::vector<ggml_backend_t>                                 backends;
    std::vector<ggml_status (*)(ggml_backend_t, ggml_cgraph *)> delegates;
    int                                                         grouped_splits[2]  = {};
    int                                                         imported_splits[2] = {};
    int                                                         fail_layer         = -1;
    bool                                                        failed             = false;
    const ggml_graph_execution_certificate *                     expected_certificate = nullptr;
    uint64_t                                                    expected_source_uid = 0;
    int                                                         certificate_calls = 0;
    std::function<void(ggml_backend_t, ggml_cgraph *)>              before_dispatch;
    std::function<void(ggml_backend_t, ggml_cgraph *)>              after_success;
};

split_probe * active_probe = nullptr;

ggml_status probe_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    CHECK(active_probe != nullptr);
    auto & probe = *active_probe;
    CHECK(!probe.failed);
    if (probe.expected_certificate != nullptr) {
        const auto & expected = *probe.expected_certificate;
        const auto & actual = graph->execution_certificate;
        CHECK(actual.magic == expected.magic && actual.abi_version == expected.abi_version);
        CHECK(actual.struct_size == expected.struct_size && actual.flags == expected.flags);
        CHECK(actual.domain == expected.domain && actual.row_semantics == expected.row_semantics);
        CHECK(actual.owner_namespace == expected.owner_namespace && actual.owner_generation == expected.owner_generation);
        CHECK(actual.n_rows == expected.n_rows && actual.n_sequences == expected.n_sequences);
        CHECK(actual.source_graph_uid == probe.expected_source_uid);
        CHECK(actual.split_graph_uid == graph->uid && actual.split_graph_uid != actual.source_graph_uid);
        ++probe.certificate_calls;
    }
    const auto found = std::find(probe.backends.begin(), probe.backends.end(), backend);
    CHECK(found != probe.backends.end());
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        for (int layer = 0; layer < 2; ++layer) {
            if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != probe.fixture->down[layer]) {
                continue;
            }
            bool local = false;
            for (int j = 0; j < i; ++j) {
                local = local || (ggml_graph_node(graph, j) == probe.fixture->ids[layer]->src[0] &&
                                  node->src[2] == probe.fixture->ids[layer]);
            }
            if (local) {
                ++probe.grouped_splits[layer];
            } else {
                ++probe.imported_splits[layer];
            }
            if (probe.fail_layer == layer) {
                probe.failed = true;
                return GGML_STATUS_FAILED;
            }
        }
    }
    if (probe.before_dispatch) {
        probe.before_dispatch(backend, graph);
    }
    const auto status = probe.delegates[found - probe.backends.begin()](backend, graph);
    if (status == GGML_STATUS_SUCCESS && probe.after_success) {
        probe.after_success(backend, graph);
    }
    return status;
}

void test_boundary_copy(ggml_backend_t source, ggml_backend_t destination, int source_device, int destination_device) {
    ggml_context_ptr source_ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
    ggml_context_ptr destination_ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
    CHECK(source_ctx && destination_ctx);
    auto *                  src = ggml_new_tensor_1d(source_ctx.get(), GGML_TYPE_F32, n_dim);
    auto *                  dst = ggml_new_tensor_1d(destination_ctx.get(), GGML_TYPE_F32, n_dim);
    ggml_backend_buffer_ptr source_buffer(ggml_backend_alloc_ctx_tensors(source_ctx.get(), source));
    ggml_backend_buffer_ptr destination_buffer(ggml_backend_alloc_ctx_tensors(destination_ctx.get(), destination));
    CHECK(source_buffer && destination_buffer);
    int peer = 0;
    CUDA_OK(cudaDeviceCanAccessPeer(&peer, source_device, destination_device));
    if (!peer) {
        CHECK(!destination->iface.cpy_tensor_async(source, destination, src, dst));
        CHECK(!destination_buffer->iface.cpy_tensor(destination_buffer.get(), src, dst));
    }
    std::vector<float> expected(n_dim);
    for (int i = 0; i < n_dim; ++i) {
        expected[i] = i - 123.25f;
    }
    ggml_backend_tensor_set_async(source, src, expected.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_copy_async(source, destination, src, dst);
    ggml_backend_synchronize(destination);
    CHECK(active_grouped_tensor_values(dst) == expected);
    expected[0] += 1;
    ggml_backend_tensor_set(src, expected.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_copy(src, dst);
    CHECK(active_grouped_tensor_values(dst) == expected);
    fprintf(stderr, "test-moe-cache: boundary copy %d -> %d exact, peer=%d\n", source_device, destination_device, peer);
}

void run_layers(const std::vector<int> & devices,
                size_t                   host_budget,
                bool                     split_route = false,
                uint32_t                 layout      = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
                ggml_type                type        = GGML_TYPE_Q4_0,
                bool                     pageable    = false,
                uint32_t                 cached_layers = all_cached_layers,
                uint32_t                 cache_slots   = n_slots,
                bool                     named_weights = false) {
    int physical_devices = 0;
    CUDA_OK(cudaGetDeviceCount(&physical_devices));
    std::array<int, 2> ordinary_devices = {};
    for (int layer = 0; layer < 2; ++layer) {
        const int device = devices[layer % devices.size()];
        ordinary_devices[layer] = device < physical_devices ? device : 0;
    }
    layer_fixture                 fixture(host_budget, layout, type, pageable, ordinary_devices, cached_layers, cache_slots, named_weights);
    std::vector<ggml_backend_ptr> owners;
    std::vector<ggml_backend_t>   backends;
    for (int device : devices) {
        owners.emplace_back(ggml_backend_cuda_init(device));
        CHECK(owners.back() != nullptr);
        backends.push_back(owners.back().get());
    }
    const bool virtual_devices = std::any_of(devices.begin(), devices.end(), [&](int device) { return device >= physical_devices; });
    if (devices.size() == 2 && devices[0] != devices[1] && devices[0] < physical_devices && devices[1] < physical_devices) {
        test_boundary_copy(backends[0], backends[1], devices[0], devices[1]);
        test_boundary_copy(backends[1], backends[0], devices[1], devices[0]);
    }
    owners.emplace_back(ggml_backend_cpu_init());
    CHECK(owners.back() != nullptr);
    backends.push_back(owners.back().get());
    split_probe probe{ &fixture, backends, {} };
    for (auto * backend : backends) {
        probe.delegates.push_back(backend->iface.graph_compute);
        backend->iface.graph_compute = probe_compute;
    }
    active_probe = &probe;
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 256, true, true));
    CHECK(sched != nullptr);
    const auto allocate = [&](bool affinity = true, bool measure = false, bool reference = false) {
        CHECK(fixture.result.get_moe_regions().size() == 2);
        for (int layer = 0; layer < 2; ++layer) {
            auto * owner  = backends[reference ? 0 : layer % devices.size()];
            auto & region = fixture.result.get_moe_regions()[layer];
            if (!affinity) {
                continue;
            }
            CHECK(region.layer == layer && region.down == fixture.down[layer] && region.route == fixture.ids[layer]);
            CHECK(std::find(region.operations.begin(), region.operations.end(), fixture.input) ==
                  region.operations.end());
            CHECK(std::find(region.inputs.begin(), region.inputs.end(), fixture.logits[layer]) != region.inputs.end());
            region.external_route = true;
            CHECK(!region.place(sched.get(), owner));
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), region.output) == nullptr);
            region.external_route                = false;
            auto unsupported                     = *owner;
            auto unsupported_device              = *ggml_backend_get_device(owner);
            unsupported.device                   = &unsupported_device;
            unsupported_device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * op) {
                return op->op != GGML_OP_ADD;
            };
            std::vector<ggml_tensor *> inplace;
            for (auto * node : region.operations) {
                if (node->op == GGML_OP_ADD) {
                    CHECK(node->view_src == nullptr);
                    node->view_src = node->src[0];
                    inplace.push_back(node);
                }
            }
            CHECK(!inplace.empty());
            CHECK(!region.place(sched.get(), &unsupported));
            for (auto * node : inplace) {
                node->view_src = nullptr;
            }
            for (auto * node : region.operations) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), node) == nullptr);
            }
            CHECK(!region.place(sched.get(), &unsupported));
            for (auto * node : region.operations) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), node) == nullptr);
            }
            region.output->buffer = fixture.down[layer]->buffer;
            CHECK(!region.place(sched.get(), owner));
            region.output->buffer = nullptr;
            for (auto * node : region.operations) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), node) == nullptr);
            }
            CHECK(region.place(sched.get(), owner));
        }
        if (affinity) {
            CHECK(llama_speculative_grouped_intent_test_access::graph_supported(sched.get(), fixture.result.get_gf()));
            for (auto * node : fixture.result.get_moe_regions()[1].operations) {
                if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == fixture.down[1]) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), node, backends.back());
                    CHECK(!llama_speculative_grouped_intent_test_access::graph_supported(sched.get(),
                                                                                         fixture.result.get_gf()));
                    ggml_backend_sched_set_tensor_backend(sched.get(), node, backends[reference ? 0 : 1 % devices.size()]);
                }
            }
        }
        if (affinity && split_route) {
            ggml_backend_sched_set_tensor_backend(sched.get(), fixture.ids[1]->src[0], backends.back());
            ggml_backend_sched_set_tensor_backend(sched.get(), fixture.ids[1], backends.back());
        }
        if (measure) {
            std::vector<size_t> sizes(backends.size());
            ggml_backend_sched_reserve_size(sched.get(), fixture.result.get_gf(), sizes.data());
        } else {
            CHECK(ggml_backend_sched_alloc_graph(sched.get(), fixture.result.get_gf()));
        }
        fprintf(stderr, "test-moe-cache: layer fixture devices=%zu splits=%d host_budget=%zu cached_layers=0x%x slots=%u\n",
                devices.size(), ggml_backend_sched_get_n_splits(sched.get()), host_budget, cached_layers, cache_slots);
        for (int layer = 0; layer < 2; ++layer) {
            if (affinity) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]) ==
                      backends[reference ? 0 : layer % devices.size()]);
                for (auto * node : fixture.result.get_moe_regions()[layer].operations) {
                    auto * actual = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                    if (ggml_is_view(node)) {
                        CHECK(actual == ggml_backend_sched_get_tensor_backend(sched.get(), node->view_src));
                    } else {
                        auto * expected = split_route && layer == 1 && node == fixture.ids[layer]->src[0] ?
                                              backends.back() :
                                              backends[reference ? 0 : layer % devices.size()];
                        CHECK(actual == expected);
                    }
                }
            } else {
                auto * actual = ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]);
                fprintf(stderr, "test-moe-cache: default-placement layer=%d expected=%s actual=%s\n", layer,
                        ggml_backend_name(backends[layer % devices.size()]), ggml_backend_name(actual));
            }
        }
    };
    const auto enabled  = fixture.manifest();
    const auto disabled = candidate_snapshot_v2(cache_slots, nullptr, 0, nullptr, 0);
    const auto publish  = [&](const ggml_backend_moe_candidate_snapshot_v2 & snapshot) {
        ggml_backend_sched_synchronize(sched.get());
        for (size_t i = 0; i < devices.size(); ++i) {
            CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backends[i], &snapshot) ==
                  GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        }
    };
    using layer_outputs = std::array<std::vector<float>, 2>;
    const auto compute = [&](int iteration, layer_outputs * outputs = nullptr) {
        fixture.set_inputs(iteration);
        const auto certificate = layer_certificate();
        CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), fixture.result.get_gf(), &certificate) ==
              GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        if (outputs != nullptr) {
            for (int layer = 0; layer < 2; ++layer) {
                (*outputs)[layer] = active_grouped_tensor_values(fixture.output[layer]);
            }
        }
        return active_grouped_tensor_values(fixture.output[1]);
    };
    std::vector<layer_outputs> ordinary_expected;
    if (cached_layers != 0 && cached_layers != all_cached_layers) {
        // Use ordinary device weights to check both legacy and grouped cache execution.
        layer_fixture ordinary(0, layout, type, pageable, ordinary_devices, 0, cache_slots, named_weights);
        ggml_backend_sched_ptr ordinary_sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 256, true, true));
        CHECK(ordinary_sched != nullptr);
        for (int layer = 0; layer < 2; ++layer) {
            CHECK(ordinary.result.get_moe_regions()[layer].place(ordinary_sched.get(), backends[layer % devices.size()]));
        }
        CHECK(ggml_backend_sched_alloc_graph(ordinary_sched.get(), ordinary.result.get_gf()));
        for (int iteration = 0; iteration < 8; ++iteration) {
            ordinary.set_inputs(iteration);
            CHECK(ggml_backend_sched_graph_compute(ordinary_sched.get(), ordinary.result.get_gf()) == GGML_STATUS_SUCCESS);
            ordinary_expected.emplace_back();
            for (int layer = 0; layer < 2; ++layer) {
                ordinary_expected.back()[layer] = active_grouped_tensor_values(ordinary.output[layer]);
            }
        }
    }
    if (devices.size() == 2) {
        allocate(true, true);
        for (int layer = 0; layer < 2; ++layer) {
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]) == backends[layer]);
        }
        ggml_backend_sched_reset(sched.get());
        fixture.build_graph();
    }
    ggml_backend_buffer_ptr reference_auxiliaries;
    std::array<void *, 2> pageable_bias_data = {};
    std::array<ggml_backend_buffer_t, 2> pageable_bias_buffers = {};
    if (pageable) {
        const size_t size = ggml_nbytes(fixture.down_bias[0]);
        reference_auxiliaries.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_moe_cached_buffer_type(), 2 * size));
        CHECK(reference_auxiliaries != nullptr);
        for (int layer = 0; layer < 2; ++layer) {
            auto * bias = fixture.down_bias[layer];
            std::vector<char> bytes(size);
            ggml_backend_tensor_get(bias, bytes.data(), 0, size);
            pageable_bias_data[layer] = bias->data;
            pageable_bias_buffers[layer] = bias->buffer;
            bias->data = static_cast<char *>(ggml_backend_buffer_get_base(reference_auxiliaries.get())) + layer * size;
            bias->buffer = reference_auxiliaries.get();
            memcpy(bias->data, bytes.data(), size);
        }
    }
    publish(disabled);
    allocate(false);
    ggml_backend_sched_reset(sched.get());
    fixture.build_graph();
    // The legacy reference uses physical GPU 0; its raw CUDA ordinal API does not emulate devices.
    allocate(true, false, virtual_devices);
    std::vector<std::vector<float>> expected;
    std::vector<layer_outputs> expected_layers;
    for (int iteration = 0; iteration < 8; ++iteration) {
        expected_layers.emplace_back();
        expected.push_back(compute(iteration, &expected_layers.back()));
        if (!ordinary_expected.empty()) {
            for (int layer = 0; layer < 2; ++layer) {
                check_active_grouped_exact_output(ordinary_expected[iteration][layer], expected_layers.back()[layer]);
            }
        }
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::legacy_debug_telemetry(
            *ggml_cuda_moe_grouped_context_for_test(backends[i]), true);
        fprintf(stderr, "test-moe-cache: route publication cached_layers=0x%x owner=%zu slots=%u named=%d ops=%llu reused_ids=%llu d2h_syncs=%llu publications=%zu\n",
            cached_layers, i, cache_slots, named_weights, (unsigned long long) telemetry.ops,
            (unsigned long long) telemetry.ids_cache_hits, (unsigned long long) telemetry.ids_d2h_sync_count,
            ggml_cuda_moe_ids_cache_count_for_test(backends[i]));
        if (cached_layers == 0) {
            CHECK(ggml_cuda_moe_ids_cache_count_for_test(backends[i]) == 0);
        }
        const char * disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION");
        if (named_weights && cached_layers != 0 && !split_route && !virtual_devices &&
                (!disable_fusion || std::atoi(disable_fusion) == 0)) {
            CHECK(telemetry.ids_d2h_sync_count == 0);
        }
        if (!split_route && !virtual_devices && (!disable_fusion || std::atoi(disable_fusion) == 0)) {
            size_t expected_publications = 0;
            for (int layer = 0; layer < 2; ++layer) {
                expected_publications += (cached_layers & (1u << layer)) != 0 && layer % devices.size() == i;
            }
            CHECK(ggml_cuda_moe_ids_cache_count_for_test(backends[i]) == expected_publications);
        }
    }
    if (pageable) {
        for (int layer = 0; layer < 2; ++layer) {
            fixture.down_bias[layer]->data = pageable_bias_data[layer];
            fixture.down_bias[layer]->buffer = pageable_bias_buffers[layer];
        }
    }
    publish(enabled);
    ggml_backend_sched_reset(sched.get());
    fixture.build_graph();
    allocate();
    std::vector<uint64_t> legacy_before;
    for (size_t i = 0; i < devices.size(); ++i) {
        legacy_before.push_back(active_grouped_legacy_op_count(backends[i]));
    }
    for (int iteration = 0; iteration < 8; ++iteration) {
        layer_outputs actual;
        const auto final = compute(iteration, &actual);
        check_active_grouped_exact_output(expected[iteration], final);
        for (int layer = 0; layer < 2; ++layer) {
            check_active_grouped_exact_output(expected_layers[iteration][layer], actual[layer]);
        }
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(backends[i]);
        CHECK(context != nullptr);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        fprintf(stderr, "test-moe-cache: device=%d grouped_completed=%llu legacy_ops=%llu\n", devices[i],
                (unsigned long long) telemetry.completed,
                (unsigned long long) active_grouped_legacy_op_count(backends[i]));
        const bool fallback_owner = split_route && (devices.size() == 1 || i == 1);
        CHECK(telemetry.decode_legacy == (fallback_owner ? 8 : 0));
        CHECK(telemetry.fallback == telemetry.decode_legacy);
        uint64_t expected_completed = 0;
        for (int layer = 0; layer < 2; ++layer) {
            if ((cached_layers & (1u << layer)) && layer % devices.size() == i &&
                !(split_route && layer == 1)) {
                expected_completed += 8;
            }
        }
        CHECK(telemetry.completed == expected_completed && telemetry.decode_grouped == expected_completed);
        CHECK(telemetry.ready == expected_completed);
        if (!fallback_owner && expected_completed > 0) {
            CHECK(active_grouped_legacy_op_count(backends[i]) == legacy_before[i]);
            if (getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
                CHECK(telemetry.captures > 0 && telemetry.replays > 0);
            } else {
                CHECK(telemetry.direct > 0 && telemetry.captures == 0 && telemetry.replays == 0);
            }
        }
        CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        CHECK(telemetry.submitted == telemetry.completed);
        if (cached_layers == 0) {
            CHECK(telemetry.admitted_banks == 0 && telemetry.h2d_banks == 0 && telemetry.h2d_bytes == 0);
        }
        for (int layer = 0; layer < 2; ++layer) {
            ggml_cuda_moe_candidate_group_key key;
            if (cached_layers & (1u << layer)) {
                CHECK(context->find_down_group_key(fixture.down[layer], &key));
                ggml_cuda_moe_candidate_group_info info;
                CHECK(context->get_group(key, &info) && info.semantic_group_index == (uint32_t) layer);
                if (layer % devices.size() != i) {
                    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
                }
            } else {
                CHECK(!context->find_down_group_key(fixture.down[layer], &key));
            }
        }
    }
    CHECK(probe.grouped_splits[0] > 0 && probe.imported_splits[0] == 0);
    CHECK(split_route ? probe.imported_splits[1] > 0 : probe.imported_splits[1] == 0);
    CHECK(split_route || probe.grouped_splits[1] > 0);

    if (split_route || devices.size() == 2) {
        probe.fail_layer       = 1;
        const auto certificate = layer_certificate();
        CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), fixture.result.get_gf(), &certificate) ==
              GGML_STATUS_FAILED);
        ggml_backend_sched_synchronize(sched.get());
        CHECK(probe.failed);  // No downstream split may run after the injected failure.
        if (devices.size() == 2) {
            for (size_t i = 0; i < devices.size(); ++i) {
                const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(
                    *ggml_cuda_moe_grouped_context_for_test(backends[i]));
                CHECK(telemetry.submitted == (i == 0 && (cached_layers & 1u) ? 1 : 0) && telemetry.completed == telemetry.submitted);
                CHECK(telemetry.fallback == 0);
            }
        }
        // Failed graph outputs can alias scratch. Read them only after successful recomputation.
        probe.failed     = false;
        probe.fail_layer = -1;
        check_active_grouped_exact_output(expected[0], compute(0));
    }
    if (split_route) {
        const auto required        = layer_certificate(true);
        const int  imported_before = probe.imported_splits[1];
        CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), fixture.result.get_gf(), &required) !=
              GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        CHECK(probe.imported_splits[1] > imported_before);
    } else if (devices.size() == 2) {
        auto *     first             = ggml_cuda_moe_grouped_context_for_test(backends[0]);
        auto *     second            = ggml_cuda_moe_grouped_context_for_test(backends[1]);
        const auto first_generation  = first->state().generation;
        const auto second_generation = second->state().generation;
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backends[1], &enabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(first->state().generation == first_generation && second->state().generation > second_generation);
        ggml_backend_sched_reset(sched.get());
        fixture.build_graph();
        allocate();
        check_active_grouped_exact_output(expected[1], compute(1));
    }
    if (!split_route) {
        fixture.set_inputs(0);
        const auto required = layer_certificate(true);
        CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), fixture.result.get_gf(), &required) ==
              GGML_STATUS_SUCCESS);
        check_active_grouped_exact_output(expected[0], active_grouped_tensor_values(fixture.output[1]));
    }
    ggml_backend_sched_synchronize(sched.get());
    sched.reset();
    for (size_t i = 0; i < backends.size(); ++i) {
        backends[i]->iface.graph_compute = probe.delegates[i];
    }
    active_probe = nullptr;
    int survivor_layer = -1;
    for (int layer = 0; layer < 2; ++layer) {
        if ((cached_layers & (1u << layer)) &&
            (survivor_layer < 0 || layer % devices.size() == devices.size() - 1)) {
            survivor_layer = layer;
        }
    }
    if (!split_route && !virtual_devices && survivor_layer >= 0) {
        const size_t survivor_owner = survivor_layer % devices.size();
        if (devices.size() == 2) {
            owners[1 - survivor_owner].reset();
        }
        auto *                        survivor = backends[survivor_owner];
        active_grouped_dispatch_graph shared;
        shared.n_experts = n_experts;
        shared.n_used    = n_used;
        for (const auto & bank : fixture.tensors) {
            if (bank.group_index == (uint32_t) survivor_layer) {
                if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                    shared.banks.push_back(const_cast<ggml_tensor *>(bank.tensor));
                    shared.roles.push_back(bank.role);
                } else {
                    CHECK(bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS);
                    shared.biases.push_back(const_cast<ggml_tensor *>(bank.tensor));
                    shared.bias_roles.push_back(bank.role);
                }
            }
        }
        auto graph = build_active_grouped_dispatch_graph(survivor, fixture.buft, type, layout, false, 1, n_experts,
                                                         n_used, n_dim, &shared);
        // Fill only graph inputs; the shared immutable weights remain unchanged.
        const auto input = cached_fusion_test_data(graph.input, 71);
        ggml_backend_tensor_set(graph.input, input.data(), 0, input.size());
        set_active_grouped_dispatch_logits({ &graph }, 0);
        if (pageable) {
            fixture.down_bias[survivor_layer]->data = static_cast<char *>(ggml_backend_buffer_get_base(reference_auxiliaries.get())) +
                survivor_layer * ggml_nbytes(fixture.down_bias[survivor_layer]);
            fixture.down_bias[survivor_layer]->buffer = reference_auxiliaries.get();
        }
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(survivor, &disabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        const auto reference = run_active_grouped_dispatch(survivor, graph, 0);
        if (pageable) {
            fixture.down_bias[survivor_layer]->data = pageable_bias_data[survivor_layer];
            fixture.down_bias[survivor_layer]->buffer = pageable_bias_buffers[survivor_layer];
        }
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(survivor, &enabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(survivor);
        CHECK(context != nullptr);
        (void) ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        const auto legacy_before = active_grouped_legacy_op_count(survivor);
        const auto capability =
            native_mmid_capability(devices[survivor_owner], graph.banks[0], graph.n_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
        const char * disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION");
        // F3 writes the GLU result directly and leaves the gate/up sentinels intact.
        const bool   f3_skipped =
            (!disable_fusion || std::atoi(disable_fusion) == 0) &&
            layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ && graph.banks[0]->type == graph.banks[1]->type &&
            ggml_are_same_shape(graph.banks[0], graph.banks[1]) && ggml_are_same_stride(graph.banks[0], graph.banks[1]);
        fprintf(stderr, "test-moe-cache: survivor device=%d layout=%u type=%s generation=%llu f3_skipped=%d\n",
                devices[survivor_owner], layout, ggml_type_name(type), (unsigned long long) context->state().generation,
                f3_skipped);
        for (int pass = 0; pass < 3; ++pass) {
            check_active_grouped_exact_output(reference, run_active_grouped_dispatch(survivor, graph, 0, f3_skipped));
        }
        ggml_backend_synchronize(survivor);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(telemetry.decode_grouped == 3 && telemetry.ready == 3 && telemetry.submitted == 3 &&
              telemetry.completed == 3);
        CHECK(telemetry.decode_legacy == 0 && telemetry.fallback == 0 &&
              active_grouped_legacy_op_count(survivor) == legacy_before);
        CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        if (getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
            CHECK(telemetry.direct == 1 && telemetry.captures == 1 && telemetry.replays == 1);
        } else {
            CHECK(telemetry.direct == 3 && telemetry.captures == 0 && telemetry.replays == 0);
        }
    }
    owners.clear();
    if (host_budget != 0) {
        auto * budget = static_cast<moe_host_budget *>(fixture.buft->context);
        CHECK(budget != nullptr);
        std::lock_guard<std::mutex> lock(budget->mutex);
        CHECK(budget->source_bytes + budget->staging_peak <= budget->limit);
        CHECK(budget->materialized_bytes > 0);
        CHECK(budget->staging_bytes == 0 && budget->staging_optional_bytes == 0);
    }
    fixture.result.reset();
    CHECK(fixture.result.get_moe_regions().empty());
}

void test_shared_source_owners(int device, size_t host_budget, bool pageable = false) {
    ggml_backend_ptr first(ggml_backend_cuda_init(device));
    ggml_backend_ptr second(ggml_backend_cuda_init(device));
    ggml_backend_ptr third(ggml_backend_cuda_init(device));
    CHECK(first && second && third);
    auto * buft = pageable ? pageable_cached_buffer_type() : ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
    auto   a    = build_active_grouped_dispatch_graph(first.get(), buft, GGML_TYPE_Q4_0,
                                                      GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, pageable);
    auto   b = build_active_grouped_dispatch_graph(second.get(), buft, GGML_TYPE_Q4_0,
                                                   GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, pageable, 1, n_experts,
                                                   n_used, n_dim, &a);
    auto c = build_active_grouped_dispatch_graph(third.get(), buft, GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, pageable, 1, n_experts, n_used, n_dim, &a);
    initialize_active_grouped_dispatch_graphs({ &a, &b, &c });
    candidate_stamp_execution(b.graph, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    candidate_stamp_execution(c.graph, GGML_GRAPH_EXECUTION_DOMAIN_MTP, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, 1, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    c.graph->execution_certificate.owner_namespace++;
    ggml_backend_moe_candidate_group_v2               group{ GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
                                                             GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0 };
    std::vector<ggml_backend_moe_candidate_tensor_v2> banks;
    for (size_t i = 0; i < a.banks.size(); ++i) {
        banks.push_back({ a.banks[i], 0, a.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
                          GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0 });
    }
    if (pageable) {
        banks.push_back({a.down_scale, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE, GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0});
    }
    const auto snapshot = candidate_snapshot_v2(n_slots, &group, 1, banks.data(), banks.size());
    if (host_budget) {
        CHECK(ggml_backend_cuda_moe_cached_configure_sources(buft, &snapshot));
    }
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(first.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(second.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(third.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    (void) candidate_certify_graph(*ggml_cuda_moe_grouped_context_for_test(second.get()), b.graph);
    (void) candidate_certify_graph(*ggml_cuda_moe_grouped_context_for_test(third.get()), c.graph);
    const auto expected = run_active_grouped_dispatch(first.get(), a, 0);
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(second.get(), b, 0));
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(third.get(), c, 0));
    std::vector<float> draft_output, mtp_output;
    std::thread draft([&] { draft_output = run_active_grouped_dispatch(second.get(), b, 0); });
    std::thread mtp([&] { mtp_output = run_active_grouped_dispatch(third.get(), c, 0); });
    draft.join();
    mtp.join();
    check_active_grouped_exact_output(expected, draft_output);
    check_active_grouped_exact_output(expected, mtp_output);
    const auto generation = ggml_cuda_moe_grouped_context_for_test(second.get())->state().generation;
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(first.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_cuda_moe_grouped_context_for_test(second.get())->state().generation == generation);
    first.reset();
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(second.get(), b, 0));
    second.reset();
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(third.get(), c, 0));
    CHECK(active_grouped_legacy_op_count(third.get()) == 0);
    third.reset();
    if (host_budget) {
        auto * budget = static_cast<moe_host_budget *>(buft->context);
        CHECK(budget->staging_bytes == 0 && budget->staging_peak + budget->source_bytes <= budget->limit);
    }
    ggml_backend_cuda_moe_cached_free_buffer_type(buft);
    fprintf(stderr, "test-moe-cache: three shared-source owners replacement/teardown pageable=%d OK (one physical device)\n", pageable);
}

void test_hybrid_ffn_insertion() {
    ggml_backend_ptr cuda(ggml_backend_cuda_init(0));
    CHECK(cuda != nullptr);
    auto params = ggml_threadpool_params_default(2);
    auto pool = ggml_threadpool_new(&params);
    CHECK(pool != nullptr);
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    CHECK(cpu != nullptr);
    ggml_backend_cpu_set_threadpool(cpu.get(), pool);
    ggml_backend_cpu_set_n_threads(cpu.get(), params.n_threads);
    size_t parity_failures = 0;
    double max_nmse = 0.0;
    double bound = 0.0;
    const auto compare = [&](const std::vector<float> & reference, const std::vector<float> & actual,
                             const char * stage, uint32_t layout, int activation, int n_rows, int iteration, int mode) {
        CHECK(reference.size() == actual.size());
        double error = 0.0, norm = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            CHECK(std::isfinite(reference[i]) && std::isfinite(actual[i]));
            const double difference = (double) actual[i] - reference[i];
            error += difference * difference;
            norm += (double) reference[i] * reference[i];
        }
        const double nmse = norm > 0.0 ? error / norm : error;
        max_nmse = std::max(max_nmse, nmse);
        if (nmse > bound) {
            ++parity_failures;
            fprintf(stderr, "hybrid FFN insertion: stage=%s layout=%u activation=%d rows=%d iteration=%d mode=%d NMSE=%g bound=%g\n",
                stage, layout, activation, n_rows, iteration, mode, nmse, bound);
        }
    };
    // Quantized banks use the per-op MUL_MAT_ID bound from test-backend-ops; this FFN chains several such matmuls.
    for (auto probe_type : {GGML_TYPE_BF16, GGML_TYPE_Q4_0}) {
    bound = ggml_is_quantized(probe_type) ? 5e-4 : 2e-5;
    max_nmse = 0.0;
    for (auto layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
                        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED}) {
        for (auto activation : {LLM_FFN_SILU, LLM_FFN_GELU}) {
            for (int n_rows : {1, 4}) {
                layer_fixture host(0, layout, probe_type);
                layer_fixture gpu(0, layout, probe_type, false, {0, 0}, 0);
                host.build_graph(activation, 0.0f, false, false, n_rows);
                gpu.build_graph(activation, 0.0f, false, false, n_rows);
                const auto & host_region = host.result.get_moe_regions().front();
                ggml_set_output(host_region.hybrid_ffn.down_output);
                ggml_set_output(host_region.route);
                ggml_gallocr_ptr host_allocator(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
                CHECK(ggml_gallocr_alloc_graph(host_allocator.get(), host.result.get_gf()));
                const auto & region = gpu.result.get_moe_regions().front();
                const auto & ffn = region.hybrid_ffn;
                auto * graph = gpu.result.get_gf();
                auto * masked = ggml_new_tensor_2d(gpu.result.get_ctx(), GGML_TYPE_I32, n_used, n_rows);
                ggml_set_input(masked);
                ggml_set_output(region.route);
                ggml_set_output(ffn.input);
                ggml_set_output(ffn.down_output);
                const auto span = llm_graph_moe_hybrid_find_span(graph, region);
                CHECK(span.valid());
                const int first = span.routing_end, down_end = span.expert_end, output_end = span.region_end;
                CHECK(first > 0 && down_end > first && output_end > down_end);
                for (int i = first; i < down_end; ++i) {
                    auto * node = ggml_graph_node(graph, i);
                    if (node->op == GGML_OP_MUL_MAT_ID &&
                            (node->src[0] == ffn.up || node->src[0] == ffn.gate || node->src[0] == ffn.gate_up || node->src[0] == ffn.down)) {
                        node->src[2] = masked;
                    }
                }
                ggml_build_forward_expand(graph, masked);
                auto * buft = ggml_backend_cuda_buffer_type(0);
                ggml_backend_buffer_ptr raw_buffer(ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, ffn.down_output)));
                CHECK(raw_buffer != nullptr);
                CHECK(ggml_backend_tensor_alloc(raw_buffer.get(), ffn.down_output, ggml_backend_buffer_get_base(raw_buffer.get())) == GGML_STATUS_SUCCESS);
                ggml_gallocr_ptr allocator(ggml_gallocr_new(buft));
                CHECK(ggml_gallocr_alloc_graph(allocator.get(), graph));
                auto routing = ggml_graph_view(graph, 0, first);
                auto experts = ggml_graph_view(graph, first, down_end);
                auto reduction = ggml_graph_view(graph, down_end, output_end);
                llm_graph_moe_hybrid_executor executor;
                CHECK(executor.init(host.result.get_moe_regions().front().hybrid_ffn, pool, params.n_threads,
                    n_rows * n_used, n_experts, n_experts));
                CHECK(!executor.init(host.result.get_moe_regions().front().hybrid_ffn, pool, params.n_threads,
                    n_rows * n_used, n_experts, n_experts));
                executor.certify_domain(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, true);
                CHECK(executor.domain_certified(GGML_GRAPH_EXECUTION_DOMAIN_MAIN) && executor.workspace_bytes() != 0);
                ggml_graph_execution_certificate certificate = layer_certificate();
                // A single row cannot be target verification; verification needs fewer sequences than rows.
                certificate.row_semantics = n_rows > 1 ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE :
                    GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
                certificate.n_rows = n_rows;
                certificate.n_sequences = 1;
                llm_graph_moe_hybrid_routes packet;
                CHECK(packet.init(n_rows * n_used, n_experts, n_experts));
                const auto & executor_gpu_ids = executor.gpu_ids();
                std::vector<float> input(n_rows * n_dim), raw(n_rows * n_used * n_dim), cold(raw.size()), expected_raw(raw.size()), cpu_raw(raw.size());
                std::vector<float> result(n_rows * n_dim), expected(result.size()), poison(raw.size(), NAN);
                std::vector<uint8_t> ids(ggml_nbytes(region.route));
                float * staging = nullptr;
                CUDA_OK(cudaMallocHost(&staging, raw.size() * sizeof(float)));
                for (int iteration : {0, 3}) {
                    gpu.set_inputs(iteration);
                    CHECK(ggml_backend_graph_compute(cuda.get(), &routing) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(region.route, ids.data(), 0, ids.size());
                    ggml_backend_tensor_get(ffn.input, input.data(), 0, input.size() * sizeof(float));
                    host.set_inputs(iteration);
                    ggml_backend_tensor_set(host.input, input.data(), 0, input.size() * sizeof(float));
                    CHECK(ggml_backend_graph_compute(cpu.get(), host.result.get_gf()) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(host_region.hybrid_ffn.down_output, cpu_raw.data(), 0, cpu_raw.size() * sizeof(float));
                    std::vector<uint8_t> cpu_ids(ids.size());
                    ggml_backend_tensor_get(host_region.route, cpu_ids.data(), 0, cpu_ids.size());
                    CHECK(cpu_ids == ids);
                    std::array<int32_t, n_experts> slots;
                    std::iota(slots.begin(), slots.end(), 0);
                    CHECK(packet.build(ids.data(), ids.size(), region.route->nb[1], n_used, n_rows, slots.data(), slots.size()));
                    ggml_backend_tensor_set(masked, packet.gpu_ids().data(), 0, ggml_nbytes(masked));
                    CHECK(ggml_backend_graph_compute(cuda.get(), &experts) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(ffn.down_output, expected_raw.data(), 0, expected_raw.size() * sizeof(float));
                    CHECK(ggml_backend_graph_compute(cuda.get(), &reduction) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(gpu.output[0], expected.data(), 0, expected.size() * sizeof(float));
                    for (int mode : {0, 1, 2}) {
                        for (int expert = 0; expert < n_experts; ++expert) {
                            slots[expert] = mode == 2 || (mode == 1 && expert % 2 == 0) ? expert : -1;
                        }
                        CHECK(packet.build(ids.data(), ids.size(), region.route->nb[1], n_used, n_rows, slots.data(), slots.size()));
                        ggml_backend_tensor_set(masked, packet.gpu_ids().data(), 0, ggml_nbytes(masked));
                        ggml_backend_tensor_set(ffn.down_output, poison.data(), 0, poison.size() * sizeof(float));
                        CHECK(executor.begin(certificate, region.layer, region.semantic_group, ids.data(), ids.size(),
                            region.route->nb[1], n_used, n_rows, slots.data(), slots.size(), input.data(), cold.data()) ==
                            llm_graph_moe_hybrid_admission::ADMITTED);
                        CHECK(executor.busy() && executor_gpu_ids == packet.gpu_ids() &&
                            executor.cold_columns() == packet.cold_columns());
                        CHECK(ggml_backend_graph_compute_async(cuda.get(), &experts) == GGML_STATUS_SUCCESS);
                        // A second begin cannot displace work already in flight.
                        CHECK(executor.begin(certificate, region.layer, region.semantic_group, ids.data(), ids.size(),
                            region.route->nb[1], n_used, n_rows, slots.data(), slots.size(), input.data(), cold.data()) !=
                            llm_graph_moe_hybrid_admission::ADMITTED);
                        CHECK(executor.wait());
                        CHECK(!executor.busy() && !executor.wait());
                        ggml_backend_synchronize(cuda.get());
                        ggml_backend_tensor_get(ffn.down_output, raw.data(), 0, raw.size() * sizeof(float));
                        for (size_t column = 0; column < packet.gpu_ids().size(); ++column) {
                            if (packet.gpu_ids()[column] >= 0) {
                                CHECK(memcmp(raw.data() + column * n_dim, expected_raw.data() + column * n_dim, n_dim * sizeof(float)) == 0);
                            } else {
                                const bool untouched = std::all_of(raw.begin() + column * n_dim, raw.begin() + (column + 1) * n_dim,
                                    [](float value) { return std::isnan(value); });
                                if (!untouched) {
                                    fprintf(stderr, "hybrid FFN cold write: layout=%u activation=%d rows=%d iteration=%d mode=%d column=%zu value=%g\n",
                                        layout, activation, n_rows, iteration, mode, column, raw[column * n_dim]);
                                }
                                CHECK(untouched);
                            }
                        }
                        for (size_t job = 0; job < packet.cold_ids().size(); ++job) {
                            const auto column = packet.cold_columns()[job];
                            for (int i = 0; i < n_dim; ++i) {
                                const float reference = cpu_raw[column * n_dim + i];
                                CHECK(cold[job * n_dim + i] == reference);
                            }
                            ggml_backend_tensor_set(ffn.down_output, cold.data() + job * n_dim, column * n_dim * sizeof(float), n_dim * sizeof(float));
                        }
                        ggml_backend_tensor_get(ffn.down_output, raw.data(), 0, raw.size() * sizeof(float));
                        if (mode == 0) {
                            compare(expected_raw, cpu_raw, "full-CPU-oracle", layout, activation, n_rows, iteration, mode);
                        }
                        compare(expected_raw, raw, "raw", layout, activation, n_rows, iteration, mode);
                        CHECK(ggml_backend_graph_compute(cuda.get(), &reduction) == GGML_STATUS_SUCCESS);
                        ggml_backend_tensor_get(gpu.output[0], result.data(), 0, result.size() * sizeof(float));
                        compare(expected, result, "reduction", layout, activation, n_rows, iteration, mode);

                        // repeat without an intermediate drain: stream-ordered insertion from pinned staging
                        ggml_backend_tensor_set(ffn.down_output, poison.data(), 0, poison.size() * sizeof(float));
                        std::fill(cold.begin(), cold.end(), NAN);
                        CHECK(executor.begin(certificate, region.layer, region.semantic_group, ids.data(), ids.size(),
                            region.route->nb[1], n_used, n_rows, slots.data(), slots.size(), input.data(), cold.data()) ==
                            llm_graph_moe_hybrid_admission::ADMITTED);
                        CHECK(ggml_backend_graph_compute_async(cuda.get(), &experts) == GGML_STATUS_SUCCESS);
                        CHECK(executor.wait());
                        for (size_t job = 0; job < packet.cold_ids().size(); ++job) {
                            const auto column = packet.cold_columns()[job];
                            memcpy(staging + job * n_dim, cold.data() + job * n_dim, n_dim * sizeof(float));
                            ggml_backend_tensor_set_async(cuda.get(), ffn.down_output, staging + job * n_dim,
                                column * n_dim * sizeof(float), n_dim * sizeof(float));
                        }
                        CHECK(ggml_backend_graph_compute_async(cuda.get(), &reduction) == GGML_STATUS_SUCCESS);
                        ggml_backend_synchronize(cuda.get());
                        std::vector<float> streamed(result.size());
                        ggml_backend_tensor_get(gpu.output[0], streamed.data(), 0, streamed.size() * sizeof(float));
                        CHECK(streamed == result);
                    }
                }
                CUDA_OK(cudaFreeHost(staging));
            }
        }
    }
    fprintf(stderr, "test-moe-cache: hybrid FFN insertion type=%s max_NMSE=%g bound=%g\n",
        ggml_type_name(probe_type), max_nmse, bound);
    }
    cpu.reset();
    ggml_threadpool_free(pool);
    fprintf(stderr, "test-moe-cache: hybrid FFN insertion parity failures=%zu\n", parity_failures);
    CHECK(parity_failures == 0);
    fprintf(stderr, "test-moe-cache: CUDA/CPU hybrid FFN insertion and unchanged reduction OK\n");
}

// Frozen partition: residency is whatever is already committed, so no decode is planned and nothing is filled.
void test_hybrid_frozen_partition() {
    constexpr uint32_t frozen_slots = 4;
    ggml_backend_ptr cuda(ggml_backend_cuda_init(0));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    CHECK(cuda && cpu);
    auto params = ggml_threadpool_params_default(2);
    auto * pool = ggml_threadpool_new(&params);
    CHECK(pool != nullptr);
    ggml_backend_cpu_set_threadpool(cpu.get(), pool);
    ggml_backend_cpu_set_n_threads(cpu.get(), params.n_threads);
    size_t total_cold = 0, total_resident_all = 0, total_mixed = 0, total_hot = 0, total_layers = 0;
    double half_difference = 0.0, half_square = 0.0;
    size_t half_count = 0;
    for (int n_rows : {1, 4}) {
        layer_fixture cached(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, false, {0, 0},
                             all_cached_layers, frozen_slots);
        layer_fixture host(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0);
        cached.build_graph(LLM_FFN_SILU, 0.0f, false, false, n_rows);
        host.build_graph(LLM_FFN_SILU, 0.0f, false, false, n_rows);
        const auto & region = cached.result.get_moe_regions().front();
        const auto & host_region = host.result.get_moe_regions().front();
        ggml_set_output(host_region.hybrid_ffn.down_output);
        ggml_set_output(host_region.route);
        ggml_gallocr_ptr host_allocator(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
        CHECK(ggml_gallocr_alloc_graph(host_allocator.get(), host.result.get_gf()));
        const auto manifest = cached.manifest();
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(cuda.get(), &manifest) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(cuda.get());
        CHECK(context != nullptr);
        ggml_backend_t backends[] = {cuda.get(), cpu.get()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 512, false, true));
        CHECK(sched != nullptr);
        auto * graph = cached.result.get_gf();
        for (auto & placed : cached.result.get_moe_regions()) {
            CHECK(placed.place(sched.get(), cuda.get()));
        }
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
        ggml_set_output(region.route);
        ggml_set_output(region.hybrid_ffn.input);
        // Warm residency through the ordinary cached path; the frozen phase below must add nothing.
        // DRAFT independent stays grouped-eligible when routes exceed slots, which is the case being frozen.
        auto certificate = layer_certificate(true);
        certificate.n_rows = n_rows;
        certificate.n_sequences = n_rows;
        cached.set_inputs(0);
        CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        // Residency lives in the backend context, so a cut reallocation keeps what warming filled.
        ggml_backend_sched_reset(sched.get());
        cached.build_graph(LLM_FFN_SILU, 0.0f, false, false, n_rows);
        graph = cached.result.get_gf();
        const auto & routed = cached.result.get_moe_regions().front();
        for (auto & placed : cached.result.get_moe_regions()) {
            CHECK(placed.place(sched.get(), cuda.get()));
        }
        ggml_set_output(routed.route);
        ggml_set_output(routed.hybrid_ffn.input);
        ggml_set_output(routed.hybrid_ffn.down_output);
        ggml_set_output(cached.output[0]);
        const auto span = llm_graph_moe_hybrid_find_span(graph, routed);
        CHECK(span.valid());
        const int routing_end = span.routing_end;
        const int boundaries[] = {routing_end, span.expert_end, span.region_end};
        CHECK(ggml_backend_sched_set_split_boundaries(sched.get(), graph, boundaries,
            span.region_end < graph->n_nodes ? 3 : 2));
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
        const ggml_cuda_moe_candidate_group_key candidate{context->state().generation, 0};
        const auto stream = ggml_cuda_moe_owner_stream_for_test(cuda.get());
        CHECK(stream != nullptr);
        int32_t * device_slots = nullptr;
        CUDA_OK(cudaMalloc(&device_slots, n_experts * sizeof(int32_t)));
        llm_graph_moe_hybrid_routes packet;
        llm_graph_moe_hybrid_executor executor;
        CHECK(packet.init(n_rows * n_used, n_experts, frozen_slots));
        CHECK(executor.init(host_region.hybrid_ffn, pool, params.n_threads, n_rows * n_used, n_experts, frozen_slots));
        executor.certify_domain(certificate.domain, true);
        std::vector<uint8_t> ids(ggml_nbytes(region.route));
        std::vector<float> input(n_rows * n_dim), cold(n_rows * n_used * n_dim), oracle(cold.size());
        size_t cold_columns = 0, mixed_partitions = 0, hot_halves = 0, hybrid_layers = 0;
        uint32_t total_resident = 0;

        for (int iteration : {1, 3, 5}) {
            // Holding a transaction blocks the ordinary cached path, so route first, then freeze.
            ggml_cuda_moe_grouped_acquisition acquisition;
            ggml_cuda_moe_grouped_transaction held;
            ggml_backend_buffer_ptr scratch;
            int32_t * device_hot_slots = nullptr;
            int32_t * device_hot_rows = nullptr;
            int32_t * device_hot_columns = nullptr;
            int32_t * device_counts = nullptr;
            // Routing only: later batches get routes without executing or filling any expert.
            cached.set_inputs(iteration);
            CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate, 0, routing_end) ==
                GGML_STATUS_SUCCESS);
            ggml_backend_sched_synchronize(sched.get());
            ggml_backend_tensor_get(routed.route, ids.data(), 0, ids.size());
            ggml_backend_tensor_get(routed.hybrid_ffn.input, input.data(), 0, input.size() * sizeof(float));
            host.set_inputs(iteration);
            ggml_backend_tensor_set(host.input, input.data(), 0, input.size() * sizeof(float));
            CHECK(ggml_backend_graph_compute(cpu.get(), host.result.get_gf()) == GGML_STATUS_SUCCESS);
            std::vector<uint8_t> host_ids(ids.size());
            ggml_backend_tensor_get(host_region.route, host_ids.data(), 0, host_ids.size());
            CHECK(host_ids == ids);
            ggml_backend_tensor_get(host_region.hybrid_ffn.down_output, oracle.data(), 0, oracle.size() * sizeof(float));
            CHECK(context->acquire_group_resources(candidate, &acquisition));
            CHECK(context->begin_group_transaction(acquisition, &held));
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
            // The committed host map can lag a deferred commit, so the device snapshot is the only authority.
            const auto snapshot = [&]() {
                std::array<int32_t, n_experts> slots;
                slots.fill(-9);
                CUDA_OK(cudaMemcpyAsync(device_slots, slots.data(), sizeof(slots), cudaMemcpyHostToDevice, stream));
                CHECK(context->snapshot_resident_slots(held, stream, device_slots, slots.size()));
                CUDA_OK(cudaMemcpyAsync(slots.data(), device_slots, sizeof(slots), cudaMemcpyDeviceToHost, stream));
                CUDA_OK(cudaStreamSynchronize(stream));
                return slots;
            };
            CHECK(!context->snapshot_resident_slots(held, stream, device_slots, n_experts - 1));
            const auto resident = snapshot();
            uint32_t n_resident = 0;
            for (int expert = 0; expert < n_experts; ++expert) {
                CHECK(resident[expert] >= -1 && resident[expert] < (int32_t) frozen_slots);
                n_resident += resident[expert] >= 0;
            }
            // A shape whose routes exceed the slot pool can hold nothing resident; then every column is cold.
            CHECK(n_resident <= frozen_slots);
            total_resident += n_resident;
            CHECK(packet.build(ids.data(), ids.size(), routed.route->nb[1], n_used, n_rows, resident.data(), resident.size()));
            CHECK(executor.begin(certificate, routed.layer, routed.semantic_group, ids.data(), ids.size(),
                routed.route->nb[1], n_used, n_rows, resident.data(), resident.size(), input.data(), cold.data()) ==
                llm_graph_moe_hybrid_admission::ADMITTED);
            CHECK(executor.wait());
            const auto * routes = reinterpret_cast<const int32_t *>(ids.data());
            const size_t row_stride = routed.route->nb[1] / sizeof(int32_t);
            size_t cold_index = 0;
            for (int row = 0; row < n_rows; ++row) {
                for (int used = 0; used < n_used; ++used) {
                    const int column = row * n_used + used;
                    const int32_t expert = routes[row * row_stride + used];
                    CHECK(packet.gpu_ids()[column] == resident[expert]);
                    if (resident[expert] >= 0) {
                        continue;
                    }
                    CHECK(packet.cold_ids()[cold_index] == expert && packet.cold_columns()[cold_index] == column);
                    for (int i = 0; i < n_dim; ++i) {
                        CHECK(cold[cold_index * n_dim + i] == oracle[column * n_dim + i]);
                    }
                    ++cold_index;
                }
            }
            CHECK(cold_index == packet.cold_ids().size());
            // Strata's shape: a compact resident list, each entry carrying the router index it must land on.
            CHECK(packet.hot_slots().size() + packet.cold_ids().size() == packet.gpu_ids().size());
            CHECK(packet.hot_slots().size() == packet.hot_columns().size() &&
                packet.hot_experts().size() == packet.hot_slots().size() &&
                packet.hot_rows().size() == packet.hot_slots().size());
            std::vector<int> owner(packet.gpu_ids().size(), 0);
            for (size_t job = 0; job < packet.hot_slots().size(); ++job) {
                const auto column = packet.hot_columns()[job];
                CHECK(column >= 0 && (size_t) column < owner.size() && owner[column]++ == 0);
                CHECK(packet.hot_rows()[job] == column / n_used);
                CHECK(packet.hot_slots()[job] == resident[packet.hot_experts()[job]]);
                CHECK(packet.hot_slots()[job] >= 0 && packet.hot_slots()[job] < (int32_t) frozen_slots);
                CHECK(packet.hot_experts()[job] == routes[packet.hot_rows()[job] * row_stride + column % n_used]);
                CHECK(packet.gpu_ids()[column] == packet.hot_slots()[job]);
            }
            for (size_t job = 0; job < packet.cold_columns().size(); ++job) {
                const auto column = packet.cold_columns()[job];
                CHECK(column >= 0 && (size_t) column < owner.size() && owner[column]++ == 0);
            }
            // Every routed column belongs to exactly one side: no duplicate and no dropped contribution.
            CHECK(std::all_of(owner.begin(), owner.end(), [](int count) { return count == 1; }));
            cold_columns += cold_index;
            mixed_partitions += cold_index != 0 && cold_index != packet.gpu_ids().size();
            // Frozen means the partition itself fills nothing: no bank upload, no slot churn.
            const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
            CHECK(telemetry.h2d_banks == 0 && telemetry.h2d_bytes == 0 && telemetry.demand_materialized_bytes == 0);
            CHECK(telemetry.vacant_fills == 0 && telemetry.replacement_fills == 0 && telemetry.invalidation_refills == 0);
            CHECK(telemetry.host_staged_ops == 0 && telemetry.fallback == 0);
            // Device-side selection must reproduce the host packet exactly, in route order.
            {
                const size_t n_routes = packet.gpu_ids().size();
                scratch.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0),
                    (5 * n_routes + 2) * sizeof(int32_t) + ids.size()));
                CHECK(scratch != nullptr);
                auto * base = static_cast<int32_t *>(ggml_backend_buffer_get_base(scratch.get()));
                device_hot_slots = base;
                device_hot_rows = device_hot_slots + n_routes;
                device_hot_columns = device_hot_rows + n_routes;
                int32_t * device_cold_ids = device_hot_columns + n_routes;
                int32_t * device_cold_columns = device_cold_ids + n_routes;
                device_counts = device_cold_columns + n_routes;
                int32_t * device_ids = device_counts + 2;
                CUDA_OK(cudaMemcpyAsync(device_ids, ids.data(), ids.size(), cudaMemcpyHostToDevice, stream));
                const auto select = [&](int32_t top_k) {
                    return ggml_cuda_moe_hybrid_select(device_ids, row_stride, top_k, n_rows, device_slots,
                        n_experts, frozen_slots, device_hot_slots, device_hot_rows, device_hot_columns,
                        device_cold_ids, device_cold_columns, device_counts, stream);
                };
                CHECK(!ggml_cuda_moe_hybrid_select(device_ids, row_stride, n_used, n_rows, device_slots, n_experts,
                    frozen_slots, device_hot_slots, device_hot_rows, device_hot_columns, device_cold_ids,
                    device_cold_columns, device_counts, nullptr));
                CHECK(!select(0) && !select(n_experts + 1));
                CHECK(select(n_used));
                std::vector<int32_t> counts(2), got_hot_slots(n_routes), got_hot_rows(n_routes), got_hot_columns(n_routes);
                std::vector<int32_t> got_cold_ids(n_routes), got_cold_columns(n_routes);
                const auto fetch = [&](std::vector<int32_t> & out, const int32_t * src) {
                    CUDA_OK(cudaMemcpyAsync(out.data(), src, out.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                };
                fetch(counts, device_counts);
                fetch(got_hot_slots, device_hot_slots);
                fetch(got_hot_rows, device_hot_rows);
                fetch(got_hot_columns, device_hot_columns);
                fetch(got_cold_ids, device_cold_ids);
                fetch(got_cold_columns, device_cold_columns);
                CUDA_OK(cudaStreamSynchronize(stream));
                CHECK(counts[0] == (int32_t) packet.hot_slots().size() && counts[1] == (int32_t) packet.cold_ids().size());
                got_hot_slots.resize(counts[0]);
                got_hot_rows.resize(counts[0]);
                got_hot_columns.resize(counts[0]);
                got_cold_ids.resize(counts[1]);
                got_cold_columns.resize(counts[1]);
                CHECK(got_hot_slots == packet.hot_slots() && got_hot_columns == packet.hot_columns());
                CHECK(got_hot_rows == packet.hot_rows());
                CHECK(got_cold_ids == packet.cold_ids() && got_cold_columns == packet.cold_columns());
                // A route outside the expert range is refused instead of selected.
                const int32_t invalid = n_experts;
                CUDA_OK(cudaMemcpyAsync(device_ids, &invalid, sizeof(invalid), cudaMemcpyHostToDevice, stream));
                CHECK(select(n_used));
                fetch(counts, device_counts);
                CUDA_OK(cudaStreamSynchronize(stream));
                CHECK(counts[0] == -1 && counts[1] == -1);
                // A refused selection leaves its lists unusable, so the valid one must be produced again.
                CUDA_OK(cudaMemcpyAsync(device_ids, ids.data(), ids.size(), cudaMemcpyHostToDevice, stream));
                CHECK(select(n_used));
                fetch(counts, device_counts);
                CUDA_OK(cudaStreamSynchronize(stream));
                CHECK(counts[0] == (int32_t) packet.hot_slots().size() && counts[1] == (int32_t) packet.cold_ids().size());
            }

            // Strata's hit path in ggml terms: slot-indexed banks, compact routes, results placed by dst.
            if (!packet.hot_slots().empty()) {
                ggml_cuda_moe_grouped_bank_descriptor gate_bank, up_bank, down_bank;
                const auto bank_of = [&](const ggml_tensor * tensor, ggml_cuda_moe_grouped_bank_descriptor * out) {
                    for (uint32_t slot_index = 0; slot_index < GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS; ++slot_index) {
                        if (context->get_group_resource_bank(held, slot_index, out) && out->tensor == tensor) {
                            return true;
                        }
                    }
                    return false;
                };
                CHECK(bank_of(cached.gate[0], &gate_bank) && bank_of(cached.up[0], &up_bank) &&
                    bank_of(cached.down[0], &down_bank));
                const size_t n_hits = packet.hot_slots().size();
                ggml_context_ptr hot_ctx(ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false),
                    nullptr, true}));
                CHECK(hot_ctx != nullptr);
                const auto bank_tensor = [&](const ggml_cuda_moe_grouped_bank_descriptor & bank) {
                    auto * tensor = ggml_new_tensor_3d(hot_ctx.get(), bank.tensor->type, n_dim, n_dim, frozen_slots);
                    CHECK(bank.expert_stride == ggml_nbytes(tensor) / frozen_slots);
                    return tensor;
                };
                auto * gate_slots = bank_tensor(gate_bank);
                auto * up_slots = bank_tensor(up_bank);
                auto * down_slots = bank_tensor(down_bank);
                // Gather the activation rows on the device from the selector's own row list.
                auto * hot_rows_ids = ggml_new_tensor_1d(hot_ctx.get(), GGML_TYPE_I32, n_hits);
                auto * hot_ids = ggml_new_tensor_2d(hot_ctx.get(), GGML_TYPE_I32, 1, n_hits);
                ggml_set_input(hot_rows_ids);
                ggml_set_input(hot_ids);
                auto * batch = ggml_new_tensor_2d(hot_ctx.get(), GGML_TYPE_F32, n_dim, n_rows);
                ggml_set_input(batch);
                auto * hot_input = ggml_reshape_3d(hot_ctx.get(),
                    ggml_get_rows(hot_ctx.get(), batch, hot_rows_ids), n_dim, 1, n_hits);
                auto * up_act = ggml_mul_mat_id(hot_ctx.get(), up_slots, hot_input, hot_ids);
                auto * gate_act = ggml_mul_mat_id(hot_ctx.get(), gate_slots, hot_input, hot_ids);
                auto * hot_out = ggml_mul_mat_id(hot_ctx.get(),
                    down_slots, ggml_swiglu_split(hot_ctx.get(), gate_act, up_act), hot_ids);
                ggml_set_output(hot_out);
                auto * hot_graph = ggml_new_graph_custom(hot_ctx.get(), 32, false);
                ggml_build_forward_expand(hot_graph, hot_out);
                ggml_gallocr_ptr hot_allocator(ggml_gallocr_new(ggml_backend_cuda_buffer_type(0)));
                CHECK(ggml_gallocr_alloc_graph(hot_allocator.get(), hot_graph));
                // The slot arena is copied so this test owns the storage; the layout and indexing are the bank's.
                for (const auto & pair : {std::make_pair(gate_slots, &gate_bank), std::make_pair(up_slots, &up_bank),
                                          std::make_pair(down_slots, &down_bank)}) {
                    CUDA_OK(cudaMemcpyAsync(pair.first->data, pair.second->source_data, ggml_nbytes(pair.first),
                        cudaMemcpyDeviceToDevice, stream));
                }
                // Ids and rows stay on the device: the selector's output is consumed where it was produced.
                CUDA_OK(cudaMemcpyAsync(hot_ids->data, device_hot_slots, n_hits * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream));
                CUDA_OK(cudaMemcpyAsync(hot_rows_ids->data, device_hot_rows, n_hits * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream));
                // The batch is staged once; the per-route gather stays on the device via the selector's rows.
                ggml_backend_tensor_set(batch, input.data(), 0, input.size() * sizeof(float));
                CUDA_OK(cudaStreamSynchronize(stream));
                CHECK(ggml_backend_graph_compute(cuda.get(), hot_graph) == GGML_STATUS_SUCCESS);
                {
                    std::vector<float> activations(n_hits * n_dim);
                    ggml_backend_tensor_get(hot_input, activations.data(), 0, activations.size() * sizeof(float));
                    const std::vector<float> original = activations;
                    for (int sample = 0; sample < 3; ++sample) {
                        if (sample == 1) {
                            std::fill(activations.begin(), activations.end(), 0.0f);
                        } else if (sample == 2) {
                            for (size_t i = 0; i < activations.size(); ++i) {
                                const int lane = i % QK8_0;
                                activations[i] = lane == 0 ? 127.0f : (lane % 2 ? -1.0f : 1.0f) * (lane + 0.5f);
                            }
                        }
                        ggml_backend_tensor_set(hot_input, activations.data(), 0, activations.size() * sizeof(float));
                        const size_t blocks = activations.size() / QK8_0;
                        std::vector<block_q8_0> cpu_blocks(blocks);
                        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(
                            activations.data(), cpu_blocks.data(), activations.size());
                        ggml_backend_buffer_ptr quant_buffer(ggml_backend_buft_alloc_buffer(
                            ggml_backend_cuda_buffer_type(0), blocks * sizeof(block_q8_1)));
                        CHECK(quant_buffer != nullptr);
                        CHECK(ggml_cuda_moe_quantize_q8_1_for_test(static_cast<const float *>(hot_input->data),
                            ggml_backend_buffer_get_base(quant_buffer.get()), n_dim, n_hits, stream));
                        CUDA_OK(cudaStreamSynchronize(stream));
                        std::vector<block_q8_1> gpu_blocks(blocks);
                        CUDA_OK(cudaMemcpy(gpu_blocks.data(), ggml_backend_buffer_get_base(quant_buffer.get()),
                            blocks * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
                        size_t different_scales = 0, different_codes = 0;
                        double max_dequant_difference = 0.0;
                        for (size_t b = 0; b < blocks; ++b) {
                            different_scales += cpu_blocks[b].d != gpu_blocks[b].d;
                            const float cpu_scale = ggml_fp16_to_fp32(cpu_blocks[b].d);
                            const float gpu_scale = ggml_fp16_to_fp32(gpu_blocks[b].d);
                            for (int q = 0; q < QK8_0; ++q) {
                                different_codes += cpu_blocks[b].qs[q] != gpu_blocks[b].qs[q];
                                max_dequant_difference = std::max(max_dequant_difference,
                                    std::abs((double) cpu_scale * cpu_blocks[b].qs[q] -
                                             (double) gpu_scale * gpu_blocks[b].qs[q]));
                            }
                        }
                        fprintf(stderr, "test-moe-cache: activation quantization sample=%d blocks=%zu scale_differences=%zu code_differences=%zu max_dequant_difference=%g\n",
                            sample, blocks, different_scales, different_codes, max_dequant_difference);
                        if (sample == 1) {
                            CHECK(different_scales == 0 && different_codes == 0 && max_dequant_difference == 0.0);
                        }
                    }
                    ggml_backend_tensor_set(hot_input, original.data(), 0, original.size() * sizeof(float));
                }
                std::vector<float> hot_values(n_hits * n_dim);
                ggml_backend_tensor_get(hot_out, hot_values.data(), 0, hot_values.size() * sizeof(float));

                // The same half through the backend entry point, reading the arena in place instead of a copy.
                {
                    const size_t scratch_bytes = ggml_backend_cuda_moe_hybrid_hot_scratch_bytes(n_hits, n_dim);
                    CHECK(scratch_bytes != 0);
                    ggml_backend_buffer_ptr hot_buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0),
                        scratch_bytes + n_hits * n_dim * sizeof(float)));
                    CHECK(hot_buffer != nullptr);
                    auto * hot_base = static_cast<char *>(ggml_backend_buffer_get_base(hot_buffer.get()));
                    auto * in_place = reinterpret_cast<float *>(hot_base + scratch_bytes);
                    const auto execute = [&](size_t bytes) {
                        return ggml_backend_cuda_moe_hybrid_execute_hot(cuda.get(), held, cached.gate[0], cached.up[0],
                            cached.down[0], static_cast<const float *>(hot_input->data), device_hot_slots, n_hits,
                            false, hot_base, bytes, in_place);
                    };
                    CHECK(!execute(scratch_bytes - 1));
                    CHECK(execute(scratch_bytes));
                    CUDA_OK(cudaStreamSynchronize(stream));
                    std::vector<float> in_place_values(hot_values.size());
                    CUDA_OK(cudaMemcpy(in_place_values.data(), in_place, in_place_values.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
                    double place_error = 0.0, place_norm = 0.0;
                    for (size_t i = 0; i < hot_values.size(); ++i) {
                        CHECK(std::isfinite(in_place_values[i]));
                        const double difference = (double) in_place_values[i] - hot_values[i];
                        place_error += difference * difference;
                        place_norm += (double) hot_values[i] * hot_values[i];
                    }
                    const double place_nmse = place_norm > 0.0 ? place_error / place_norm : place_error;
                    if (place_nmse != 0.0) {
                        fprintf(stderr, "frozen in-place hot rows=%d hits=%zu NMSE=%g\n", n_rows, n_hits, place_nmse);
                    }
                    CHECK(place_nmse == 0.0);
                }
                double error = 0.0, norm = 0.0;
                for (size_t job = 0; job < n_hits; ++job) {
                    const auto column = packet.hot_columns()[job];
                    for (int i = 0; i < n_dim; ++i) {
                        const double reference = oracle[column * n_dim + i];
                        const double difference = (double) hot_values[job * n_dim + i] - reference;
                        error += difference * difference;
                        norm += reference * reference;
                        // Measure full-FFN disagreement against signal RMS, not near-zero elements.
                        half_difference = std::max(half_difference, std::abs(difference));
                        half_square += reference * reference;
                        ++half_count;
                    }
                }
                const double nmse = norm > 0.0 ? error / norm : error;
                if (nmse > 5e-4) {
                    fprintf(stderr, "frozen hot half rows=%d iteration=%d hits=%zu NMSE=%g\n",
                        n_rows, iteration, n_hits, nmse);
                }
                CHECK(nmse <= 5e-4);
                hot_halves += n_hits;

                // Strata's merge: the CPU half owns its rows and leaves the hot rows at zero, then the GPU adds.
                std::vector<float> parts(packet.gpu_ids().size() * n_dim, 0.0f);
                for (size_t job = 0; job < packet.cold_columns().size(); ++job) {
                    memcpy(parts.data() + packet.cold_columns()[job] * n_dim, cold.data() + job * n_dim,
                        n_dim * sizeof(float));
                }
                ggml_backend_tensor_set(routed.hybrid_ffn.down_output, parts.data(), 0, parts.size() * sizeof(float));
                CHECK(ggml_cuda_moe_hybrid_scatter_add(
                    static_cast<float *>(routed.hybrid_ffn.down_output->data), static_cast<const float *>(hot_out->data),
                    device_hot_columns, device_counts, packet.gpu_ids().size(), n_dim, stream));
                CUDA_OK(cudaStreamSynchronize(stream));
                CHECK(!ggml_cuda_moe_hybrid_scatter_add(nullptr, static_cast<const float *>(hot_out->data),
                    device_hot_columns, device_counts, packet.gpu_ids().size(), n_dim, stream));
                // Every column must now be covered: the scatter filled exactly the rows the CPU left at zero.
                std::vector<float> merged(parts.size());
                ggml_backend_tensor_get(routed.hybrid_ffn.down_output, merged.data(), 0, merged.size() * sizeof(float));
                std::vector<unsigned char> owners(packet.gpu_ids().size(), 0);
                for (size_t job = 0; job < n_hits; ++job) {
                    const auto column = packet.hot_columns()[job];
                    CHECK(owners[column]++ == 0);
                    CHECK(memcmp(merged.data() + column * n_dim, hot_values.data() + job * n_dim,
                        n_dim * sizeof(float)) == 0);
                }
                for (size_t job = 0; job < packet.cold_columns().size(); ++job) {
                    const auto column = packet.cold_columns()[job];
                    CHECK(owners[column]++ == 0);
                    CHECK(memcmp(merged.data() + column * n_dim, cold.data() + job * n_dim,
                        n_dim * sizeof(float)) == 0);
                }
                CHECK(std::all_of(owners.begin(), owners.end(), [](unsigned char value) { return value == 1; }));

                const auto scatter = [&]() {
                    CHECK(ggml_cuda_moe_hybrid_scatter_add(
                        static_cast<float *>(routed.hybrid_ffn.down_output->data),
                        static_cast<const float *>(hot_out->data), device_hot_columns, device_counts,
                        packet.gpu_ids().size(), n_dim, stream));
                    CUDA_OK(cudaStreamSynchronize(stream));
                };
                // Zero and refused counts must leave every destination unchanged, including poison.
                std::vector<float> poison(parts.size(), std::numeric_limits<float>::quiet_NaN());
                for (int32_t count : {0, -1, (int32_t) packet.gpu_ids().size() + 1}) {
                    ggml_backend_tensor_set(routed.hybrid_ffn.down_output, poison.data(), 0, poison.size() * sizeof(float));
                    CUDA_OK(cudaMemcpy(device_counts, &count, sizeof(count), cudaMemcpyHostToDevice));
                    scatter();
                    ggml_backend_tensor_get(routed.hybrid_ffn.down_output, merged.data(), 0, merged.size() * sizeof(float));
                    CHECK(memcmp(merged.data(), poison.data(), poison.size() * sizeof(float)) == 0);
                }
                const int32_t count = (int32_t) n_hits;
                CUDA_OK(cudaMemcpy(device_counts, &count, sizeof(count), cudaMemcpyHostToDevice));
                // A legitimate zero hot output must still cover its columns and preserve the cold half.
                std::vector<float> zeros(hot_values.size(), 0.0f);
                ggml_backend_tensor_set(hot_out, zeros.data(), 0, zeros.size() * sizeof(float));
                ggml_backend_tensor_set(routed.hybrid_ffn.down_output, parts.data(), 0, parts.size() * sizeof(float));
                scatter();
                ggml_backend_tensor_get(routed.hybrid_ffn.down_output, merged.data(), 0, merged.size() * sizeof(float));
                CHECK(memcmp(merged.data(), parts.data(), parts.size() * sizeof(float)) == 0);
                // Reinitialize before each merge so reused storage cannot accumulate the previous layer.
                ggml_backend_tensor_set(hot_out, hot_values.data(), 0, hot_values.size() * sizeof(float));
                std::vector<float> expected = parts;
                for (size_t job = 0; job < n_hits; ++job) {
                    memcpy(expected.data() + packet.hot_columns()[job] * n_dim, hot_values.data() + job * n_dim,
                        n_dim * sizeof(float));
                }
                for (int repeat = 0; repeat < 2; ++repeat) {
                    ggml_backend_tensor_set(routed.hybrid_ffn.down_output, parts.data(), 0, parts.size() * sizeof(float));
                    scatter();
                    ggml_backend_tensor_get(routed.hybrid_ffn.down_output, merged.data(), 0, merged.size() * sizeof(float));
                    CHECK(memcmp(merged.data(), expected.data(), expected.size() * sizeof(float)) == 0);
                }
                CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate, span.expert_end,
                    span.region_end) == GGML_STATUS_SUCCESS);
                ggml_backend_sched_synchronize(sched.get());
                std::vector<float> combined(n_rows * n_dim), reference(combined.size());
                ggml_backend_tensor_get(cached.output[0], combined.data(), 0, combined.size() * sizeof(float));
                ggml_backend_tensor_get(host.output[0], reference.data(), 0, reference.size() * sizeof(float));
                double combine_error = 0.0, combine_norm = 0.0;
                for (size_t i = 0; i < combined.size(); ++i) {
                    CHECK(std::isfinite(combined[i]));
                    const double difference = (double) combined[i] - reference[i];
                    combine_error += difference * difference;
                    combine_norm += (double) reference[i] * reference[i];
                }
                const double combine_nmse = combine_norm > 0.0 ? combine_error / combine_norm : combine_error;
                if (combine_nmse > 5e-4) {
                    fprintf(stderr, "frozen hybrid layer rows=%d iteration=%d NMSE=%g\n", n_rows, iteration, combine_nmse);
                }
                CHECK(combine_nmse <= 5e-4);
                ++hybrid_layers;
            }
            CHECK(snapshot() == resident);
            ggml_cuda_moe_grouped_decode_acquisition finish;
            finish.transaction = held;
            CHECK(context->finish_decode(finish, stream));
            CHECK(!context->snapshot_resident_slots(held, stream, device_slots, resident.size()));
        }
        total_cold += cold_columns;
        total_resident_all += total_resident;
        total_mixed += mixed_partitions;
        CUDA_OK(cudaFree(device_slots));
        total_hot += hot_halves;
        total_layers += hybrid_layers;
        fprintf(stderr, "test-moe-cache: frozen partition rows=%d resident=%u hot=%zu cold=%zu mixed=%d layers=%zu OK\n",
            n_rows, total_resident, hot_halves, cold_columns, (int) mixed_partitions, hybrid_layers);
        ggml_backend_sched_synchronize(sched.get());
    }
    // A mixed partition is the point of the split; all-hot or all-cold alone would not test it.
    CHECK(total_cold != 0 && total_resident_all != 0 && total_mixed != 0 && total_hot != 0 && total_layers != 0);
    // A hit and a miss must eventually produce the same number; this records how far apart they are today.
    const double half_rms = half_count != 0 ? std::sqrt(half_square / half_count) : 0.0;
    fprintf(stderr, "test-moe-cache: hybrid half disagreement max_abs=%g rms=%g ratio=%g (GPU resident vs CPU cold, q4_0)\n",
        half_difference, half_rms, half_rms > 0.0 ? half_difference / half_rms : 0.0);
    cpu.reset();
    ggml_threadpool_free(pool);
}

void test_sched_compute_range(bool cpu_output = false, bool semantic_cuts = false, int n_rows = 1) {
    layer_fixture fixture(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, false, {0, 0}, 0);
    fixture.build_graph(LLM_FFN_SILU, 0.0f, false, false, n_rows);
    std::vector<ggml_backend_ptr> owners;
    std::vector<ggml_backend_t> backends;
    owners.emplace_back(ggml_backend_cuda_init(0));
    CHECK(owners.back() != nullptr);
    backends.push_back(owners.back().get());
    owners.emplace_back(ggml_backend_cpu_init());
    CHECK(owners.back() != nullptr);
    backends.push_back(owners.back().get());
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 512, false, true));
    CHECK(sched != nullptr);
    auto * graph = fixture.result.get_gf();
    ggml_set_output(fixture.output[1]);
    // Production places a region on one owner; without that the routes can reach CUDA as a contiguous copy.
    for (auto & region : fixture.result.get_moe_regions()) {
        CHECK(region.place(sched.get(), backends[0]));
    }
    if (cpu_output) {
        ggml_backend_sched_set_tensor_backend(sched.get(), fixture.output[1], backends[1]);
    }
    std::vector<int> cuts;
    if (semantic_cuts) {
        for (const auto & region : fixture.result.get_moe_regions()) {
            const auto span = llm_graph_moe_hybrid_find_span(graph, region);
            CHECK(span.valid());
            for (int boundary : {span.routing_end, span.expert_end, span.region_end}) {
                if (boundary > 0 && boundary < graph->n_nodes) {
                    cuts.push_back(boundary);
                }
            }
        }
        for (int i = 1; i < graph->n_nodes; ++i) {
            if (ggml_graph_node(graph, i)->op == GGML_OP_VIEW) {
                cuts.push_back(i);
                break;
            }
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        CHECK(!cuts.empty());
        auto requested = cuts;
        CHECK(ggml_backend_sched_set_split_boundaries(sched.get(), graph, requested.data(), requested.size()));
        std::fill(requested.begin(), requested.end(), -1);
        const int invalid[] = {0, graph->n_nodes};
        CHECK(!ggml_backend_sched_set_split_boundaries(sched.get(), graph, invalid, 2));
        const int duplicate[] = {cuts.front(), cuts.front()};
        CHECK(!ggml_backend_sched_set_split_boundaries(sched.get(), graph, duplicate, 2));
        CHECK(!ggml_backend_sched_set_split_boundaries(sched.get(), graph, nullptr, 1));
        CHECK(!ggml_backend_sched_set_split_boundaries(sched.get(), graph, nullptr, -1));
        const int descending[] = {cuts.back(), cuts.front()};
        CHECK(!ggml_backend_sched_set_split_boundaries(sched.get(), graph, descending, 2));
        ggml_cgraph foreign = *graph;
        CHECK(!ggml_backend_sched_alloc_graph(sched.get(), &foreign));
        const auto uid = graph->uid;
        ++graph->uid;
        CHECK(!ggml_backend_sched_alloc_graph(sched.get(), graph));
        graph->uid = uid;
        --graph->n_nodes;
        CHECK(!ggml_backend_sched_alloc_graph(sched.get(), graph));
        ++graph->n_nodes;
    }
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
    CHECK(!ggml_backend_sched_set_split_boundaries(sched.get(), graph, nullptr, 0));
    const auto run_whole = [&]() {
        fixture.set_inputs(2);
        CHECK(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
        std::vector<float> values(ggml_nelements(fixture.output[1]));
        ggml_backend_tensor_get(fixture.output[1], values.data(), 0, values.size() * sizeof(float));
        return values;
    };
    const auto expected = run_whole();
    {
        // Use a fresh graph because scheduler splitting rewrites source pointers.
        layer_fixture reference_fixture(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, false, {0, 0}, 0);
        reference_fixture.build_graph(LLM_FFN_SILU, 0.0f, false, false, n_rows);
        auto * reference_graph = reference_fixture.result.get_gf();
        ggml_set_output(reference_fixture.output[1]);
        ggml_gallocr_ptr allocator(ggml_gallocr_new(ggml_backend_cuda_buffer_type(0)));
        CHECK(ggml_gallocr_alloc_graph(allocator.get(), reference_graph));
        reference_fixture.set_inputs(2);
        CHECK(ggml_backend_graph_compute(backends[0], reference_graph) == GGML_STATUS_SUCCESS);
        std::vector<float> reference(expected.size());
        ggml_backend_tensor_get(reference_fixture.output[1], reference.data(), 0, reference.size() * sizeof(float));
        double error = 0.0, norm = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double difference = (double) expected[i] - reference[i];
            error += difference * difference;
            norm += (double) reference[i] * reference[i];
        }
        const double nmse = norm > 0.0 ? error / norm : error;
        if (nmse > 5e-4) {
            fprintf(stderr, "scheduler vs direct CUDA mismatch NMSE=%g\n", nmse);
        }
        CHECK(nmse <= 5e-4);
    }

    // Keep the allocation and discover its split boundaries.
    fixture.set_inputs(2);
    std::vector<std::pair<int, int>> ranges;
    const int n_splits = ggml_backend_sched_get_n_splits(sched.get());
    CHECK(n_splits >= (cpu_output ? 2 : 1));
    for (int i = 0; i < n_splits; ++i) {
        int begin = 0, end = 0;
        CHECK(ggml_backend_sched_get_split_range(sched.get(), i, &begin, &end));
        ranges.emplace_back(begin, end);
    }
    for (int cut : cuts) {
        CHECK(std::any_of(ranges.begin(), ranges.end(), [cut](const std::pair<int, int> & range) {
            return range.second == cut;
        }));
    }
    for (const auto & range : ranges) {
        CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, range.first, range.second) ==
            GGML_STATUS_SUCCESS);
    }
    ggml_backend_sched_synchronize(sched.get());
    std::vector<float> ranged(expected.size());
    ggml_backend_tensor_get(fixture.output[1], ranged.data(), 0, ranged.size() * sizeof(float));
    CHECK(ranged == expected);

    // A range that does not land on split boundaries is refused, and refusal changes nothing.
    CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, ranges.front().first,
        ranges.front().second - 1) == GGML_STATUS_FAILED);
    CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, ranges.front().first + 1,
        ranges.front().second) == GGML_STATUS_FAILED);
    CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, -1, 4) == GGML_STATUS_FAILED);
    CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, 4, 4) == GGML_STATUS_FAILED);
    CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, 0, graph->n_nodes + 1) == GGML_STATUS_FAILED);
    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_tensor_get(fixture.output[1], ranged.data(), 0, ranged.size() * sizeof(float));
    CHECK(ranged == expected);

    if (semantic_cuts) {
        split_probe probe{ &fixture, backends, {} };
        probe.expected_source_uid = graph->uid;
        for (auto * backend : backends) {
            probe.delegates.push_back(backend->iface.graph_compute);
            backend->iface.graph_compute = probe_compute;
        }
        active_probe = &probe;
        for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_MAIN, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT,
                                GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
            for (uint32_t semantics : {GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT,
                                       GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL}) {
                auto certificate = layer_certificate(true);
                certificate.domain = domain;
                certificate.n_rows = n_rows;
                certificate.n_sequences = semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ? n_rows : 1;
                certificate.row_semantics = semantics;
                if (domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN) {
                    if (semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT || n_rows == 1) {
                        continue;
                    }
                    certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
                }
                probe.expected_certificate = &certificate;
                probe.certificate_calls = 0;
                fixture.set_inputs(2);
                for (const auto & range : ranges) {
                    CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate,
                        range.first, range.second) == GGML_STATUS_SUCCESS);
                }
                ggml_backend_sched_synchronize(sched.get());
                CHECK(probe.certificate_calls == n_splits);
                ggml_backend_tensor_get(fixture.output[1], ranged.data(), 0, ranged.size() * sizeof(float));
                CHECK(ranged == expected);
                auto invalid = certificate;
                invalid.owner_generation = 0;
                CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &invalid,
                    ranges.front().first, ranges.front().second) == GGML_STATUS_FAILED);
                CHECK(probe.certificate_calls == n_splits);
                probe.expected_certificate = nullptr;
            }
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            backends[i]->iface.graph_compute = probe.delegates[i];
        }
        active_probe = nullptr;
    }

    // Whole-graph execution still works after ranged execution.
    CHECK(run_whole() == expected);
    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_sched_reset(sched.get());
    fixture.build_graph(LLM_FFN_SILU, 0.0f, false, false, n_rows);
    graph = fixture.result.get_gf();
    for (auto & region : fixture.result.get_moe_regions()) {
        CHECK(region.place(sched.get(), backends[0]));
    }
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
    CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 1);
    fprintf(stderr, "test-moe-cache: scheduler ranged execution matches whole-graph execution OK\n");
}

void test_sched_cached_boundaries(llm_ffn_op_type activation = LLM_FFN_SILU) {
    const bool debug = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    for (int n_rows : {1, 4}) {
        layer_fixture fixture(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0,
                              false, {0, 0}, all_cached_layers, n_experts);
        ggml_backend_ptr cuda(ggml_backend_cuda_init(0));
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        ggml_backend_ptr other_cuda(ggml_backend_cuda_init(0));
        CHECK(cuda && cpu && other_cuda);
        ggml_backend_t backends[] = {cuda.get(), cpu.get()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 512, false, true));
        CHECK(sched != nullptr);
        const auto manifest = fixture.manifest();
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(cuda.get(), &manifest) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(cuda.get());
        CHECK(context != nullptr);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(other_cuda.get(), &manifest) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * other_context = ggml_cuda_moe_grouped_context_for_test(other_cuda.get());
        CHECK(other_context != nullptr);
        uint64_t next_submission = 1;
        const auto allocate = [&](int cut) {
            ggml_backend_sched_synchronize(sched.get());
            ggml_backend_sched_reset(sched.get());
            fixture.build_graph(activation, 0.0f, false, false, n_rows);
            auto * graph = fixture.result.get_gf();
            for (auto & region : fixture.result.get_moe_regions()) {
                CHECK(region.place(sched.get(), cuda.get()));
            }
            if (cut != 0) {
                const auto span = llm_graph_moe_hybrid_find_span(graph, fixture.result.get_moe_regions()[0]);
                CHECK(span.valid());
                if (cut == 1) {
                    const int boundary = span.region_end;
                    CHECK(ggml_backend_sched_set_split_boundaries(sched.get(), graph, &boundary, 1));
                } else {
                    const int boundaries[] = {span.routing_end, span.expert_end};
                    CHECK(ggml_backend_sched_set_split_boundaries(sched.get(), graph, boundaries, 2));
                }
            }
            CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
            fixture.set_inputs(0);
            return graph;
        };
        for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_MAIN, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT,
                                GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
            for (uint32_t semantics : {GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT,
                                       GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL}) {
                auto certificate = layer_certificate(true);
                certificate.domain = domain;
                certificate.n_rows = n_rows;
                certificate.n_sequences = semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ? n_rows : 1;
                certificate.row_semantics = semantics;
                if (domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN) {
                    if (n_rows == 1 || semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT) {
                        continue;
                    }
                    certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
                }
                auto * graph = allocate(0);
                std::vector<std::vector<float>> expected_runs;
                for (int iteration = 0; iteration < 3; ++iteration) {
                    fixture.set_inputs(iteration);
                    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
                    expected_runs.push_back(active_grouped_tensor_values(fixture.output[1]));
                }
                const auto & expected = expected_runs[0];
                graph = allocate(1);
                CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 2);
                ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
                for (int repeat = 0; repeat < 3; ++repeat) {
                    fixture.set_inputs(repeat);
                    for (int split = 0; split < 2; ++split) {
                        int begin = 0, end = 0;
                        CHECK(ggml_backend_sched_get_split_range(sched.get(), split, &begin, &end));
                        CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate,
                            begin, end) == GGML_STATUS_SUCCESS);
                    }
                    ggml_backend_sched_synchronize(sched.get());
                    CHECK(active_grouped_tensor_values(fixture.output[1]) == expected_runs[repeat]);
                }
                const auto complete = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
                CHECK(complete.submitted >= 2 && complete.completed == complete.submitted);
                CHECK(complete.host_staged_ops == 0 && complete.required_unsupported == 0);
                graph = allocate(2);
                ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
                CHECK(ggml_backend_sched_get_n_splits(sched.get()) >= 3);
                int route_begin = 0, route_end = 0, expert_begin = 0, expert_end = 0;
                CHECK(ggml_backend_sched_get_split_range(sched.get(), 0, &route_begin, &route_end));
                CHECK(ggml_backend_sched_get_split_range(sched.get(), 1, &expert_begin, &expert_end));
                CHECK(route_begin == 0 && route_end == expert_begin);
                CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate,
                    route_begin, route_end) == GGML_STATUS_SUCCESS);
                CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate,
                    expert_begin, expert_end) == GGML_STATUS_FAILED);
                const auto refused = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
                CHECK(refused.required_unsupported > 0 && refused.h2d_banks == 0);
                CHECK(refused.host_staged_ops == 0 && refused.fallback == 0 && refused.submitted == 0);
                fixture.set_inputs(0);
                ggml_cuda_moe_split_route_lease lease;
                const ggml_cuda_moe_candidate_group_key candidate{context->state().generation, 0};
                uint64_t submission = next_submission++;
                CHECK(context->begin_split_routes(graph, certificate, submission));
                CHECK(!context->begin_split_routes(graph, certificate, next_submission));
                const auto stream = ggml_cuda_moe_owner_stream_for_test(cuda.get());
                CHECK(stream != nullptr);
                const auto other_stream = ggml_cuda_moe_owner_stream_for_test(other_cuda.get());
                CHECK(other_stream != nullptr && other_stream != stream);
                split_probe probe{ &fixture, {cuda.get()}, {cuda->iface.graph_compute} };
                ggml_cgraph consumer_witness = {};
                int publications = 0, validations = 0;
                probe.after_success = [&](ggml_backend_t, ggml_cgraph * producer) {
                    if (publications != 0) {
                        return;
                    }
                    ggml_cuda_moe_split_route_lease refused;
                    auto missing_route = *producer;
                    missing_route.n_nodes = 1;
                    CHECK(!context->publish_split_route(graph, &missing_route, candidate, fixture.ids[0], stream, &refused));
                    auto foreign_producer = *producer;
                    ++foreign_producer.execution_certificate.owner_namespace;
                    CHECK(!context->publish_split_route(graph, &foreign_producer, candidate, fixture.ids[0], stream, &refused));
                    CHECK(!context->publish_split_route(graph, producer, candidate, fixture.ids[1], stream, &refused));
                    CHECK(context->publish_split_route(graph, producer, candidate, fixture.ids[0], stream, &lease));
                    ggml_cuda_moe_split_route_lease duplicate;
                    CHECK(!context->publish_split_route(graph, producer, candidate, fixture.ids[0], stream, &duplicate));
                    ++publications;
                };
                probe.before_dispatch = [&](ggml_backend_t, ggml_cgraph * consumer) {
                    if (publications == 0) {
                        return;
                    }
                    bool has_down = false;
                    for (int i = 0; i < consumer->n_nodes; ++i) {
                        const auto * node = consumer->nodes[i];
                        has_down = has_down || (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == fixture.down[0]);
                    }
                    if (!has_down) {
                        return;
                    }
                    consumer_witness = *consumer;
                    const auto matches = [&](const ggml_cgraph * witness, const ggml_cuda_moe_candidate_group_key & key,
                                              uint64_t id, ggml_cuda_moe_stream_t owner_stream) {
                        return context->matches_split_route(lease, graph, witness, key, fixture.ids[0], id, owner_stream);
                    };
                    CHECK(matches(consumer, candidate, submission, stream));
                    CHECK(!matches(consumer, candidate, submission + 1, stream));
                    CHECK(!matches(consumer, candidate, submission, nullptr));
                    CHECK(!matches(consumer, candidate, submission, other_stream));
                    CHECK(other_context->begin_split_routes(graph, certificate, submission));
                    CHECK(!other_context->matches_split_route(lease, graph, consumer, candidate,
                        fixture.ids[0], submission, stream));
                    other_context->end_split_routes();
                    auto wrong_group = candidate;
                    ++wrong_group.group_index;
                    CHECK(!matches(consumer, wrong_group, submission, stream));
                    auto wrong_generation = candidate;
                    ++wrong_generation.generation;
                    CHECK(!matches(consumer, wrong_generation, submission, stream));
                    auto changed = *consumer;
                    ++changed.execution_certificate.owner_namespace;
                    CHECK(!matches(&changed, candidate, submission, stream));
                    changed = *consumer;
                    ++changed.execution_certificate.owner_generation;
                    CHECK(!matches(&changed, candidate, submission, stream));
                    changed = *consumer;
                    ++changed.execution_certificate.source_graph_uid;
                    CHECK(!matches(&changed, candidate, submission, stream));
                    changed = *consumer;
                    ++changed.execution_certificate.n_rows;
                    CHECK(!matches(&changed, candidate, submission, stream));
                    changed = *consumer;
                    ++changed.execution_certificate.n_sequences;
                    CHECK(!matches(&changed, candidate, submission, stream));
                    changed = *consumer;
                    changed.execution_certificate.domain = GGML_GRAPH_EXECUTION_DOMAIN_INVALID;
                    CHECK(!matches(&changed, candidate, submission, stream));
                    changed = *consumer;
                    changed.execution_certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID;
                    CHECK(!matches(&changed, candidate, submission, stream));
                    auto foreign_source = *graph;
                    CHECK(!context->matches_split_route(lease, &foreign_source, consumer, candidate,
                        fixture.ids[0], submission, stream));
                    ++graph->n_nodes;
                    CHECK(!matches(consumer, candidate, submission, stream));
                    --graph->n_nodes;
                    const auto source_uid = graph->uid;
                    ++graph->uid;
                    CHECK(!matches(consumer, candidate, submission, stream));
                    graph->uid = source_uid;
                    const auto route_root = fixture.ids[0]->src[0];
                    fixture.ids[0]->src[0] = fixture.input;
                    CHECK(!matches(consumer, candidate, submission, stream));
                    fixture.ids[0]->src[0] = route_root;
                    const auto stride = fixture.ids[0]->nb[1];
                    fixture.ids[0]->nb[1] += sizeof(int32_t);
                    CHECK(!matches(consumer, candidate, submission, stream));
                    fixture.ids[0]->nb[1] = stride;
                    const auto data = fixture.ids[0]->data;
                    fixture.ids[0]->data = nullptr;
                    CHECK(!matches(consumer, candidate, submission, stream));
                    fixture.ids[0]->data = data;
                    CHECK(matches(consumer, candidate, submission, stream));
                    ++validations;
                };
                active_probe = &probe;
                cuda->iface.graph_compute = probe_compute;
                for (int repeat = 0; repeat < 3; ++repeat) {
                    if (repeat != 0) {
                        context->end_split_routes();
                        CHECK(!context->matches_split_route(lease, graph, &consumer_witness, candidate,
                            fixture.ids[0], submission, stream));
                        lease.reset();
                        submission = next_submission++;
                        CHECK(context->begin_split_routes(graph, certificate, submission));
                        publications = validations = 0;
                        fixture.set_inputs(repeat);
                    }
                    for (int split = 0; split < ggml_backend_sched_get_n_splits(sched.get()); ++split) {
                        int begin = 0, end = 0;
                        CHECK(ggml_backend_sched_get_split_range(sched.get(), split, &begin, &end));
                        CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate,
                            begin, end) == GGML_STATUS_SUCCESS);
                    }
                    ggml_backend_sched_synchronize(sched.get());
                    CHECK(active_grouped_tensor_values(fixture.output[1]) == expected_runs[repeat]);
                    CHECK(publications == 1 && validations == 1);
                }
                fprintf(stderr, "test-moe-cache: certified cached semantic cuts activation=%d domain=%u semantics=%u rows=%d exact OK\n",
                    activation, domain, certificate.row_semantics, n_rows);
                const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
                CHECK(telemetry.required_unsupported == 0 && telemetry.h2d_banks == 0);
                CHECK(telemetry.host_staged_ops == 0 && telemetry.fallback == 0);
                CHECK(telemetry.submitted >= 2 && telemetry.completed == telemetry.submitted);
                cuda->iface.graph_compute = probe.delegates[0];
                active_probe = nullptr;
                if (n_rows == 4 && domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN) {
                    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(cuda.get(), &manifest) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
                    CHECK(context->state().generation > candidate.generation);
                } else {
                    context->end_split_routes();
                }
                CHECK(!context->matches_split_route(lease, graph, &consumer_witness, candidate,
                    fixture.ids[0], submission, stream));
                CHECK(!context->begin_split_routes(graph, certificate, submission));
                auto foreign_owner = certificate;
                ++foreign_owner.owner_namespace;
                CHECK(!context->begin_split_routes(graph, foreign_owner, next_submission));
                const auto fresh_submission = next_submission++;
                CHECK(context->begin_split_routes(graph, certificate, fresh_submission));
                CHECK(!context->matches_split_route(lease, graph, &consumer_witness, candidate,
                    fixture.ids[0], fresh_submission, stream));
                CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, &certificate,
                    expert_begin, expert_end) == GGML_STATUS_FAILED);
                context->end_split_routes();
                lease.reset();
                graph = allocate(0);
                CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
                CHECK(active_grouped_tensor_values(fixture.output[1]) == expected);
            }
        }
        auto advanced = layer_certificate(true);
        advanced.owner_generation = 2;
        CHECK(context->begin_split_routes(fixture.result.get_gf(), advanced, 1));
        context->end_split_routes();
        advanced.owner_generation = 1;
        CHECK(!context->begin_split_routes(fixture.result.get_gf(), advanced, next_submission));
        ggml_backend_sched_synchronize(sched.get());
    }
    ggml_backend_cuda_moe_set_debug_mm(debug);
}

void test_sched_boundaries_auto_alloc() {
    for (bool ranged : {false, true}) {
        ggml_context_ptr ctx(ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
        CHECK(ctx != nullptr);
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        CHECK(cpu != nullptr);
        auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_backend_buffer_ptr input_buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 64));
        CHECK(input_buffer != nullptr);
        CHECK(ggml_backend_tensor_alloc(input_buffer.get(), input, ggml_backend_buffer_get_base(input_buffer.get())) == GGML_STATUS_SUCCESS);
        const float value = 1.0f;
        ggml_backend_tensor_set(input, &value, 0, sizeof(value));
        auto * first = ggml_scale(ctx.get(), input, 2.0f);
        auto * output = ggml_scale(ctx.get(), first, 3.0f);
        ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_t backends[] = {cpu.get()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 8, false, false));
        CHECK(sched != nullptr);
        ggml_backend_sched_set_tensor_backend(sched.get(), first, cpu.get());
        const int cut = 1;
        CHECK(ggml_backend_sched_set_split_boundaries(sched.get(), graph, &cut, 1));
        if (ranged) {
            CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, 0, 1) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_sched_graph_compute_range(sched.get(), graph, nullptr, 1, 2) == GGML_STATUS_SUCCESS);
        } else {
            CHECK(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
        }
        CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 2);
        ggml_backend_sched_synchronize(sched.get());
        float result = 0.0f;
        ggml_backend_tensor_get(output, &result, 0, sizeof(result));
        CHECK(result == 6.0f);
    }
}

void test_hybrid_span() {
    layer_fixture fixture(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, false, {0, 0}, 0);
    for (int n_rows : {1, 4}) {
        fixture.build_graph(LLM_FFN_SILU, 0.0f, false, false, n_rows);
        auto * graph = fixture.result.get_gf();
        const auto & regions = fixture.result.get_moe_regions();
        CHECK(regions.size() == 2);
        int previous_end = 0;
        for (const auto & region : regions) {
            const auto span = llm_graph_moe_hybrid_find_span(graph, region);
            CHECK(span.valid());
            const auto & ffn = region.hybrid_ffn;
            CHECK(ggml_graph_node(graph, span.expert_end - 1) == ffn.down_output);
            CHECK(ggml_graph_node(graph, span.region_end - 1) == region.output);
            CHECK(span.routing_end >= previous_end);
            previous_end = span.region_end;
            // The first expert matmul starts the expert segment, and the routes precede it.
            auto * first = ggml_graph_node(graph, span.routing_end);
            CHECK(first->op == GGML_OP_MUL_MAT_ID &&
                (first->src[0] == ffn.up || first->src[0] == ffn.gate || first->src[0] == ffn.gate_up));
            bool route_before = false;
            for (int i = 0; i < span.routing_end; ++i) {
                route_before = route_before || ggml_graph_node(graph, i) == region.route;
            }
            CHECK(route_before);
        }
        // A region the descriptor rejects has no span to split.
        const auto & region = regions.front();
        auto unsupported = region;
        unsupported.hybrid_ffn = {};
        CHECK(!llm_graph_moe_hybrid_find_span(graph, unsupported).valid());
        auto no_output = region;
        no_output.output = nullptr;
        CHECK(!llm_graph_moe_hybrid_find_span(graph, no_output).valid());
        auto foreign = region;
        foreign.operations.clear();  // expert matmuls outside the region cannot be split here
        CHECK(!llm_graph_moe_hybrid_find_span(graph, foreign).valid());
        CHECK(!llm_graph_moe_hybrid_find_span(nullptr, region).valid());
    }
    // Clamped, scaled and externally routed regions are excluded before any span exists.
    for (int variant : {0, 1, 2}) {
        fixture.build_graph(variant == 0 ? LLM_FFN_RELU : LLM_FFN_SILU, variant == 1 ? 7.0f : 0.0f, false, variant == 2, 1);
        for (const auto & region : fixture.result.get_moe_regions()) {
            CHECK(region.hybrid_ffn.input == nullptr);
            CHECK(!llm_graph_moe_hybrid_find_span(fixture.result.get_gf(), region).valid());
        }
    }
    fprintf(stderr, "test-moe-cache: hybrid region span discovery and rejection OK\n");
}

void test_hybrid_admission() {
    const auto base = [](uint32_t domain, uint32_t rows, uint32_t sequences, uint64_t generation) {
        ggml_graph_execution_certificate certificate{};
        certificate.magic = GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC;
        certificate.abi_version = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;
        certificate.struct_size = sizeof(certificate);
        certificate.domain = domain;
        certificate.row_semantics = domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN ?
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE : GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
        certificate.owner_namespace = 723;
        certificate.owner_generation = generation;
        certificate.n_rows = rows;
        certificate.n_sequences = sequences;
        return certificate;
    };
    llm_graph_moe_hybrid_admission admission;
    llm_graph_moe_hybrid_submission submission;
    const auto main_certificate = base(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, 4, 1, 1);

    // No domain is certified by default.
    CHECK(admission.admit(main_certificate, 0, 0, 4, &submission) == llm_graph_moe_hybrid_admission::REJECTED_DOMAIN);
    CHECK(!submission.valid());
    admission.certify_domain(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, true);
    CHECK(admission.domain_certified(GGML_GRAPH_EXECUTION_DOMAIN_MAIN));

    // MAIN target verification must not certify the speculative domains.
    for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        CHECK(!admission.domain_certified(domain));
        CHECK(admission.admit(base(domain, 4, 1, 1), 0, 0, 4, &submission) == llm_graph_moe_hybrid_admission::REJECTED_DOMAIN);
    }
    CHECK(admission.admit(base(GGML_GRAPH_EXECUTION_DOMAIN_INVALID, 4, 1, 1), 0, 0, 4, &submission) ==
        llm_graph_moe_hybrid_admission::REJECTED_DOMAIN);

    CHECK(admission.admit(main_certificate, 0, 0, 4, &submission) == llm_graph_moe_hybrid_admission::ADMITTED);
    CHECK(submission.kind == LLM_GRAPH_MOE_HYBRID_KIND_TARGET_VERIFICATION);
    CHECK(submission.valid() && submission.submission_id == 1 && submission.domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN &&
        submission.owner_namespace == 723 && submission.owner_generation == 1 && submission.layer == 0 &&
        submission.semantic_group == 0);
    auto previous = submission;
    CHECK(admission.admit(main_certificate, 1, 2, 4, &submission) == llm_graph_moe_hybrid_admission::ADMITTED);
    CHECK(submission.submission_id == 2 && submission.layer == 1 && submission.semantic_group == 2);
    CHECK(!(submission == previous));

    // Malformed or uncertified certificates.
    const auto reject = [&](ggml_graph_execution_certificate certificate, int32_t layer, uint32_t group, int32_t rows,
                            llm_graph_moe_hybrid_admission::result expected) {
        CHECK(admission.admit(certificate, layer, group, rows, &submission) == expected);
        CHECK(!submission.valid());
    };
    auto bad = main_certificate; bad.magic = 0;
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_CERTIFICATE);
    bad = main_certificate; bad.abi_version = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION + 1;
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_CERTIFICATE);
    bad = main_certificate; bad.struct_size = sizeof(bad) - 1;
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_CERTIFICATE);
    bad = main_certificate; bad.owner_namespace = 0;
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_CERTIFICATE);
    bad = main_certificate; bad.owner_generation = 0;
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_CERTIFICATE);
    reject(main_certificate, -1, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_CERTIFICATE);
    reject(main_certificate, 0, UINT32_MAX, 4, llm_graph_moe_hybrid_admission::REJECTED_CERTIFICATE);
    bad = main_certificate; bad.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID;
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_DOMAIN);

    // Only the combinations the CUDA seam recognizes are classified.
    bad = main_certificate; bad.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_SEMANTICS);
    bad = base(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, 4, 4, 1);
    bad.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;  // verification needs n_sequences < n_rows
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_SEMANTICS);
    bad = base(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, 4, 1, 1);
    bad.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;  // ordinary decode, not verification
    CHECK(admission.admit(bad, 0, 0, 4, &submission) == llm_graph_moe_hybrid_admission::ADMITTED);
    CHECK(submission.kind == LLM_GRAPH_MOE_HYBRID_KIND_ORDINARY);

    // The certificate must describe the rows actually packed.
    reject(main_certificate, 0, 0, 3, llm_graph_moe_hybrid_admission::REJECTED_ROWS);
    reject(main_certificate, 0, 0, 0, llm_graph_moe_hybrid_admission::REJECTED_ROWS);
    bad = base(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, 4, 0, 1);
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_ROWS);
    bad = base(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, 4, 5, 1);
    reject(bad, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_ROWS);

    // A replaced owner restarts submission ids; an older generation is refused.
    CHECK(admission.admit(base(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, 4, 1, 2), 0, 0, 4, &submission) ==
        llm_graph_moe_hybrid_admission::ADMITTED);
    CHECK(submission.submission_id == 1 && submission.owner_generation == 2);
    reject(main_certificate, 0, 0, 4, llm_graph_moe_hybrid_admission::REJECTED_GENERATION);

    // Speculative domains are certified independently of MAIN.
    admission.certify_domain(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, true);
    CHECK(admission.admit(base(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 2, 2, 2), 3, 1, 2, &submission) ==
        llm_graph_moe_hybrid_admission::ADMITTED);
    CHECK(submission.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT && submission.kind == LLM_GRAPH_MOE_HYBRID_KIND_DRAFT);
    {
        auto sequential = base(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 4, 2, 2);
        sequential.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
        CHECK(admission.admit(sequential, 3, 1, 4, &submission) == llm_graph_moe_hybrid_admission::ADMITTED);
        CHECK(submission.kind == LLM_GRAPH_MOE_HYBRID_KIND_DRAFT_SEQUENTIAL);
        auto speculative = base(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 4, 2, 2);
        speculative.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
        reject(speculative, 3, 1, 4, llm_graph_moe_hybrid_admission::REJECTED_SEMANTICS);
        auto batched = base(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 4, 2, 2);  // independent draft rows must be one per sequence
        reject(batched, 3, 1, 4, llm_graph_moe_hybrid_admission::REJECTED_SEMANTICS);
    }
    CHECK(admission.admit(base(GGML_GRAPH_EXECUTION_DOMAIN_MTP, 2, 2, 2), 3, 1, 2, &submission) ==
        llm_graph_moe_hybrid_admission::REJECTED_DOMAIN);
    admission.certify_domain(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, false);
    CHECK(admission.admit(base(GGML_GRAPH_EXECUTION_DOMAIN_MAIN, 4, 1, 2), 0, 0, 4, &submission) ==
        llm_graph_moe_hybrid_admission::REJECTED_DOMAIN);
    CHECK(admission.admit(base(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 2, 2, 2), 3, 1, 2, &submission) ==
        llm_graph_moe_hybrid_admission::ADMITTED);

    admission.reset();
    CHECK(!admission.domain_certified(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT));
    CHECK(admission.admit(base(GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 2, 2, 2), 3, 1, 2, &submission) ==
        llm_graph_moe_hybrid_admission::REJECTED_DOMAIN);
    fprintf(stderr, "test-moe-cache: hybrid submission admission and per-domain certification OK\n");
}

void test_hybrid_worker_identity(const llm_graph_moe_hybrid_ffn & ffn, ggml_threadpool_t pool, int n_threads,
                                 const float * input, const int32_t * ids, const int32_t * rows, size_t n_routes,
                                 const std::vector<float> & expected, int n_dim) {
    llm_graph_moe_hybrid_submission submission;
    submission.owner_namespace = 9;
    submission.owner_generation = 2;
    submission.submission_id = 5;
    submission.domain = GGML_GRAPH_EXECUTION_DOMAIN_DRAFT;
    submission.semantic_group = 1;
    submission.layer = 0;
    submission.kind = LLM_GRAPH_MOE_HYBRID_KIND_DRAFT;
    std::vector<float> output(expected.size(), NAN);
    {
        llm_graph_moe_hybrid_worker worker;
        CHECK(!worker.busy() && worker.workspace_bytes() == 0);
        CHECK(!worker.submit(submission, input, ids, n_routes, output.data(), rows));
        worker.drain();
        CHECK(worker.init(ffn, pool, n_threads));
        CHECK(!worker.init(ffn, pool, n_threads));
        CHECK(worker.workspace_bytes() != 0);
        auto invalid = submission;
        invalid.submission_id = 0;
        CHECK(!worker.submit(invalid, input, ids, n_routes, output.data(), rows));
        CHECK(worker.submit(submission, input, ids, n_routes, output.data(), rows));
        for (auto field : {0, 1, 2, 3, 4, 5, 6}) {
            auto other = submission;
            switch (field) {
                case 6: other.kind = LLM_GRAPH_MOE_HYBRID_KIND_DRAFT_SEQUENTIAL; break;
                case 0: other.owner_namespace = 10; break;
                case 1: other.owner_generation = 3; break;
                case 2: other.submission_id = 6; break;
                case 3: other.domain = GGML_GRAPH_EXECUTION_DOMAIN_MAIN; break;
                case 4: other.semantic_group = 2; break;
                default: other.layer = 1; break;
            }
            CHECK(!worker.wait(other));
        }
        CHECK(worker.wait(submission));
        for (size_t i = 0; i < expected.size(); ++i) {
            CHECK(output[i] == expected[i]);
        }
        std::fill(output.begin(), output.end(), NAN);
        CHECK(worker.submit(submission, input, ids, n_routes, output.data(), rows));
        worker.drain();
        CHECK(!worker.busy() && !worker.wait(submission));
        worker.drain();
        int32_t invalid_id = n_experts;
        CHECK(worker.submit(submission, input, &invalid_id, 1, output.data(), rows));
        CHECK(!worker.wait(submission));
        CHECK(worker.submit(submission, input, ids, n_routes, output.data(), rows));
    }
    fprintf(stderr, "test-moe-cache: hybrid CPU worker submission identity and drain OK\n");
}

void test_hybrid_route_packet() {
    llm_graph_moe_hybrid_routes packet;
    CHECK(packet.init(4, 4, 2));
    const auto workspace = packet.workspace_bytes();
    std::vector<uint8_t> bytes(19, 0xff);
    const int32_t first[] = {3, 0}, second[] = {0, 2};
    memcpy(bytes.data(), first, sizeof(first));
    memcpy(bytes.data() + 11, second, sizeof(second));
    const int32_t slots[] = {1, -1, 0, -1};
    const auto valid = [&]() { return packet.build(bytes.data(), bytes.size(), 11, 2, 2, slots, 4); };
    CHECK(valid());
    CHECK((packet.gpu_ids() == std::vector<int32_t>{-1, 1, 1, 0}));
    CHECK((packet.cold_ids() == std::vector<int32_t>{3}));
    CHECK((packet.cold_rows() == std::vector<int32_t>{0}));
    CHECK((packet.cold_columns() == std::vector<int32_t>{0}));
    const auto reject = [&](bool success) {
        CHECK(!success);
        CHECK(packet.gpu_ids().empty() && packet.cold_ids().empty() && packet.cold_rows().empty() && packet.cold_columns().empty());
        CHECK(packet.workspace_bytes() == workspace);
        CHECK(valid());
    };
    reject(packet.build(bytes.data(), 18, 11, 2, 2, slots, 4));
    reject(packet.build(bytes.data(), bytes.size(), 7, 2, 2, slots, 4));
    reject(packet.build(bytes.data(), bytes.size(), SIZE_MAX, 2, 2, slots, 4));
    reject(packet.build(bytes.data(), bytes.size(), 11, 3, 2, slots, 4));
    reject(packet.build(bytes.data(), bytes.size(), 11, 2, 2, slots, 3));
    reject(packet.build(nullptr, bytes.size(), 11, 2, 2, slots, 4));
    for (int32_t invalid_slot : {-2, 2, 1}) {
        int32_t bad_slots[] = {1, invalid_slot, 0, -1};
        reject(packet.build(bytes.data(), bytes.size(), 11, 2, 2, bad_slots, 4));
    }
    for (int32_t invalid_id : {-1, 4, 3}) {
        auto bad_bytes = bytes;
        memcpy(bad_bytes.data() + sizeof(int32_t), &invalid_id, sizeof(invalid_id));
        reject(packet.build(bad_bytes.data(), bad_bytes.size(), 11, 2, 2, slots, 4));
    }
    llm_graph_moe_hybrid_routes no_gpu;
    CHECK(no_gpu.init(4, 4, 0));
    const int32_t cold[] = {-1, -1, -1, -1};
    CHECK(no_gpu.build(bytes.data(), bytes.size(), 11, 2, 2, cold, 4));
    CHECK((no_gpu.gpu_ids() == std::vector<int32_t>{-1, -1, -1, -1}));
    CHECK((no_gpu.cold_ids() == std::vector<int32_t>{3, 0, 0, 2}));
    CHECK((no_gpu.cold_columns() == std::vector<int32_t>{0, 1, 2, 3}));
    fprintf(stderr, "test-moe-cache: bounded hybrid route packet validation OK\n");
}

void test_hybrid_cpu_experts() {
    auto params = ggml_threadpool_params_default(2);
    auto * pool = ggml_threadpool_new(&params);
    CHECK(pool != nullptr);
    {
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        CHECK(cpu != nullptr);
        ggml_backend_cpu_set_threadpool(cpu.get(), pool);
        ggml_backend_cpu_set_n_threads(cpu.get(), params.n_threads);
        for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                                GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
                                GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED}) {
            layer_fixture fixture(0, layout, GGML_TYPE_Q4_0, false, {0, 0}, all_cached_layers, n_slots, false, GGML_TYPE_Q4_K);
            for (const auto & config : {std::make_pair(1, LLM_FFN_SILU), std::make_pair(1, LLM_FFN_GELU),
                                       std::make_pair(4, LLM_FFN_SILU), std::make_pair(4, LLM_FFN_GELU)}) {
                const int n_rows = config.first;
                const auto activation = config.second;
                fixture.build_graph(activation, 0.0f, false, false, n_rows);
                const auto & region = fixture.result.get_moe_regions().front();
                const auto & ffn = region.hybrid_ffn;
                ggml_set_output(region.route);
                ggml_set_output(ffn.input);
                ggml_set_output(ffn.down_output);
                ggml_gallocr_ptr allocator(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
                CHECK(ggml_gallocr_alloc_graph(allocator.get(), fixture.result.get_gf()));
                const int n_columns = n_rows * n_used;
                std::vector<float> input(n_dim * n_rows), expected(n_dim * n_columns), output(n_dim * n_columns);
                std::vector<int32_t> routes(n_columns), rows(n_columns), columns(n_columns);
                std::vector<uint8_t> route_bytes(ggml_nbytes(region.route));
                for (int i = 0; i < n_columns; ++i) {
                    rows[i] = i / n_used;
                    columns[i] = i;
                }
                int reference_iteration = 0;
                const auto reference = [&](int iteration) {
                    reference_iteration = iteration;
                    fixture.set_inputs(iteration);
                    for (int row = 0; row < n_rows; ++row) {
                        for (int i = 0; i < n_dim; ++i) {
                            input[row * n_dim + i] = 0.02f * (row + 1) * ((i + 5 * row + iteration) % 17 - 8);
                        }
                    }
                    ggml_backend_tensor_set(fixture.input, input.data(), 0, input.size() * sizeof(float));
                    CHECK(ggml_backend_graph_compute(cpu.get(), fixture.result.get_gf()) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(ffn.input, input.data(), 0, input.size() * sizeof(float));
                    ggml_backend_tensor_get(ffn.down_output, expected.data(), 0, expected.size() * sizeof(float));
                    ggml_backend_tensor_get(region.route, route_bytes.data(), 0, route_bytes.size());
                    for (int row = 0; row < n_rows; ++row) {
                        ggml_backend_tensor_get(region.route, routes.data() + row * n_used, row * region.route->nb[1], n_used * sizeof(int32_t));
                    }
                };
                reference(0);
                llm_graph_moe_hybrid_cpu helper;
                CHECK(helper.init(ffn, pool, params.n_threads));
                const auto workspace = helper.workspace_bytes();
                const auto check = [&](const std::vector<int32_t> & selected, const int32_t * packed_ids = nullptr, const int32_t * packed_rows = nullptr) {
                    std::vector<int32_t> job_ids, job_rows;
                    for (auto column : selected) {
                        job_ids.push_back(routes[column]);
                        job_rows.push_back(rows[column]);
                    }
                    std::fill(output.begin(), output.end(), NAN);
                    CHECK(helper.compute(input.data(), packed_ids ? packed_ids : job_ids.data(), selected.size(), output.data(),
                        packed_rows ? packed_rows : job_rows.data()));
                    for (size_t column = 0; column < selected.size(); ++column) {
                        for (size_t i = 0; i < n_dim; ++i) {
                            const float ref = expected[selected[column] * n_dim + i];
                            const float actual = output[column * n_dim + i];
                            const bool match = std::isfinite(actual) && fabsf(actual - ref) <= 1e-4f * (1.0f + fabsf(ref));
                            if (!match) {
                                fprintf(stderr, "hybrid CPU mismatch: layout=%u activation=%d rows=%d iteration=%d job=%zu column=%zu source=%d element=%zu actual=%g reference=%g\n",
                                    layout, activation, n_rows, reference_iteration, selected.size(), column, selected[column], i, actual, ref);
                            }
                            CHECK(match);
                        }
                    }
                    CHECK(std::all_of(output.begin() + selected.size() * n_dim, output.end(), [](float value) { return std::isnan(value); }));
                    CHECK(helper.workspace_bytes() == workspace);
                };
                llm_graph_moe_hybrid_routes packet;
                CHECK(packet.init(n_columns, n_experts, n_experts));
                const auto packet_workspace = packet.workspace_bytes();
                const auto check_packet = [&](int mode) {
                    std::vector<int32_t> slots(n_experts, -1);
                    for (int expert = 0; expert < n_experts; ++expert) {
                        if (mode == 2 || (mode == 1 && expert % 2 == 0)) {
                            slots[expert] = expert;
                        }
                    }
                    CHECK(packet.build(route_bytes.data(), route_bytes.size(), region.route->nb[1], n_used, n_rows, slots.data(), slots.size()));
                    CHECK(packet.gpu_ids().size() == routes.size());
                    std::vector<int> coverage(n_columns, 0);
                    for (int column = 0; column < n_columns; ++column) {
                        CHECK(packet.gpu_ids()[column] == slots[routes[column]]);
                        coverage[column] += packet.gpu_ids()[column] >= 0;
                    }
                    for (size_t cold = 0; cold < packet.cold_ids().size(); ++cold) {
                        const auto column = packet.cold_columns()[cold];
                        CHECK(packet.cold_ids()[cold] == routes[column] && packet.cold_rows()[cold] == rows[column]);
                        CHECK(packet.gpu_ids()[column] == -1);
                        ++coverage[column];
                    }
                    CHECK(std::all_of(coverage.begin(), coverage.end(), [](int count) { return count == 1; }));
                    check(packet.cold_columns(), packet.cold_ids().data(), packet.cold_rows().data());
                    CHECK(packet.workspace_bytes() == packet_workspace);
                };
                check_packet(0);
                check_packet(1);
                check_packet(2);
                check(columns);
                check({1});
                check({1, 1});
                if (n_rows > 1) {
                    CHECK(routes[1] == routes[2 * n_used + 1]);
                    check({n_columns - 1, 1, 2 * n_used + 1, 1});
                    std::vector<float> worker_expected(n_columns * n_dim);
                    CHECK(helper.compute(input.data(), routes.data(), n_columns, worker_expected.data(), rows.data()));
                    test_hybrid_worker_identity(ffn, pool, params.n_threads, input.data(), routes.data(), rows.data(),
                        n_columns, worker_expected, n_dim);
                } else {
                    CHECK(helper.compute(input.data(), routes.data(), n_columns, output.data()));
                }
                reference(3);
                check_packet(1);
                check({1, 1});
                check(columns);
                CHECK(helper.compute(nullptr, nullptr, 0, nullptr));
                std::fill(output.begin(), output.end(), -100.0f);
                int32_t invalid[] = {routes[0], n_experts};
                CHECK(!helper.compute(input.data(), invalid, n_used, output.data(), rows.data()));
                invalid[1] = -1;
                CHECK(!helper.compute(input.data(), invalid, n_used, output.data(), rows.data()));
                CHECK(!helper.compute(nullptr, routes.data(), n_used, output.data(), rows.data()));
                CHECK(!helper.compute(input.data(), routes.data(), n_columns + 1, output.data(), rows.data()));
                int32_t bad_row = n_rows;
                CHECK(!helper.compute(input.data(), routes.data(), 1, output.data(), &bad_row));
                bad_row = -1;
                CHECK(!helper.compute(input.data(), routes.data(), 1, output.data(), &bad_row));
                if (n_rows > 1) {
                    CHECK(!helper.compute(input.data(), routes.data(), n_columns, output.data()));
                }
                CHECK(std::all_of(output.begin(), output.end(), [](float value) { return value == -100.0f; }));
            }
            fixture.build_graph(LLM_FFN_SILU, 0.0f, true);
            llm_graph_moe_hybrid_cpu helper;
            CHECK(!helper.init(fixture.result.get_moe_regions().front().hybrid_ffn, pool, params.n_threads));
        }
        layer_fixture device_weights(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, {0, 0}, 0);
        llm_graph_moe_hybrid_cpu helper;
        CHECK(!helper.init(device_weights.result.get_moe_regions().front().hybrid_ffn, pool, params.n_threads));
    }
    ggml_threadpool_free(pool);
    fprintf(stderr, "test-moe-cache: hybrid CPU expert single/multi-row arithmetic parity OK\n");
}

}  // namespace

void test_moe_sched_compute_range() {
    test_sched_boundaries_auto_alloc();
    test_sched_cached_boundaries();
    test_sched_cached_boundaries(LLM_FFN_GELU);
    test_sched_compute_range();
    test_sched_compute_range(true);
    test_sched_compute_range(false, true);
    test_sched_compute_range(true, true);
    test_sched_compute_range(false, true, 4);
    test_sched_compute_range(true, true, 4);
}

void test_moe_hybrid_ffn_insertion() {
    test_hybrid_ffn_insertion();
    test_hybrid_frozen_partition();
}

void test_grouped_layer_placement() {
    const bool debug = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    test_hybrid_route_packet();
    test_hybrid_admission();
    test_hybrid_span();
    test_moe_sched_compute_range();
    test_hybrid_cpu_experts();
    {
        layer_fixture fixture(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0);
        fixture.build_graph(LLM_FFN_GELU);
        fixture.build_graph(LLM_FFN_SILU, 7.0f);
        fixture.build_graph(LLM_FFN_RELU);
        fixture.build_graph(LLM_FFN_SILU, 0.0f, true);
        fixture.build_graph(LLM_FFN_SILU, 0.0f, false, true);
        fixture.result.reset();
        CHECK(fixture.result.get_moe_regions().empty());
    }
    run_layers({ device }, 0);
    run_layers({ device }, bounded_host_budget);
    run_layers({ device }, 0, true);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 0);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 1);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 0);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false,
               all_cached_layers, n_experts);
    for (uint32_t mask : {1u, 2u, all_cached_layers}) {
        run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false,
                   mask, n_slots, true);
    }
    test_shared_source_owners(device, 0);
    test_shared_source_owners(device, 1024 * 1024);
    test_shared_source_owners(device, 0, true);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, true);
    if (ggml_backend_reg_dev_count(ggml_backend_cuda_reg()) >= 3) {
        run_layers({ 0, 2 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, true);
        fprintf(stderr, "test-moe-cache: pageable layer owners 0/2 exact OK (logical device coverage)\n");
    }
    ggml_backend_cuda_moe_set_debug_mm(debug);
    fprintf(stderr, "test-moe-cache: single-device layer fixture OK (not multi-GPU validation)\n");
}

int test_grouped_multigpu() {
    int        count  = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count < 2)) {
        fprintf(stderr, "SKIP: grouped multi-GPU requires two distinct physical CUDA devices; found %d\n", count);
        return 77;
    }
    CUDA_OK(status);
    ggml_backend_dev_props props[2]{};
    auto *                 reg = ggml_backend_cuda_reg();
    if (ggml_backend_reg_dev_count(reg) < 2) {
        fprintf(stderr, "SKIP: CUDA registry exposes fewer than two backends\n");
        return 77;
    }
    for (int i = 0; i < 2; ++i) {
        ggml_backend_dev_get_props(ggml_backend_reg_dev_get(reg, i), &props[i]);
    }
    if (props[0].device_id && props[1].device_id && strcmp(props[0].device_id, props[1].device_id) == 0) {
        fprintf(stderr, "SKIP: logical CUDA backends share physical identity %s\n", props[0].device_id);
        return 77;
    }
    cudaDeviceProp first, second;
    CUDA_OK(cudaGetDeviceProperties(&first, 0));
    CUDA_OK(cudaGetDeviceProperties(&second, 1));
    if (memcmp(&first.uuid, &second.uuid, sizeof(first.uuid)) == 0 ||
        (first.pciDomainID == second.pciDomainID && first.pciBusID == second.pciBusID &&
         first.pciDeviceID == second.pciDeviceID)) {
        fprintf(stderr, "SKIP: grouped multi-GPU devices share a physical GPU\n");
        return 77;
    }
    for (int device = 0; device < 2; ++device) {
        cudaDeviceProp prop;
        CUDA_OK(cudaGetDeviceProperties(&prop, device));
        int peer = 0;
        CUDA_OK(cudaDeviceCanAccessPeer(&peer, device, 1 - device));
        fprintf(stderr, "test-moe-cache: physical device=%d pci=%04x:%02x:%02x cc=%d.%d peer=%d name=%s\n", device,
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.major, prop.minor, peer, prop.name);
    }
    const bool debug = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    run_layers({ 0, 1 }, 0);
    run_layers({ 0, 1 }, bounded_host_budget);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 1);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 0);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, true);
    run_layers({ 0, 1 }, 0, true);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    ggml_backend_cuda_moe_set_debug_mm(debug);
    fprintf(stderr, "test-moe-cache: two-physical-device grouped layer execution OK\n");
    return 0;
}
