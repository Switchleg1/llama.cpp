#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;
class llama_moe_expert_store;

// CPU expert FFN that consumes the GPU's 32-wide Q8_1 activation, so the hybrid halves share quantization.
// Weights are read in place from host memory, indexed by expert.

bool llama_moe_hybrid_q8_1_supported(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down);

// x_q8_1 holds gate->ne[0]/32 block_q8_1, shared by all routes. Route i writes down->ne[1] floats at
// out + i*out_stride. All routes run as one parallel job per projection.
// Route i uses activation row rows[i] at x_q8_1 + rows[i]*x_row_bytes. Routes sharing an expert are computed
// together, so each weight row is read once for all of them.
bool llama_moe_hybrid_q8_1_experts(
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const int32_t * experts, const int32_t * rows, uint32_t count, const void * x_q8_1, size_t x_row_bytes,
        float * out, size_t out_stride, bool gelu, const float * x_f32 = nullptr, size_t x_f32_stride = 0);

// Same arithmetic for up to 64 routes, with bounded store leases. Holds as many bundles as staging allows, then flushes and retries.
// A false result leaves output unchanged. Fail the batch; do not fall back to original bank pointers.
bool llama_moe_hybrid_q8_1_store_experts(
        llama_moe_expert_store & store, uint32_t group,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const int32_t * experts, const int32_t * rows, uint32_t count, const void * x_q8_1, size_t x_row_bytes,
        float * out, size_t out_stride, bool gelu, const float * x_f32 = nullptr, size_t x_f32_stride = 0);

// Threads of the CPU expert pool. Takes effect only before the first expert computation.
void llama_moe_hybrid_set_threads(int n_threads);

// The CUDA quantize_q8_1 rule, for the intermediate activation.
void llama_moe_hybrid_quantize_q8_1(const float * x, void * y, int64_t n);
