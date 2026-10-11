#pragma once

#include <cstdint>

struct ggml_tensor;

// CPU expert FFN that consumes the GPU's 32-wide Q8_1 activation, so the hybrid halves share quantization.
// Weights are read in place from host memory, indexed by expert.

bool llama_moe_hybrid_q8_1_supported(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down);

// x_q8_1 holds gate->ne[0]/32 block_q8_1, shared by all routes. Route i writes down->ne[1] floats at
// out + i*out_stride. All routes run as one parallel job per projection.
bool llama_moe_hybrid_q8_1_experts(
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const int32_t * experts, uint32_t count, const void * x_q8_1, float * out, size_t out_stride, bool gelu);

// The CUDA quantize_q8_1 rule, for the intermediate activation.
void llama_moe_hybrid_quantize_q8_1(const float * x, void * y, int64_t n);
