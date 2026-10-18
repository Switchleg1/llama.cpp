#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post_inject(ggml_backend_cuda_context & ctx, const ggml_tensor * scale1, const ggml_tensor * scale2, ggml_tensor * dst);
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, ggml_tensor * dst);
