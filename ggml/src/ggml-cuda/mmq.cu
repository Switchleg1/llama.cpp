#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

template <bool use_x_map>
static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, use_x_map>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, use_x_map>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, use_x_map>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4, use_x_map>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mmq_prepare_id_input(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_cuda_mmq_id_input & input) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && src1->ne[3] == 1);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t width = src1->ne[0];
    const int64_t padded = GGML_PAD(width, MATRIX_ROW_PADDING);
    const int64_t channels = src1->ne[1];
    const int64_t tokens = src1->ne[2];
    const int64_t used = ids->ne[0];
    const int64_t rows = tokens * used;
    const int tail = ggml_cuda_mmq_get_J_max(src0->type, src0->ne[1] % 128 != 0, cc);
    const bool native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t block_bytes = native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t block_values = native_fp4 ? QK_FP4_MMQ : QK8_1_MMQ;
    GGML_ASSERT(rows > 0 && tail > 0 && (uint64_t) rows <= SIZE_MAX - (size_t) tail);
    input.ids_dst.alloc(rows + tail);
    input.expert_bounds.alloc(src0->ne[2] + 1);
    input.data.alloc(rows * padded * block_bytes / block_values + tail * sizeof(block_q8_1_mmq));
    if (src0->type == GGML_TYPE_NVFP4 && native_fp4) {
        input.scales.alloc(rows + tail);
    }
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), rows);
    const bool broadcast = channels == 1 && used > 1;
    const size_t ts = ggml_type_size(src1->type);
    GGML_ASSERT(ids->nb[0] == sizeof(int32_t) && src1->nb[2] % src1->nb[1] == 0);
    ggml_cuda_launch_mm_ids_helper(static_cast<const int32_t *>(ids->data), ids_src1.get(), input.ids_dst.get(),
        input.expert_bounds.get(), src0->ne[2], tokens, used, channels, ids->nb[1] / sizeof(int32_t),
        src1->nb[2] / src1->nb[1], broadcast, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    const float * source = static_cast<const float *>(src1->data);
    if (native_fp4) {
        const bool aligned = ggml_cuda_is_aligned(src1, 32);
        if (broadcast) {
            quantize_scatter_mmq_fp4_cuda(source, ids_src1.get(), input.data.get(), input.scales.ptr, src0->type,
                aligned, width, src1->nb[2] / ts, padded, tokens, rows, used, ctx.stream());
        } else {
            quantize_mmq_fp4_cuda(source, ids_src1.get(), input.data.get(), input.scales.ptr, src0->type, aligned,
                width, src1->nb[1] / ts, src1->nb[2] / ts, src1->nb[3] / ts, padded, rows, 1, 1, ctx.stream());
        }
    } else if (broadcast) {
        quantize_scatter_mmq_q8_1_cuda(source, ids_src1.get(), input.data.get(), src0->type,
            width, src1->nb[2] / ts, padded, tokens, rows, used, ctx.stream());
    } else {
        quantize_mmq_q8_1_cuda(source, ids_src1.get(), input.data.get(), src0->type,
            width, src1->nb[1] / ts, src1->nb[2] / ts, src1->nb[3] / ts, padded, rows, 1, 1, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
    input.n_experts = src0->ne[2];
}

// MoE launch columns: twice the average tokens per expert
static int64_t ggml_cuda_mmq_moe_launch_cols(int64_t rows, int64_t n_experts) {
    return 2*((rows + n_experts - 1) / n_experts);
}

static void ggml_cuda_mul_mat_q_impl(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const void * src0_secondary,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const int32_t * source_map, int32_t source_split,
        const int32_t * source_wait_class, const uint32_t * stage_ready) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT((source_map == nullptr) == (src0_secondary == nullptr));
    GGML_ASSERT(source_map == nullptr || (ids != nullptr && source_split > 0));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;
    const int mmq_max_tile_y = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc);
    GGML_ASSERT(mmq_max_tile_y > 0);

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            mmq_max_tile_y * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11 + mmq_max_tile_y);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);
    GGML_ASSERT(ne_get_rows > 0 && (uint64_t) ne_get_rows <= SIZE_MAX - (size_t) mmq_max_tile_y);
    ggml_cuda_mmq_id_input input(ctx.pool());
    ggml_cuda_mmq_prepare_id_input(ctx, src0, src1, ids, input);

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    mmq_args args = {
        src0_d, src0->type, reinterpret_cast<const int *>(input.data.ptr), input.ids_dst.ptr, input.expert_bounds.ptr, dst_d,
        input.scales.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};
    args.ncols_launch = ggml_cuda_mmq_moe_launch_cols(ne_get_rows, ne02);

    args.x_secondary = (const char *) src0_secondary;
    args.x_channel_map = source_map;
    args.x_channel_split = source_split;
    args.x_wait_class = source_wait_class;
    args.x_stage_ready = stage_ready;
    if (source_map) {
        ggml_cuda_mul_mat_q_switch_type<true>(ctx, args, stream);
    } else {
        ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, stream);
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    ggml_cuda_mul_mat_q_impl(ctx, src0, nullptr, src1, ids, dst, nullptr, 0, nullptr, nullptr);
}

void ggml_cuda_mul_mat_q_prepared(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mmq_id_input & input, int64_t first_expert) {
    GGML_ASSERT(first_expert >= 0 && first_expert + src0->ne[2] <= input.n_experts);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[3] == 1);
    GGML_TENSOR_BINARY_OP_LOCALS;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const int64_t padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t rows = ne12 * ids->ne[0];
    const int64_t stride_y = native_fp4 ? ne11 * padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
        ne11 * padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (rows + ne02 - 1) / ne02;
    }
    mmq_args args = {
        static_cast<const char *>(src0->data), src0->type, reinterpret_cast<const int *>(input.data.ptr),
        input.ids_dst.ptr, input.expert_bounds.ptr + first_expert, static_cast<float *>(dst->data), input.scales.ptr,
        ne00, ne01, rows, static_cast<int64_t>(nb01 / ggml_type_size(src0->type)), rows, static_cast<int64_t>(nb1 / sizeof(float)),
        ne02, ne02, static_cast<int64_t>(nb02 / ggml_type_size(src0->type)), stride_y, static_cast<int64_t>(nb2 / sizeof(float)),
        ne03, ne13, static_cast<int64_t>(nb03 / ggml_type_size(src0->type)), ne12 * stride_y, static_cast<int64_t>(nb3 / sizeof(float)), ne12, ncols_opt};
    // the average over all experts of the input, not just this group
    args.ncols_launch = ggml_cuda_mmq_moe_launch_cols(rows, input.n_experts);
    ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, ctx.stream());
}

void ggml_cuda_mul_mat_q_mapped(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const void * src0_secondary,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const int32_t * source_map, int32_t source_split,
        const int32_t * source_wait_class, const uint32_t * stage_ready) {
    GGML_ASSERT(src0_secondary != nullptr && source_map != nullptr && source_split > 0);
    GGML_ASSERT((source_wait_class == nullptr) == (stage_ready == nullptr));
    GGML_ASSERT(src0->type != GGML_TYPE_MXFP4 && src0->type != GGML_TYPE_NVFP4);
    ggml_cuda_mul_mat_q_impl(
        ctx, src0, src0_secondary, src1, ids, dst,
        source_map, source_split, source_wait_class, stage_ready);
}

static bool ggml_cuda_mmq_type_supported(enum ggml_type type) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

static bool ggml_cuda_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts, size_t smpbo) {
    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    if (smpbo < 48 * 1024) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
    if (!ggml_cuda_mmq_type_supported(type)) {
        return false;
    }
    const int id = ggml_cuda_get_device();
    return ggml_cuda_should_use_mmq_impl(type, cc, ne11, n_experts, ggml_cuda_info().devices[id].smpbo);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts, size_t smpbo) {
    if (!ggml_cuda_mmq_type_supported(type)) {
        return false;
    }
    return ggml_cuda_should_use_mmq_impl(type, cc, ne11, n_experts, smpbo);
}
