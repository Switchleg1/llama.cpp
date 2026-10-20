#include "llama-moe-hybrid-kernels.h"
#include "llama-moe-expert-store.h"
#include "../ggml/src/ggml-backend-moe.h"

#include "ggml.h"
#include "ggml-cpu.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../ggml/src/ggml-common.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// CPU dot products against 32-wide Q8_1 activations, the CUDA rule. Both halves of the hybrid split then
// consume identical quantized activations, so their remaining disagreement is float summation order.

static inline float fp16(ggml_half h) {
    return ggml_fp16_to_fp32(h);
}

static float dot_iq2_s(const void * vx, const block_q8_1 * y, int64_t n) {
    const auto * x = static_cast<const block_iq2_s *>(vx);
    float sumf = 0.0f;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float dx = fp16(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const block_q8_1 & b = y[i*(QK_K/32) + ib32];
            const int8_t * q8 = b.qs;
            const int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            const int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *) (iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2*l) & 0x300)));
                int sumi = 0;
                for (int j = 0; j < 8; ++j) {
                    sumi += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                (l < 2 ? sumi1 : sumi2) += sumi;
                q8 += 8;
            }
            sumf += dx * fp16(b.data.data.d) * (float) (ls1*sumi1 + ls2*sumi2) * 0.125f;
            qs += 4;
            signs += 4;
        }
    }
    return sumf;
}

static float dot_iq3_xxs(const void * vx, const block_q8_1 * y, int64_t n) {
    const auto * x = static_cast<const block_iq3_xxs *>(vx);
    float sumf = 0.0f;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float dx = fp16(x[i].d);
        const uint8_t * q3 = x[i].qs;
        const uint8_t * gas = x[i].qs + QK_K/4;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const block_q8_1 & b = y[i*(QK_K/32) + ib32];
            const int8_t * q8 = b.qs;
            uint32_t aux32;
            memcpy(&aux32, gas, sizeof(aux32));
            gas += sizeof(aux32);
            const int ls = 2*(aux32 >> 28) + 1;
            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2*l + 0]);
                const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2*l + 1]);
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * (signs & kmask_iq2xs[j + 0] ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * (signs & kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            sumf += dx * fp16(b.data.data.d) * (float) (sumi * ls) * 0.25f;
        }
    }
    return sumf;
}


static float dot_iq3_s(const void * vx, const block_q8_1 * y, int64_t n) {
    const auto * x = static_cast<const block_iq3_s *>(vx);
    float sumf = 0.0f;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float dx = fp16(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const block_q8_1 & b = y[i*(QK_K/32) + ib32];
            const int8_t * q8 = b.qs;
            const int ls = 2*((ib32 % 2 == 0 ? x[i].scales[ib32/2] & 0xf : x[i].scales[ib32/2] >> 4)) + 1;
            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *) (iq3s_grid + (qs[2*l+0] | ((qh[ib32] << (8 - 2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *) (iq3s_grid + (qs[2*l+1] | ((qh[ib32] << (7 - 2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * (signs[l] & kmask_iq2xs[j + 0] ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * (signs[l] & kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            sumf += dx * fp16(b.data.data.d) * (float) (sumi * ls);
        }
    }
    return sumf;
}

static float dot_iq4_xs(const void * vx, const block_q8_1 * y, int64_t n) {
    const auto * x = static_cast<const block_iq4_xs *>(vx);
    float sumf = 0.0f;
    for (int64_t ibl = 0; ibl < n / QK_K; ++ibl) {
        const float dx = fp16(x[ibl].d);
        const uint8_t * qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const block_q8_1 & b = y[ibl*(QK_K/32) + ib];
            const int ls = ((x[ibl].scales_l[ib/2] >> 4*(ib % 2)) & 0xf) | (((x[ibl].scales_h >> 2*ib) & 3) << 4);
            int sumi = 0;
            for (int j = 0; j < 16; ++j) {
                sumi += b.qs[j +  0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi += b.qs[j + 16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += dx * fp16(b.data.data.d) * (float) ((ls - 32) * sumi);
            qs += 16;
        }
    }
    return sumf;
}


#if defined(__x86_64__) || defined(_M_X64)
#define LLAMA_MOE_HYBRID_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

static bool cpu_has_avx2_fma() {
    static const bool has = []() {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 1);
        const bool fma = (info[2] & (1 << 12)) != 0;
        const bool osxsave = (info[2] & (1 << 27)) != 0;
        __cpuidex(info, 7, 0);
        const bool avx2 = (info[1] & (1 << 5)) != 0;
        return fma && osxsave && avx2 && (_xgetbv(0) & 6) == 6;
#else
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    }();
    return has;
}

#if !defined(_MSC_VER) || defined(__clang__)
#define LLAMA_MOE_AVX2 __attribute__((target("avx2,fma")))
#else
#define LLAMA_MOE_AVX2
#endif

LLAMA_MOE_AVX2 static inline float hsum8(__m256 v) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// Eight +1/-1 bytes per iq2xs sign pattern, parity bit included, like ggml's keven_signs_q2xs.
static const uint64_t * even_signs() {
    static const std::vector<uint64_t> table = []() {
        std::vector<uint64_t> t(128);
        for (int i = 0; i < 128; ++i) {
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j) {
                const uint8_t b = (ksigns_iq2xs[i] >> j) & 1 ? 0xff : 0x01;
                v |= (uint64_t) b << (8*j);
            }
            t[i] = v;
        }
        return t;
    }();
    return table.data();
}

// ggml's AVX2 iq3_xxs kernel, with the integer sum scaled per 32-block by the Q8_1 block's own scale.
LLAMA_MOE_AVX2 static float dot_iq3_xxs_avx2(const void * vx, const block_q8_1 * y, int64_t n) {
    const auto * x = static_cast<const block_iq3_xxs *>(vx);
    const uint64_t * signs64 = even_signs();
    uint32_t aux32[2];
    __m256 accumf = _mm256_setzero_ps();
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float dx = fp16(x[i].d);
        const uint8_t * q3 = x[i].qs;
        const uint8_t * gas = x[i].qs + QK_K/4;
        const block_q8_1 * yb = y + i*(QK_K/32);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) yb[ib32 + 0].qs);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) yb[ib32 + 1].qs);
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8);
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short) (2*(aux32[0] >> 28) + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short) (2*(aux32[1] >> 28) + 1)));
            accumf = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib32 + 0].data.data.d)), _mm256_cvtepi32_ps(p1), accumf);
            accumf = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib32 + 1].data.data.d)), _mm256_cvtepi32_ps(p2), accumf);
        }
    }
    return 0.25f * hsum8(accumf);
}

// ggml's AVX2 iq2_s kernel, with the same per 32-block scaling.
LLAMA_MOE_AVX2 static float dot_iq2_s_avx2(const void * vx, const block_q8_1 * y, int64_t n) {
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const auto * x = static_cast<const block_iq2_s *>(vx);
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);
    __m256 accumf = _mm256_setzero_ps();
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float dx = fp16(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = (const uint16_t *) (x[i].qs + QK_K/8);
        const block_q8_1 * yb = y + i*(QK_K/32);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) yb[ib32 + 0].qs);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) yb[ib32 + 1].qs);
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;
            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            // Each half of a 32-block has its own 4-bit scale: low 16 values use the low nibble.
            const __m256i sc1 = _mm256_set_m128i(_mm_set1_epi16((short) (1 + 2*(x[i].scales[ib32+0] >> 4))),
                                                 _mm_set1_epi16((short) (1 + 2*(x[i].scales[ib32+0] & 0xf))));
            const __m256i sc2 = _mm256_set_m128i(_mm_set1_epi16((short) (1 + 2*(x[i].scales[ib32+1] >> 4))),
                                                 _mm_set1_epi16((short) (1 + 2*(x[i].scales[ib32+1] & 0xf))));
            const __m256i p1 = _mm256_madd_epi16(dot1, sc1);
            const __m256i p2 = _mm256_madd_epi16(dot2, sc2);
            accumf = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib32 + 0].data.data.d)), _mm256_cvtepi32_ps(p1), accumf);
            accumf = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib32 + 1].data.data.d)), _mm256_cvtepi32_ps(p2), accumf);
        }
    }
    return 0.125f * hsum8(accumf);
}

// ggml's AVX2 iq3_s kernel, scaled per 32-block by the Q8_1 block's own scale.
LLAMA_MOE_AVX2 static float dot_iq3_s_avx2(const void * vx, const block_q8_1 * y, int64_t n) {
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const auto * x = static_cast<const block_iq3_s *>(vx);
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);
    union {
        __m256i  vec[2];
        uint32_t index[16];
    } idx;
    __m256 accumf = _mm256_setzero_ps();
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float dx = fp16(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = (const uint16_t *) x[i].signs;
        const block_q8_1 * yb = y + i*(QK_K/32);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) yb[ib32 + 0].qs);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) yb[ib32 + 1].qs);
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) qs));
            qs += 16;
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 0]), idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 1]), idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]);
            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short) (2*(x[i].scales[ib32/2] & 0xf) + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short) (2*(x[i].scales[ib32/2] >> 4) + 1)));
            accumf = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib32 + 0].data.data.d)), _mm256_cvtepi32_ps(p1), accumf);
            accumf = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib32 + 1].data.data.d)), _mm256_cvtepi32_ps(p2), accumf);
        }
    }
    return hsum8(accumf);
}

// ggml's AVX2 iq4_xs kernel, scaled per 32-block by the Q8_1 block's own scale.
LLAMA_MOE_AVX2 static float dot_iq4_xs_avx2(const void * vx, const block_q8_1 * y, int64_t n) {
    const auto * x = static_cast<const block_iq4_xs *>(vx);
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    __m256 accum = _mm256_setzero_ps();
    for (int64_t ibl = 0; ibl < n / QK_K; ++ibl) {
        const float dx = fp16(x[ibl].d);
        const uint8_t * qs = x[ibl].qs;
        const block_q8_1 * yb = y + ibl*(QK_K/32);
        uint16_t sh = x[ibl].scales_h;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *) qs);
            qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *) qs);
            qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *) yb[ib + 0].qs);
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *) yb[ib + 1].qs);
            const __m256i q4b_1 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                                   _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                                   _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            const __m256i p16_1 = _mm256_maddubs_epi16(_mm256_sign_epi8(q4b_1, q4b_1), _mm256_sign_epi8(q8b_1, q4b_1));
            const __m256i p16_2 = _mm256_maddubs_epi16(_mm256_sign_epi8(q4b_2, q4b_2), _mm256_sign_epi8(q8b_2, q4b_2));
            const short ls1 = (short) (((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const short ls2 = (short) (((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;
            const __m256i p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1));
            const __m256i p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2));
            accum = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib + 0].data.data.d)), _mm256_cvtepi32_ps(p_1), accum);
            accum = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16(yb[ib + 1].data.data.d)), _mm256_cvtepi32_ps(p_2), accum);
        }
    }
    return hsum8(accum);
}
#endif

using dot_fn = float (*)(const void *, const block_q8_1 *, int64_t);

static dot_fn custom_dot_for(ggml_type type) {
#if defined(LLAMA_MOE_HYBRID_X86)
    static const bool scalar = ggml_moe_debug_getenv("LLAMA_MOE_HYBRID_SCALAR") != nullptr;
    if (!scalar && cpu_has_avx2_fma()) {
        switch (type) {
            case GGML_TYPE_IQ2_S:   return dot_iq2_s_avx2;
            case GGML_TYPE_IQ3_XXS: return dot_iq3_xxs_avx2;
            case GGML_TYPE_IQ3_S:   return dot_iq3_s_avx2;
            case GGML_TYPE_IQ4_XS:  return dot_iq4_xs_avx2;
            default:                break;
        }
    }
#endif
    switch (type) {
        case GGML_TYPE_IQ2_S:   return dot_iq2_s;
        case GGML_TYPE_IQ3_XXS: return dot_iq3_xxs;
        case GGML_TYPE_IQ3_S:   return dot_iq3_s;
        case GGML_TYPE_IQ4_XS:  return dot_iq4_xs;
        default:                return nullptr;
    }
}

// How one bank consumes the shared Q8_1 activation. Q8_K banks (k-quants) take the fp32 activation quantized on
// the CPU instead, as ggml's CPU path does.
enum bank_kind { BANK_NONE, BANK_GGML_Q8_1, BANK_GGML_Q8_0, BANK_GGML_Q8_K, BANK_CUSTOM };

static std::atomic<bool> & native_dot() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

void llama_moe_hybrid_set_native_dot(bool enabled) {
    native_dot().store(enabled, std::memory_order_relaxed);
}

static bank_kind kind_for(ggml_type type) {
    // Types that already pair with Q8_1 or Q8_0 on the CPU: ggml's SIMD kernels take the GPU's bytes,
    // passed through or with the 2-byte sum dropped. The integer products are then the GPU's.
    const auto * traits = ggml_get_type_traits_cpu(type);
    // Q8_0 and Q8_1 are 32 wide, so any weight block that is a multiple of 32 consumes one scale per 32.
    if (traits != nullptr && traits->vec_dot != nullptr && ggml_blck_size(type) % QK8_1 == 0) {
        if (traits->vec_dot_type == GGML_TYPE_Q8_1) return BANK_GGML_Q8_1;
        if (traits->vec_dot_type == GGML_TYPE_Q8_0) return BANK_GGML_Q8_0;
    }
    // Custom kernels take the GPU's Q8_1 bytes, so they win over a Q8_K requantization.
    if (!native_dot().load(std::memory_order_relaxed) && custom_dot_for(type) != nullptr) {
        return BANK_CUSTOM;
    }
    if (traits != nullptr && traits->vec_dot != nullptr && traits->vec_dot_type == GGML_TYPE_Q8_K &&
            ggml_blck_size(type) == QK_K) {
        return BANK_GGML_Q8_K;
    }
    return BANK_NONE;
}

static void repack_q8_0(const block_q8_1 * x, block_q8_0 * y, int64_t n_blocks) {
    for (int64_t i = 0; i < n_blocks; ++i) {
        y[i].d = x[i].data.data.d;
        memcpy(y[i].qs, x[i].qs, QK8_0);
    }
}

// One row of a bank against the shared activation, in whichever form that bank consumes.
static float bank_dot(ggml_type type, bank_kind kind, const char * row, const block_q8_1 * x1,
        const block_q8_0 * x0, int64_t n, const void * xk = nullptr) {
    float s = 0.0f;
    switch (kind) {
        case BANK_GGML_Q8_1: ggml_get_type_traits_cpu(type)->vec_dot((int) n, &s, 0, row, 0, x1, 0, 1); return s;
        case BANK_GGML_Q8_0: ggml_get_type_traits_cpu(type)->vec_dot((int) n, &s, 0, row, 0, x0, 0, 1); return s;
        case BANK_GGML_Q8_K: ggml_get_type_traits_cpu(type)->vec_dot((int) n, &s, 0, row, 0, xk, 0, 1); return s;
        case BANK_CUSTOM:    return custom_dot_for(type)(row, x1, n);
        default:             return 0.0f;
    }
}

// Persistent workers that split rows. One job at a time; the caller participates. Workers spin briefly
// after a job before sleeping, so back-to-back jobs of one packet do not pay a wake each.
struct row_pool {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable wake;
    std::function<void(int64_t, int64_t)> job;
    int64_t n_rows = 0, chunk = 1;
    std::atomic<int64_t> next{0};
    std::atomic<int> pending{0};
    std::atomic<uint64_t> generation{0};
    std::atomic<bool> stop{false};

    explicit row_pool(int n) {
        for (int i = 1; i < n; ++i) {
            threads.emplace_back([this]() { loop(); });
        }
    }
    ~row_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        wake.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }
    void work() {
        for (int64_t begin; (begin = next.fetch_add(chunk)) < n_rows; ) {
            job(begin, std::min(n_rows, begin + chunk));
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            bool ready = false;
            for (int spin = 0; spin < 20000 && !ready; ++spin) {
                ready = stop.load(std::memory_order_acquire) || generation.load(std::memory_order_acquire) != seen;
                if (!ready) {
                    std::this_thread::yield();
                }
            }
            if (!ready) {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&]() { return stop.load() || generation.load() != seen; });
            }
            if (stop.load()) {
                return;
            }
            seen = generation.load(std::memory_order_acquire);
            work();
            pending.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    void run(int64_t rows, int64_t step, std::function<void(int64_t, int64_t)> fn) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            job = std::move(fn);
            n_rows = rows;
            chunk = step;
            next = 0;
            pending = (int) threads.size();
            generation.fetch_add(1, std::memory_order_acq_rel);
        }
        wake.notify_all();
        work();
        while (pending.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }
};

static std::atomic<int> & pool_threads() {
    static std::atomic<int> threads{0};
    return threads;
}

void llama_moe_hybrid_set_threads(int n_threads) {
    pool_threads().store(std::max(1, n_threads), std::memory_order_relaxed);
}

// Created on first use; a thread count set before that wins over the debug environment variable.
static row_pool & pool() {
    static row_pool instance([]() {
        const int configured = pool_threads().load(std::memory_order_relaxed);
        if (configured > 0) {
            return configured;
        }
        const char * value = ggml_moe_debug_getenv("LLAMA_MOE_HYBRID_CPU_THREADS");
        return value != nullptr ? std::max(1, atoi(value)) : 8;
    }());
    return instance;
}

// The CUDA quantize_q8_1 rule: d = amax/127 in fp32, q = round half away from zero, d stored as fp16.
void llama_moe_hybrid_quantize_q8_1(const float * x, void * vy, int64_t n) {
    auto * y = static_cast<block_q8_1 *>(vy);
    for (int64_t ib = 0; ib < n / QK8_1; ++ib) {
        const float * xb = x + ib*QK8_1;
        float amax = 0.0f, sum = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
            sum += xb[j];
        }
        const float d = amax / 127.0f;
        for (int j = 0; j < QK8_1; ++j) {
            y[ib].qs[j] = amax == 0.0f ? 0 : (int8_t) std::roundf(xb[j] / d);
        }
        y[ib].data.data.d = ggml_fp32_to_fp16(d);
        y[ib].data.data.s = ggml_fp32_to_fp16(sum);
    }
}

bool llama_moe_hybrid_q8_1_supported(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down) {
    return gate != nullptr && up != nullptr && down != nullptr &&
        kind_for(gate->type) != BANK_NONE && kind_for(up->type) != BANK_NONE && kind_for(down->type) != BANK_NONE &&
        gate->ne[0] % ggml_blck_size(gate->type) == 0 && up->ne[0] % ggml_blck_size(up->type) == 0 &&
        down->ne[0] % ggml_blck_size(down->type) == 0 && gate->ne[0] % QK8_1 == 0 && down->ne[0] % QK8_1 == 0;
}

// Q8_K banks quantize the fp32 activation on the CPU, so they need it.
static bool q8_k_input_ok(const ggml_tensor * gate, const ggml_tensor * up, const float * x_f32, size_t x_f32_stride) {
    const bool needed = kind_for(gate->type) == BANK_GGML_Q8_K || kind_for(up->type) == BANK_GGML_Q8_K;
    return !needed || (x_f32 != nullptr && x_f32_stride >= (size_t) gate->ne[0]);
}

// Expert e of each bank starts at base[bank][e]. Callers validate ids against their base tables.
static void q8_1_experts_core(
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down, const char * const * bases[3],
        const int32_t * experts, const int32_t * rows, uint32_t count, const void * x_q8_1, size_t x_row_bytes,
        float * out, size_t out_stride, bool gelu, const float * x_f32, size_t x_f32_stride) {
    int32_t max_row = 0;
    for (uint32_t i = 0; i < count; ++i) {
        max_row = std::max(max_row, rows != nullptr ? rows[i] : 0);
    }
    const char * const * gate_base = bases[0];
    const char * const * up_base   = bases[1];
    const char * const * down_base = bases[2];
    const int64_t n_embd = gate->ne[0];
    const int64_t n_ff   = gate->ne[1];
    const int64_t n_out  = down->ne[1];
    const bank_kind gate_kind = kind_for(gate->type);
    const bank_kind up_kind   = kind_for(up->type);
    const bank_kind down_kind = kind_for(down->type);
    const int64_t x_blocks  = n_embd / QK8_1;
    const int64_t ff_blocks = n_ff / QK8_1;

    // Routes ordered by expert: each group shares one weight read.
    thread_local std::vector<uint32_t> order, group_begin;
    order.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return experts[a] != experts[b] ? experts[a] < experts[b] : a < b;
    });
    group_begin.clear();
    for (uint32_t k = 0; k < count; ++k) {
        if (k == 0 || experts[order[k]] != experts[order[k - 1]]) {
            group_begin.push_back(k);
        }
    }
    group_begin.push_back(count);
    const int64_t n_groups = (int64_t) group_begin.size() - 1;
    // Plain pointers: the pool's threads would otherwise see their own, empty thread_local vectors.
    const uint32_t * order_p = order.data();
    const uint32_t * begin_p = group_begin.data();

    const auto x_row = [&](uint32_t route) {
        const int32_t row = rows != nullptr ? rows[route] : 0;
        return reinterpret_cast<const block_q8_1 *>(static_cast<const char *>(x_q8_1) + (size_t) row * x_row_bytes);
    };
    thread_local std::vector<block_q8_0> x0, act0;
    thread_local std::vector<float> act;
    thread_local std::vector<block_q8_1> act1;
    if (gate_kind == BANK_GGML_Q8_0 || up_kind == BANK_GGML_Q8_0) {
        x0.resize((size_t) (max_row + 1) * x_blocks);
        for (int32_t row = 0; row <= max_row; ++row) {
            repack_q8_0(reinterpret_cast<const block_q8_1 *>(static_cast<const char *>(x_q8_1) + (size_t) row * x_row_bytes),
                x0.data() + (size_t) row * x_blocks, x_blocks);
        }
    }
    thread_local std::vector<char> xk, actk;
    const auto q8_k_from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    const size_t xk_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, n_embd);
    if (gate_kind == BANK_GGML_Q8_K || up_kind == BANK_GGML_Q8_K) {
        xk.resize((size_t) (max_row + 1) * xk_row_bytes);
        for (int32_t row = 0; row <= max_row; ++row) {
            q8_k_from_float(x_f32 + (size_t) row * x_f32_stride, xk.data() + (size_t) row * xk_row_bytes, n_embd);
        }
    }
    act.resize((size_t) count * n_ff);
    act1.resize((size_t) count * ff_blocks);
    float * act_data = act.data();
    const block_q8_0 * x0_data = x0.data();
    const char * xk_data = xk.data();

    // Job = (group, weight row): the weight row comes from DRAM once, every route of the group reuses it.
    pool().run(n_groups * n_ff, 16, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const int64_t g = job / n_ff, row = job % n_ff;
            const int32_t expert = experts[order_p[begin_p[g]]];
            const char * g_row = gate_base[expert] + (size_t) row * gate->nb[1];
            const char * u_row = up_base[expert]   + (size_t) row * up->nb[1];
            for (uint32_t k = begin_p[g]; k < begin_p[g + 1]; ++k) {
                const uint32_t route = order_p[k];
                const int32_t r = rows != nullptr ? rows[route] : 0;
                const block_q8_0 * x0_row = x0_data != nullptr ? x0_data + (size_t) r * x_blocks : nullptr;
                const char * xk_row = xk_data != nullptr ? xk_data + (size_t) r * xk_row_bytes : nullptr;
                const float gv = bank_dot(gate->type, gate_kind, g_row, x_row(route), x0_row, n_embd, xk_row);
                const float uv = bank_dot(up->type, up_kind, u_row, x_row(route), x0_row, n_embd, xk_row);
                const float a = gelu ? 0.5f*gv*(1.0f + std::tanh(0.7978845608f*(gv + 0.044715f*gv*gv*gv)))
                                     : gv / (1.0f + std::exp(-gv));
                act_data[(size_t) route * n_ff + row] = a * uv;
            }
        }
    });
    for (uint32_t route = 0; route < count; ++route) {
        llama_moe_hybrid_quantize_q8_1(act.data() + (size_t) route * n_ff, act1.data() + (size_t) route * ff_blocks, n_ff);
    }
    if (down_kind == BANK_GGML_Q8_0) {
        act0.resize((size_t) count * ff_blocks);
        repack_q8_0(act1.data(), act0.data(), (int64_t) count * ff_blocks);
    }
    const size_t actk_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, n_ff);
    if (down_kind == BANK_GGML_Q8_K) {
        actk.resize((size_t) count * actk_row_bytes);
        for (uint32_t route = 0; route < count; ++route) {
            q8_k_from_float(act.data() + (size_t) route * n_ff, actk.data() + (size_t) route * actk_row_bytes, n_ff);
        }
    }

    const block_q8_1 * act1_data = act1.data();
    const block_q8_0 * act0_data = act0.data();
    const char * actk_data = actk.data();
    pool().run(n_groups * n_out, 32, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const int64_t g = job / n_out, row = job % n_out;
            const int32_t expert = experts[order_p[begin_p[g]]];
            const char * d_row = down_base[expert] + (size_t) row * down->nb[1];
            for (uint32_t k = begin_p[g]; k < begin_p[g + 1]; ++k) {
                const uint32_t route = order_p[k];
                out[(size_t) route * out_stride + row] = bank_dot(down->type, down_kind, d_row,
                    act1_data + (size_t) route * ff_blocks, act0_data + (size_t) route * ff_blocks, down->ne[0],
                    actk_data != nullptr ? actk_data + (size_t) route * actk_row_bytes : nullptr);
            }
        }
    });
}

bool llama_moe_hybrid_q8_1_experts(
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const int32_t * experts, const int32_t * rows, uint32_t count, const void * x_q8_1, size_t x_row_bytes,
        float * out, size_t out_stride, bool gelu, const float * x_f32, size_t x_f32_stride) {
    if (!llama_moe_hybrid_q8_1_supported(gate, up, down) || x_q8_1 == nullptr || out == nullptr ||
            experts == nullptr || count == 0 || (size_t) down->ne[1] > out_stride ||
            !q8_k_input_ok(gate, up, x_f32, x_f32_stride)) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (experts[i] < 0 || experts[i] >= gate->ne[2] || (rows != nullptr && rows[i] < 0)) {
            return false;
        }
    }
    thread_local std::vector<const char *> table[3];
    const ggml_tensor * banks[3] = {gate, up, down};
    const char * const * bases[3];
    for (int b = 0; b < 3; ++b) {
        table[b].resize(banks[b]->ne[2]);
        for (int64_t e = 0; e < banks[b]->ne[2]; ++e) {
            table[b][e] = static_cast<const char *>(banks[b]->data) + (size_t) e * banks[b]->nb[2];
        }
        bases[b] = table[b].data();
    }
    q8_1_experts_core(gate, up, down, bases, experts, rows, count, x_q8_1, x_row_bytes, out, out_stride, gelu,
        x_f32, x_f32_stride);
    return true;
}

bool llama_moe_hybrid_q8_1_store_experts(
        llama_moe_expert_store & store, uint32_t group,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const int32_t * experts, const int32_t * rows, uint32_t count, const void * x_q8_1, size_t x_row_bytes,
        float * out, size_t out_stride, bool gelu, const float * x_f32, size_t x_f32_stride) {
    if (!llama_moe_hybrid_q8_1_supported(gate, up, down) || !experts || !x_q8_1 || !out || !count || count > 64 ||
            !q8_k_input_ok(gate, up, x_f32, x_f32_stride) ||
            gate->ne[0] <= 0 || gate->ne[1] <= 0 || down->ne[1] <= 0 || gate->ne[0] != up->ne[0] ||
            gate->ne[1] != up->ne[1] || gate->ne[2] != up->ne[2] || gate->ne[2] != down->ne[2] ||
            gate->ne[1] != down->ne[0] || uint64_t(down->ne[1]) > out_stride ||
            out_stride > SIZE_MAX / sizeof(float) / count ||
            uint64_t(gate->ne[0] / QK8_1) > x_row_bytes / sizeof(block_q8_1)) {
        return false;
    }
    std::vector<int32_t> distinct(experts, experts + count);
    for (uint32_t i = 0; i < count; ++i) {
        if (experts[i] < 0 || experts[i] >= gate->ne[2] ||
                (rows && (rows[i] < 0 || rows[i] == INT32_MAX || uint64_t(rows[i]) >= SIZE_MAX / x_row_bytes))) {
            return false;
        }
    }
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    const ggml_tensor * banks[3] = {gate, up, down};
    const size_t n_out = size_t(down->ne[1]);
    std::vector<float> output(size_t(count) * n_out), chunk_out(size_t(count) * n_out);
    std::vector<llama_moe_expert_store::lease> leases;
    std::vector<const char *> table[3];
    std::vector<int32_t> local, local_rows;
    std::vector<uint32_t> routes;
    // As many bundles as staging allows are held at once; exhaustion flushes the chunk and retries.
    for (size_t next = 0; next < distinct.size();) {
        const size_t first = next;
        leases.clear();
        for (auto & t : table) {
            t.clear();
        }
        while (next < distinct.size()) {
            auto lease = store.acquire({group, uint32_t(distinct[next])});
            if (!lease) {
                if (next == first) {
                    return false;
                }
                break;
            }
            for (int b = 0; b < 3; ++b) {
                ggml_tensor view;
                if (!lease.tensor_view(b, *banks[b], view)) {
                    return false;
                }
                table[b].push_back(static_cast<const char *>(view.data));
            }
            leases.push_back(std::move(lease));
            ++next;
        }
        local.clear();
        local_rows.clear();
        routes.clear();
        for (uint32_t i = 0; i < count; ++i) {
            const auto it = std::lower_bound(distinct.begin() + first, distinct.begin() + next, experts[i]);
            if (it != distinct.begin() + next && *it == experts[i]) {
                local.push_back(int32_t(it - (distinct.begin() + first)));
                local_rows.push_back(rows ? rows[i] : 0);
                routes.push_back(i);
            }
        }
        const char * const * bases[3] = {table[0].data(), table[1].data(), table[2].data()};
        q8_1_experts_core(gate, up, down, bases, local.data(), local_rows.data(), uint32_t(local.size()),
            x_q8_1, x_row_bytes, chunk_out.data(), n_out, gelu, x_f32, x_f32_stride);
        for (size_t k = 0; k < routes.size(); ++k) {
            memcpy(output.data() + routes[k] * n_out, chunk_out.data() + k * n_out, n_out * sizeof(float));
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        memcpy(out + size_t(i) * out_stride, output.data() + size_t(i) * n_out, n_out * sizeof(float));
    }
    return true;
}
