// src/kernels/cpu/kq_mt_avx2.cpp - Unsloth UD-Q6_K_XL's expert rows (Q6_K gate/up x Q8_K, Q8_0 gate/up/down x Q8_0)
// for several tokens at once on AVX-2 (Zen 2/3, Intel 12th-14th gen), BIT-EXACT against ggml-cpu's own single-token dots.
// Opt-in: STRATA_CPU_AVX2_MT=1 (native_expert.cpp); default off keeps the per-token ggml path byte for byte.
//
// What is ported, and from where (both MIT, notices below):
//  * the loop structure is ik_llama.cpp's multi-column AVX2 GEMM (ggml/src/iqk/iqk_gemm_kquants.cpp:
//    mul_mat_qY_K_q8_K_T with DequantizerQ6K, and iqk_gemm_legacy_quants.cpp's mul_mat_qX_0_q8_0_T): a weight block is
//    unpacked ONCE into registers (DequantizerQ6K::prepare - the 6-bit values from ql/qh, the 16-bit block scales) and
//    then multiplied against all the columns (here: the tokens routed to the expert, up to 8 per pass), with one
//    integer accumulator per column kept in a register (`sumi[nrc_y]` / `accd[nrc_y]`), and one horizontal sum per
//    (row, column) at the very end;
//  * the arithmetic is NOT ik_llama's: ik folds Q6_K's -32 offset into a float accumulator (process_mins_and_scales_16)
//    and runs Q8_0 as unsigned Q8_0_1 against Q8_2_X4 activations, both of which round differently from ggml.  Every
//    integer lane and every float operation below is instead ggml-cpu's __AVX2__ branch of ggml-cpu/arch/x86/quants.c
//    (ggml_vec_dot_q6_K_q8_K: maddubs/madd with the 16-bit scales, the -32 offset as bsums.scales << 5 subtracted in
//    int32, d = y.d * fp16(x.d), one fmadd per block, hsum_float_8; ggml_vec_dot_q8_0_q8_0: sign/maddubs/madd,
//    d = fp16(x.d) * fp16(y.d), one fmadd per block, hsum_float_8), in ggml's order.  Integer lanes are exact (no
//    saturation: |q6| <= 63, |q8| <= 127 -> |maddubs pair| <= 16002), so each token's float sequence is ggml's to the
//    bit, whatever the number of tokens in the group and however rows are split across threads (kq_mt_avx2_parity
//    checks it at 1..16 tokens; native_expert.cpp re-checks once against the linked ggml before first use).
//
// The gain: the weight-side work (6-bit unpacking, scale shuffles, Q8_0 absolute values, fp16 scales) and the weight
// bytes are paid once per group of tokens instead of once per token, and the per-token work is 1 load + maddubs +
// madd + add per 32 weights.
//
// ---------------------------------------------------------------------------------------------------------------------
// MIT License (ik_llama.cpp)
// Copyright (c) 2023-2024 The ggml authors
// Copyright (c) 2023-2024 The llama.cpp authors
// Copyright (c) 2024-2025 The ik_llama.cpp authors (https://github.com/ikawrakow/ik_llama.cpp)
// MIT License (ggml / llama.cpp)
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ---------------------------------------------------------------------------------------------------------------------
#include "strata/kernels/cpu/kq_mt_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

constexpr int TMAX = 8;   // tokens per pass at most: one int32 accumulator per token stays in a register

// Tokens per pass: groups larger than `cap` are split into equal tiles (measured on Zen 3, one thread: Q6_K is fastest
// per token at 4..6 tokens a pass - 8 spills the accumulators - and Q8_0 at 4, its two-row kernel), e.g. Q6_K 8 -> 4 + 4,
// 12 -> 6 + 6, Q8_0 6 -> 3 + 3.  Every token's bits are the same in any tiling.
inline int tile_len(int nt, int t0, int cap) {
    const int left = nt - t0, tiles = (left + cap - 1) / cap;
    return (left + tiles - 1) / tiles;
}
constexpr int kQ6Cap = 6, kQ8Cap = 4;

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
alignas(16) const uint8_t k_shuffle_q6[128] = {
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15};
inline __m128i get_scale_shuffle(int i) { return _mm_load_si128((const __m128i*) k_shuffle_q6 + i); }

// ---- Q6_K x Q8_K: ggml_vec_dot_q6_K_q8_K (__AVX2__) with the token loop inside the block loop.
// One weight row, NT tokens: res[t] = row . y[t], each ggml's bits.
template <int NT>
inline void q6k_row(const block_q6_K* x, int nb, const block_q8_K* const* y, float* res) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 128; ++j) {
            // DequantizerQ6K::prepare (ggml's form of the same bits): the four 32-byte groups of 6-bit values
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) (q4 + 64 * j));
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) (q4 + 64 * j + 32));
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) (qh + 32 * j));
            __m256i q6[4];
            q6[0] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4));
            q6[1] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15),
                                    _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2));
            q6[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15),
                                    _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48)));
            q6[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15),
                                    _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2));
            for (int k = 0; k < 4; ++k) {
                const __m256i sc = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(4 * j + k)));
                for (int t = 0; t < NT; ++t) {
                    const __m256i q8 = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 128 * j + 32 * k));
                    const __m256i p16 = _mm256_madd_epi16(sc, _mm256_maddubs_epi16(q6[k], q8));
                    sumi[t] = _mm256_add_epi32(sumi[t], p16);
                }
            }
        }
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        for (int t = 0; t < NT; ++t) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            const __m256i s = _mm256_sub_epi32(sumi[t], q8sclsub);
            const float d = y[t][i].d * xd;
            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(s), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

// ---- Q8_0 x Q8_0: ggml_vec_dot_q8_0_q8_0 (__AVX2__, mul_sum_i8_pairs_float without AVX-VNNI-INT8), the token loop
// inside the block loop; the activation scales arrive pre-converted (yd[t * nb + ib], exact fp16 -> fp32), and the
// block scale is the lane-wise product of two broadcasts (the same IEEE product as ggml's scalar one).
template <int NT>
inline void q8_0_row(const block_q8_0* x, int nb, const block_q8_0* const* y, const float* yd, float* res) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 vxd = _mm256_set1_ps(h2f(x[ib].d));
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < NT; ++t) {
            const __m256 d = _mm256_mul_ps(vxd, _mm256_broadcast_ss(yd + (size_t) t * nb + ib));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax, sy)));
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

// Two rows at once (shares each token's activation load and scale broadcast between the rows); NT <= 4 keeps the
// 2 x NT accumulators, both rows' weights and the token's vectors in the 16 ymm registers.
template <int NT>
inline void q8_0_row2(const block_q8_0* x0, const block_q8_0* x1, int nb, const block_q8_0* const* y, const float* yd,
                      float* res0, float* res1) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc0[NT], acc1[NT];
    for (int t = 0; t < NT; ++t) { acc0[t] = _mm256_setzero_ps(); acc1[t] = _mm256_setzero_ps(); }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 vxd0 = _mm256_set1_ps(h2f(x0[ib].d));
        const __m256 vxd1 = _mm256_set1_ps(h2f(x1[ib].d));
        const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x0[ib].qs);
        const __m256i qx1 = _mm256_loadu_si256((const __m256i*) x1[ib].qs);
        const __m256i ax0 = _mm256_sign_epi8(qx0, qx0);
        const __m256i ax1 = _mm256_sign_epi8(qx1, qx1);
        for (int t = 0; t < NT; ++t) {
            const __m256 vyd = _mm256_broadcast_ss(yd + (size_t) t * nb + ib);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            const __m256 q0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax0, _mm256_sign_epi8(qy, qx0))));
            const __m256 q1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax1, _mm256_sign_epi8(qy, qx1))));
            acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(vxd0, vyd), q0, acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(vxd1, vyd), q1, acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { res0[t] = hsum_float_8(acc0[t]); res1[t] = hsum_float_8(acc1[t]); }
}

void q6k_row_n(const block_q6_K* x, int nb, const block_q8_K* const* y, int m, float* res) {
    switch (m) {
        case 1: q6k_row<1>(x, nb, y, res); break;
        case 2: q6k_row<2>(x, nb, y, res); break;
        case 3: q6k_row<3>(x, nb, y, res); break;
        case 4: q6k_row<4>(x, nb, y, res); break;
        case 5: q6k_row<5>(x, nb, y, res); break;
        case 6: q6k_row<6>(x, nb, y, res); break;
        case 7: q6k_row<7>(x, nb, y, res); break;
        default: q6k_row<8>(x, nb, y, res); break;
    }
}

void q8_0_row_n(const block_q8_0* x, int nb, const block_q8_0* const* y, const float* yd, int m, float* res) {
    switch (m) {
        case 1: q8_0_row<1>(x, nb, y, yd, res); break;
        case 2: q8_0_row<2>(x, nb, y, yd, res); break;
        case 3: q8_0_row<3>(x, nb, y, yd, res); break;
        case 4: q8_0_row<4>(x, nb, y, yd, res); break;
        case 5: q8_0_row<5>(x, nb, y, yd, res); break;
        case 6: q8_0_row<6>(x, nb, y, yd, res); break;
        case 7: q8_0_row<7>(x, nb, y, yd, res); break;
        default: q8_0_row<8>(x, nb, y, yd, res); break;
    }
}

// rows [r0, r1) of Q8_0 against m <= TMAX tokens: two rows at a time for 2..4 tokens (one row otherwise and for an odd tail)
// (out[t0 + t][r - ooff])
void q8_0_rows_tile(const uint8_t* w, size_t row_bytes, int nb, const block_q8_0* const* y, const float* yd, int m,
                    float* const* out, int t0, int r0, int r1, int ooff = 0) {
    float s0[TMAX], s1[TMAX];
    int r = r0;
    // two rows from two tokens on (measured: at one token the pair is ~7% slower from DRAM, 1.3-1.8x faster from 2..4)
    if (m >= 2 && m <= 4) {
        for (; r + 2 <= r1; r += 2) {
            const block_q8_0* x0 = (const block_q8_0*) (w + (size_t) r * row_bytes);
            const block_q8_0* x1 = (const block_q8_0*) (w + (size_t) (r + 1) * row_bytes);
            switch (m) {
                case 1: q8_0_row2<1>(x0, x1, nb, y, yd, s0, s1); break;
                case 2: q8_0_row2<2>(x0, x1, nb, y, yd, s0, s1); break;
                case 3: q8_0_row2<3>(x0, x1, nb, y, yd, s0, s1); break;
                default: q8_0_row2<4>(x0, x1, nb, y, yd, s0, s1); break;
            }
            for (int t = 0; t < m; ++t) { out[t0 + t][r - ooff] = s0[t]; out[t0 + t][r + 1 - ooff] = s1[t]; }
        }
    }
    for (; r < r1; ++r) {
        q8_0_row_n((const block_q8_0*) (w + (size_t) r * row_bytes), nb, y, yd, m, s0);
        for (int t = 0; t < m; ++t) out[t0 + t][r - ooff] = s0[t];
    }
}

// the activation scales of m tokens, fp32 (exact): yd[t * nb + ib], in a per-thread buffer
const float* q8_0_scales(const block_q8_0* const* y, int m, int nb) {
    thread_local std::vector<float> buf;
    if (buf.size() < (size_t) TMAX * (size_t) nb) buf.resize((size_t) TMAX * (size_t) nb);
    float* yd = buf.data();
    for (int t = 0; t < m; ++t)
        for (int ib = 0; ib < nb; ++ib) yd[(size_t) t * nb + ib] = h2f(y[t][ib].d);
    return yd;
}

}  // namespace

void kqmt_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
               int r0, int r1) {
    if (type == 14) {
        const int nb = n / QK_K;
        for (int t0 = 0, m = 0; t0 < nt; t0 += m) {
            m = tile_len(nt, t0, kQ6Cap);
            const block_q8_K* y[TMAX];
            for (int t = 0; t < m; ++t) y[t] = (const block_q8_K*) act[t0 + t];
            float s[TMAX];
            for (int r = r0; r < r1; ++r) {
                q6k_row_n((const block_q6_K*) (w + (size_t) r * row_bytes), nb, y, m, s);
                for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
            }
        }
        return;
    }
    if (type == 8) {
        const int nb = n / QK8_0;
        for (int t0 = 0, m = 0; t0 < nt; t0 += m) {
            m = tile_len(nt, t0, kQ8Cap);
            const block_q8_0* y[TMAX];
            for (int t = 0; t < m; ++t) y[t] = (const block_q8_0*) act[t0 + t];
            const float* yd = q8_0_scales(y, m, nb);
            q8_0_rows_tile(w, row_bytes, nb, y, yd, m, out, t0, r0, r1);
        }
    }
}

void kqmt_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                  float* const* ff, int r0, int r1) {
    if (type == 14) {
        const int nb = n / QK_K;
        for (int t0 = 0, m = 0; t0 < nt; t0 += m) {
            m = tile_len(nt, t0, kQ6Cap);
            const block_q8_K* y[TMAX];
            for (int t = 0; t < m; ++t) y[t] = (const block_q8_K*) act[t0 + t];
            float g[TMAX], u[TMAX];
            for (int r = r0; r < r1; ++r) {
                q6k_row_n((const block_q6_K*) (blob + (size_t) r * gu_row), nb, y, m, g);
                q6k_row_n((const block_q6_K*) (blob + up_off + (size_t) r * gu_row), nb, y, m, u);
                // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
                for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
            }
        }
        return;
    }
    if (type == 8) {
        const int nb = n / QK8_0;
        constexpr int RC = 32;   // rows per chunk: gate and up results of a chunk stay on the stack
        for (int t0 = 0, m = 0; t0 < nt; t0 += m) {
            m = tile_len(nt, t0, kQ8Cap);
            const block_q8_0* y[TMAX];
            for (int t = 0; t < m; ++t) y[t] = (const block_q8_0*) act[t0 + t];
            const float* yd = q8_0_scales(y, m, nb);
            alignas(32) float gb[TMAX][RC], ub[TMAX][RC];
            for (int c0 = r0; c0 < r1; c0 += RC) {
                const int c1 = c0 + RC < r1 ? c0 + RC : r1;
                float* gp[TMAX];
                float* up[TMAX];
                for (int t = 0; t < m; ++t) { gp[t] = gb[t]; up[t] = ub[t]; }
                q8_0_rows_tile(blob, gu_row, nb, y, yd, m, gp, 0, c0, c1, c0);
                q8_0_rows_tile(blob + up_off, gu_row, nb, y, yd, m, up, 0, c0, c1, c0);
                for (int t = 0; t < m; ++t)
                    for (int r = c0; r < c1; ++r) {
                        const float gv = gb[t][r - c0], uv = ub[t][r - c0];
                        ff[t0 + t][r] = (gv / (1.f + std::exp(-gv))) * uv;
                    }
            }
        }
    }
}

}  // namespace strata::kernels::cpu
