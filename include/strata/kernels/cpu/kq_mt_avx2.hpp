// include/strata/kernels/cpu/kq_mt_avx2.hpp - multi-token AVX-2 expert rows for Unsloth UD-Q6_K_XL's expert formats
// (Q6_K gate/up against Q8_K activations; Q8_0 gate/up/down against Q8_0), BIT-EXACT against ggml-cpu's AVX2
// single-token vec_dot for every token, whatever the number of tokens.  Opt-in: STRATA_CPU_AVX2_MT=1 (native_expert.cpp).
// See src/kernels/cpu/kq_mt_avx2.cpp (loop structure after ik_llama.cpp's iqk_mul_mat, MIT).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Q6_K (14) gate/up, Q8_0 (8) gate/up and down.
inline bool kqmt_supported(int ggml_type) noexcept { return ggml_type == 14 || ggml_type == 8; }

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
/// Any nt >= 1 (tokens are taken eight at a time).  The caller checks cpu_avx2_ok().
void kqmt_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                  int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void kqmt_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
               float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
