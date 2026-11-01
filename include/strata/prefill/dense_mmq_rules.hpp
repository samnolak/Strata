// include/strata/prefill/dense_mmq_rules.hpp - STRATA_DENSE_MMQ (opt-in): which dense GGUF prompt projections
// Gemm::native runs through llama.cpp's int8 MMQ instead of dequantize-to-FP16 + cuBLAS, and in how many row passes.
// Header-only, no GPU: the rules are checked by tests/prefill/dense_mmq_rules_test.cpp.
//
// Why on CUDA too (E22): a GeForce Ada card multiplies FP16 with FP32 accumulation at half the FP16 rate (RTX 4080:
// 97.5 TFLOPS) but int8 at 389.9 TOPS, and the FP16 path dequantizes every weight again for every chunk.  The
// UD-Q6_K_XL packs keep their dense projections in Q8_0 (attn_qkv / attn_gate / ssm_out, attn_q / k / v / output,
// the shared expert's gate / up / down): ~2.9 G weights, ~55 % of a prompt token's GEMM work, while the routed
// experts already run on MMQ.  The activations are rounded to q8_1 (W8A8, as the routed experts and the decode's
// MMVQ are): not bit-identical to the FP16 path, so it stays opt-in behind a KL gate.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::prefill::dense_mmq {

/// Rows quantized per pass (the q8_1 buffer is laid out for kRows x kMaxK once, at the first product).
constexpr int64_t kRows = 1024;
/// The widest K the q8_1 buffer holds.
constexpr int64_t kMaxK = 8192;

/// STRATA_DENSE_MMQ: on for a text starting with '1' (the HIP switch's own parse: unset, empty, 0 or anything else: off).
inline bool enabled_from(const char* v) { return v != nullptr && v[0] == '1'; }

/// STRATA_DENSE_MMQ_MIN_T=N: only products of at least N rows take MMQ (unset, empty, bad or <= 0: every T).
inline int64_t min_t_from(const char* v) {
    if (v == nullptr || v[0] == 0) return 0;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    return (end != v && n > 0) ? (int64_t) n : 0;
}

/// The shape test: llama.cpp's MMQ loads the weights in 256-value K chunks, and the chunk past a partial row reads past
/// the row (the next row's bytes, or for the last row bytes past the tensor): a dense tensor is passed as-is, so K must
/// be a whole number of chunks (the 640-value shared-expert down stays on cuBLAS), and at most kMaxK.  The weight's
/// tile fit (mmq::fits) is the caller's other test.
constexpr bool shape_ok(int64_t N, int64_t K) { return N > 0 && K > 0 && K <= kMaxK && K % 256 == 0; }

/// Rows per pass: FP32 activations of R rows of K values live in the FP16 dequant scratch (scratch_elems halves), capped
/// at kRows and at T.  0: the scratch cannot hold one row (the caller keeps the FP16 path).
constexpr int64_t rows_per_pass(int64_t scratch_elems, int64_t K, int64_t T) {
    if (K <= 0 || T <= 0 || scratch_elems <= 0) return 0;
    int64_t r = scratch_elems / (2 * K);
    if (r > kRows) r = kRows;
    if (r > T) r = T;
    return r < 1 ? 0 : r;
}

/// The number of MMQ passes a product of T rows takes (0 when rows_per_pass is 0).
constexpr int64_t passes(int64_t scratch_elems, int64_t K, int64_t T) {
    const int64_t r = rows_per_pass(scratch_elems, K, T);
    return r > 0 ? (T + r - 1) / r : 0;
}

}  // namespace strata::prefill::dense_mmq
