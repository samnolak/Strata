// include/strata/prefill/lm_plan.hpp - the super-chunk the layer-major walk reads a segment in (header-only: arithmetic).
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

/// The super-chunk Prefill::lm_prepare_chain gives an n-token read when VRAM is no limit: 0 below `min_s` tokens (the
/// chunk-major path reads it), else the target `want` on the chunk grid (at least min_s), at most n rounded up to whole
/// chunks; with `even` (STRATA_PREFILL_LM_EVEN, default on) n is then cut into the same number of near-equal pieces on
/// the chunk grid when those are smaller (40K at a 32K target: 2 x 24K, not 32K + 8K).  The walk reads the segment in
/// pieces of this size from its first token, and each piece in chunks from the piece's start - so the size changes
/// with the segment's length, and with it every piece's bits.
inline int64_t lm_super_chunk(int64_t n, int64_t want, int64_t chunk, int64_t min_s, bool even) {
    if (n <= 0 || chunk <= 0 || n < min_s) return 0;
    int64_t s = std::max<int64_t>(min_s, want / chunk * chunk);
    s = std::min<int64_t>(s, (n + chunk - 1) / chunk * chunk);
    if (s <= 0) return 0;
    const int64_t k = (n + s - 1) / s;
    const int64_t ev = ((n + k - 1) / k + chunk - 1) / chunk * chunk;
    return (even && k >= 2 && ev < s) ? ev : s;
}

/// The walk's knobs as lm_prepare reads them (prefill.cpp): STRATA_PREFILL_LM_TOKENS (`tokens`, 0: unset),
/// STRATA_PREFILL_LM_FORCE (`force`), whether STRATA_PREFILL_LAYER_MAJOR was set (`explicit_on`: the 8192 target, else
/// the on-by-default 32768) and STRATA_PREFILL_LM_EVEN (`even`).
struct LmKnobs {
    int64_t tokens = 0;
    bool force = false, explicit_on = false, even = true;
};
inline constexpr int64_t kLmPlanDefaultTokens = 8192;     // = prefill.cpp kLmDefaultTokens
inline constexpr int64_t kLmPlanDefaultOnTokens = 32768;  // = prefill.cpp kLmDefaultOnTokens
/// STRATA_BRANCH_CHECKPOINT_ALIGN: the super-chunk an n-token run takes in chunks of `chunk` tokens (the chunk the
/// run's loan lays the buffers out for - Prefill::chunk() after lend(), not before it), VRAM aside.
inline int64_t lm_plan_grid(int64_t n, int64_t chunk, const LmKnobs& k) {
    if (chunk <= 0) return 0;
    const int64_t min_s = k.force ? 1 : (k.tokens > 0 ? chunk : 2 * chunk);
    const int64_t want = k.tokens > 0 ? k.tokens : (!k.explicit_on ? kLmPlanDefaultOnTokens : kLmPlanDefaultTokens);
    return lm_super_chunk(n, want, chunk, min_s, k.even);
}

}  // namespace strata::prefill
