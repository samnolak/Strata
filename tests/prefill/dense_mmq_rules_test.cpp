// STRATA_DENSE_MMQ's rules (include/strata/prefill/dense_mmq_rules.hpp): no GPU, no model.
#include <cstdio>

#include "strata/prefill/dense_mmq_rules.hpp"

namespace dm = strata::prefill::dense_mmq;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: the HIP parse ('1' first), unchanged
    CHECK(!dm::enabled_from(nullptr));
    CHECK(!dm::enabled_from(""));
    CHECK(!dm::enabled_from("0"));
    CHECK(!dm::enabled_from("yes"));
    CHECK(dm::enabled_from("1"));
    CHECK(dm::enabled_from("1x"));
    // the row floor
    CHECK(dm::min_t_from(nullptr) == 0);
    CHECK(dm::min_t_from("") == 0);
    CHECK(dm::min_t_from("abc") == 0);
    CHECK(dm::min_t_from("-5") == 0);
    CHECK(dm::min_t_from("0") == 0);
    CHECK(dm::min_t_from("64") == 64);
    CHECK(dm::min_t_from("1024rows") == 1024);

    // Qwen3.8-Flash-Next UD-Q6_K_XL's Q8_0 dense projections (pack index: N rows x K columns), hidden 2560
    struct Shape { long long N, K; bool mmq; const char* name; };
    const Shape shapes[] = {
        {10240, 2560, true, "attn_qkv (GDN in)"},
        {6144, 2560, true, "attn_gate (GDN z)"},
        {2560, 6144, true, "ssm_out"},
        {12288, 2560, true, "attn_q"},
        {512, 2560, true, "attn_k"},
        {512, 2560, true, "attn_v"},
        {2560, 6144, true, "attn_output"},
        {640, 2560, true, "ffn_gate_shexp / ffn_up_shexp"},
        {2560, 640, false, "ffn_down_shexp (K 640: 20 blocks, not a 256 chunk multiple)"},
        {10240, 2560, true, "ple_key"},
    };
    for (const Shape& s : shapes) {
        if (dm::shape_ok(s.N, s.K) != s.mmq) {
            std::fprintf(stderr, "FAIL shape %s: N %lld K %lld -> %d\n", s.name, s.N, s.K, (int) dm::shape_ok(s.N, s.K));
            ++failed;
        }
    }
    CHECK(!dm::shape_ok(2560, 8192 + 256));   // wider than the q8_1 buffer
    CHECK(dm::shape_ok(2560, 8192));
    CHECK(!dm::shape_ok(0, 2560));
    CHECK(!dm::shape_ok(2560, 0));
    CHECK(!dm::shape_ok(2560, 2560 + 32));    // a partial 256-value chunk

    // the row passes: FP32 rows of K in a scratch of S FP16 elements, capped at kRows and T
    CHECK(dm::rows_per_pass(0, 2560, 100) == 0);
    CHECK(dm::rows_per_pass(2 * 2560 - 1, 2560, 100) == 0);       // not one FP32 row
    CHECK(dm::rows_per_pass(2 * 2560, 2560, 100) == 1);
    CHECK(dm::rows_per_pass(1ll << 26, 2560, 100) == 100);         // capped at T
    CHECK(dm::rows_per_pass(1ll << 26, 2560, 5000) == dm::kRows);  // capped at kRows
    CHECK(dm::rows_per_pass(2 * 6144 * 300, 6144, 5000) == 300);   // the scratch's own limit
    CHECK(dm::passes(1ll << 26, 2560, 1254) == 2);                 // a swarm delta: 1024 + 230 rows
    CHECK(dm::passes(1ll << 26, 2560, 1024) == 1);
    CHECK(dm::passes(1ll << 26, 2560, 27240) == 27);               // a subagent's first read (layer-major super-chunk)
    CHECK(dm::passes(2 * 6144 * 300, 6144, 1254) == 5);
    CHECK(dm::passes(0, 2560, 1254) == 0);
    static_assert(dm::shape_ok(10240, 2560) && !dm::shape_ok(2560, 640), "constexpr");
    static_assert(dm::rows_per_pass(1ll << 26, 2560, 1254) == 1024, "constexpr");

    if (failed) {
        std::fprintf(stderr, "dense_mmq_rules_test: %d failure(s)\n", failed);
        return 1;
    }
    std::printf("dense_mmq_rules_test: PASS\n");
    return 0;
}
