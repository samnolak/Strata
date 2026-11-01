// The default CPU share is not armed when every expert is resident (#1595): no GPU, no model.
#include <cstdio>
#include <string>
#include <vector>

#include "strata/prefill/share_rules.hpp"

using strata::prefill::share_rules::default_share_has_work;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const long long pairs = 48 * 512;
    CHECK(default_share_has_work(pairs - 1, pairs));       // one expert streams: the share has something to take
    CHECK(default_share_has_work(1500, pairs));            // a 12 GB card: most of them stream
    CHECK(!default_share_has_work(pairs, pairs));          // every expert resident: nothing to take
    CHECK(!default_share_has_work(pairs + 100, pairs));    // a cache with spare slots
    CHECK(default_share_has_work(0, pairs));               // empty cache
    CHECK(default_share_has_work(100000, 0));              // total unknown: keep today's behaviour
    static_assert(!default_share_has_work(10, 10), "constexpr");

    // H11: memory verdicts
    using namespace strata::prefill::share_rules;
    const uint64_t G = 1ull << 30;
    ShareMemory m;
    m.buffer_bytes = G / 3;
    CHECK(default_share_memory_verdict(m).arm);                                   // nothing known: today's behaviour
    m.avail_ram = 20 * G; m.avail_commit = 30 * G;
    CHECK(default_share_memory_verdict(m).arm);                                   // plenty of RAM and commit
    m.avail_ram = 3 * G;
    CHECK(!default_share_memory_verdict(m).arm);                                  // RAM below buffers + margin
    CHECK(default_share_memory_verdict(m).why.find("RAM is free") != std::string::npos);
    m.avail_ram = kShareRamMargin + m.buffer_bytes;
    CHECK(default_share_memory_verdict(m).arm);                                   // exactly enough
    m.avail_ram = kShareRamMargin + m.buffer_bytes - 1;
    CHECK(!default_share_memory_verdict(m).arm);                                  // one byte short
    m.avail_ram = 20 * G; m.avail_commit = 3 * G;
    {
        const ShareVerdict v = default_share_memory_verdict(m);
        CHECK(!v.arm);                                                            // Windows commit nearly used up
        CHECK(v.why.find("commit") != std::string::npos);
    }
    m.avail_commit = kUnknown;
    m.inplace_bytes = 18 * G;                                                     // file pages the RAM cannot keep
    {
        const ShareVerdict v = default_share_memory_verdict(m);
        CHECK(!v.arm);
        CHECK(v.why.find("file pages") != std::string::npos);
    }
    m.inplace_bytes = 10 * G;                                                     // they fit with the margin
    CHECK(default_share_memory_verdict(m).arm);
    m.avail_ram = kUnknown; m.inplace_bytes = 500 * G;
    CHECK(default_share_memory_verdict(m).arm);                                   // RAM unknown: never blocks
    // STRATA_BATCH_CPU_SHARE: the prompt paths' pool with batch slots
    static_assert(prompt_paths_get_pool(0, false, false), "no batch: the pool, as before");
    static_assert(!prompt_paths_get_pool(8, false, false), "batch slots, variable unset: no pool (the old rule, byte-identical)");
    static_assert(prompt_paths_get_pool(8, true, false), "batch slots, STRATA_BATCH_CPU_SHARE=1: the pool");
    static_assert(!prompt_paths_get_pool(8, true, true), "--no-pool: never");
    static_assert(!prompt_paths_get_pool(0, false, true), "--no-pool: never");
    // STRATA_PREFILL_CPU_SHARE_TOKENS
    CHECK(share_token_cap(nullptr, 8) == 8);
    CHECK(share_token_cap("", 8) == 8);
    CHECK(share_token_cap("junk", 8) == 8);
    CHECK(share_token_cap("0", 8) == 8);
    CHECK(share_token_cap("-5", 8) == 8);
    CHECK(share_token_cap("24", 8) == 24);
    CHECK(share_token_cap("4", 8) == 4);
    CHECK(share_token_cap("100000", 8) == 512);
    // the jobs of one CPU expert: pieces of at most MAXT rows, in order, covering every row once
    for (int maxt : {1, 8, 16}) {
        for (int32_t n = 0; n <= 70; ++n) {
            std::vector<int> seen((size_t) n, 0);
            int32_t next = 0;
            bool ordered = true, sized = true;
            const int jobs = for_each_share_job(n, maxt, [&](int32_t r0, int32_t nt) {
                ordered &= r0 == next;
                sized &= nt >= 1 && nt <= maxt;
                for (int32_t r = r0; r < r0 + nt && r < n; ++r) ++seen[(size_t) r];
                next = r0 + nt;
            });
            CHECK(ordered);
            CHECK(sized);
            CHECK(next == n);
            CHECK(jobs == (n + maxt - 1) / maxt);
            for (int c : seen) CHECK(c == 1);
        }
    }
    {   // up to MAXT tokens: exactly the old single job
        int calls = 0;
        for_each_share_job(8, 8, [&](int32_t r0, int32_t nt) { CHECK(r0 == 0); CHECK(nt == 8); ++calls; });
        CHECK(calls == 1);
    }
    if (failed == 0) std::printf("share_rules_test: ok\n");
    return failed ? 1 : 0;
}
