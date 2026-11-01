// branch_align_test - STRATA_BRANCH_CHECKPOINT_ALIGN's arithmetic (CPU only, header-only code).
//
// A prompt segment's bits are set by the pieces the layer-major walk reads it in: a run of [x, y) is cut into pieces
// of its super-chunk s from x (each piece one MoE launch per layer over all its tokens, its sub-chunks of the chunk T
// from the piece's start), and a run under the walk's minimum is read chunk-major (pieces of T on another path).  Two
// reads compute the same bits when they read the same list of (start, length, path) pieces - the model below.  Checked:
//   1. lm_super_chunk (lm_plan.hpp) gives lm_prepare_chain's sizes (VRAM aside), incl. the sizes the prod log shows;
//   2. the root cause: a branch cut at an arbitrary token changes the pieces (upstream STRATA_BRANCH_CHECKPOINT);
//   3. an aligned cut (aligned_branch_boundary) with both parts pinned to the segment's super-chunk reads exactly the
//      pieces of the read without the cut - for every segment length, branch point and super-chunk target;
//   4. a later prompt (another segment end) that goes on from that checkpoint: whenever the guard accepts it
//      (aligned_resume_ok and the same grid as the checkpoint's), its pieces are those of its own read from the
//      anchor; and the guard does accept in the common case (the transcript grew a little);
//   5. read_segment_of picks the segment a read cuts around the branch point;
//   6. round 2 - the real functions the engine calls: read_turn_root against the inline code it replaced in generate.cpp
//      (random prompts, every from / ckpt / tail-role / root case), aligned_resume_check (the branch_resume_ok rule),
//      lm_plan_grid with the env knobs and the chunk the loan lays out (H2: a short lend's chunk, 2304, has no 24K grid),
//      and ConversationCache::best with the usable filter.
#include "strata/core/conversation_cache.hpp"
#include "strata/prefill/lm_plan.hpp"
#include "strata/program/slot_checkpoints.hpp"

#include <random>

#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {
using strata::prefill::lm_super_chunk;
using strata::program::aligned_branch_boundary;
using strata::program::aligned_resume_check;
using strata::program::aligned_resume_ok;
using strata::program::read_turn_root;
using strata::program::read_segment_of;

constexpr int64_t T = 8192;

struct Cfg {
    int64_t want, min_s;
    bool even;
};

using Piece = std::tuple<int64_t, int64_t, int>;   // start, length, path (1: the walk, 0: chunk-major)
using Plan = std::vector<Piece>;

void add_run(Plan& p, int64_t x, int64_t y, int64_t s) {   // s: the walk's super-chunk, 0: chunk-major
    if (y <= x) return;
    const int64_t step = s > 0 ? s : T;
    for (int64_t q = x; q < y; q += step) p.emplace_back(q, std::min(step, y - q), s > 0 ? 1 : 0);
}
int64_t unpinned_s(const Cfg& c, int64_t len) { return lm_super_chunk(len, c.want, T, c.min_s, c.even); }
int64_t pinned_s(int64_t pin, int64_t len) { return std::min<int64_t>(pin, (len + T - 1) / T * T); }
// the reference: the segment [a, e) in one run
Plan whole(const Cfg& c, int64_t a, int64_t e) {
    Plan p;
    add_run(p, a, e, unpinned_s(c, e - a));
    return p;
}
}  // namespace

int main() {
    int checks = 0, fails = 0;
    auto expect = [&](bool ok, const char* what, long long x = 0, long long y = 0, long long z = 0) {
        ++checks;
        if (!ok && ++fails <= 20) std::fprintf(stderr, "branch_align_test: %s (%lld %lld %lld)\n", what, x, y, z);
        return ok;
    };
    const Cfg dflt{32768, 2 * T, true};   // a CUDA layer split, nothing set (kLmDefaultOnTokens, even split on)
    // ---- 1. sizes
    expect(lm_super_chunk(16383, 32768, T, 2 * T, true) == 0, "under two chunks: chunk-major");
    expect(lm_super_chunk(16384, 32768, T, 2 * T, true) == 16384, "two chunks");
    expect(lm_super_chunk(27240, 32768, T, 2 * T, true) == 32768, "prod log: 27240 tokens in 1 super-chunk of 32768");
    expect(lm_super_chunk(32768, 32768, T, 2 * T, true) == 32768, "one whole super-chunk");
    expect(lm_super_chunk(40000, 32768, T, 2 * T, true) == 24576, "40K: 2 x 24K (even)");
    expect(lm_super_chunk(40000, 32768, T, 2 * T, false) == 32768, "40K, even off: 32K + 8K");
    expect(lm_super_chunk(49152, 32768, T, 2 * T, true) == 24576, "48K: 2 x 24K");
    expect(lm_super_chunk(49153, 32768, T, 2 * T, true) == 32768, "past 48K: 32K");
    expect(lm_super_chunk(70000, 32768, T, 2 * T, true) == 24576, "70K: 3 x 24K");
    expect(lm_super_chunk(73729, 32768, T, 2 * T, true) == 32768, "past 72K: 32K");
    expect(lm_super_chunk(200000, 32768, T, 2 * T, true) == 32768, "200K: 32K");
    expect(lm_super_chunk(8192, 8192, T, T, true) == 8192, "STRATA_PREFILL_LM_TOKENS=8192: one chunk is enough");
    expect(lm_super_chunk(0, 32768, T, 2 * T, true) == 0 && lm_super_chunk(5, 32768, 0, 1, true) == 0, "degenerate");
    // the brute-force form of lm_prepare_chain's choice (rounds aside: VRAM is no limit here)
    for (int64_t n = 1; n <= 400000; n += 97) {
        int64_t S = std::max<int64_t>(2 * T, 32768 / T * T);
        S = std::min<int64_t>(S, (n + T - 1) / T * T);
        int64_t want = n < 2 * T ? 0 : S;
        if (want > 0) {
            const int64_t k = (n + S - 1) / S, ev = ((n + k - 1) / k + T - 1) / T * T;
            if (k >= 2 && ev < S) want = ev;
        }
        expect(lm_super_chunk(n, 32768, T, 2 * T, true) == want, "lm_super_chunk vs lm_prepare_chain", n);
    }
    // ---- 2. the root cause: the upstream cut at the branch point itself
    {   // the BATCHING.md classifier transcript: 43,379 tokens from 0, branch at 43,340 (root/turn aside)
        Plan cut;
        add_run(cut, 0, 43340, unpinned_s(dflt, 43340));
        add_run(cut, 43340, 43379, unpinned_s(dflt, 39));
        expect(cut != whole(dflt, 0, 43379), "an unaligned cut reads other pieces");
        Plan rest;   // the next query from that checkpoint: 39 + 250 tokens chunk-major, not the tail of a super-chunk
        add_run(rest, 0, 43340, unpinned_s(dflt, 43340));
        add_run(rest, 43340, 43629, unpinned_s(dflt, 289));
        expect(rest != whole(dflt, 0, 43629), "a restore from it reads other pieces too");
    }
    int64_t n_cut = 0, n_resume = 0, n_resume_ok = 0;
    const Cfg cfgs[] = {dflt, {32768, 2 * T, false}, {16384, T, true}, {24576, T, true}, {65536, 2 * T, true},
                        {40960, T, true}};
    for (const Cfg& c : cfgs) {
        for (int64_t L = 1000; L <= 300000; L += 1733) {
            const int64_t G = unpinned_s(c, L);
            const Plan ref = whole(c, 0, L);
            for (int64_t b = 1; b < L; b += 977) {
                // ---- 3. the aligned cut
                const int64_t at = aligned_branch_boundary(b, 0, G, 0, L, 1024);
                if (at < 0) {
                    expect(G == 0 || b < G || b - b % G < 1024, "no aligned point only before the first grid point", L, b);
                    continue;
                }
                ++n_cut;
                expect(at % G == 0 && at <= b && b - at < G && at > 0 && at < L, "on the grid, before the branch", L, b, at);
                Plan x;
                add_run(x, 0, at, pinned_s(G, at));
                add_run(x, at, L, pinned_s(G, L - at));
                expect(x == ref, "the aligned cut reads the pieces of the read without it", L, b, G);
                // ---- 4. the next prompt: the same transcript, another end (grown by g, or cut back)
                for (const int64_t g : {-3000, 37, 250, 1300, 5500, 14300, 30000}) {
                    const int64_t LY = L + g;
                    if (LY <= at) continue;
                    ++n_resume;
                    const int64_t GY = unpinned_s(c, LY);
                    if (!(GY == G && aligned_resume_ok(at, 0, GY, LY))) continue;   // the engine reads from the anchor
                    ++n_resume_ok;
                    Plan y;
                    add_run(y, 0, at, pinned_s(G, at));   // the checkpoint's part, as the first prompt read it
                    add_run(y, at, LY, pinned_s(GY, LY - at));
                    expect(y == whole(c, 0, LY), "a resume the guard accepts reads the pieces of the whole read", L, LY, at);
                }
            }
        }
    }
    expect(n_cut > 1000 && n_resume_ok > n_resume / 3, "the aligned path is taken", n_cut, n_resume_ok, n_resume);
    // the guard's grid check: a transcript that grows past 48K from its anchor moves from 24K to 32K pieces
    expect(unpinned_s(dflt, 47000) == 24576 && unpinned_s(dflt, 50000) == 32768 &&
               !aligned_resume_ok(24576, 0, 32768, 50000),
           "24K grid point off the 32K grid");
    expect(aligned_resume_ok(32768, 0, 32768, 50000) && !aligned_resume_ok(32768, 0, 32768, 32768) &&
               !aligned_resume_ok(0, 0, 32768, 50000) && !aligned_resume_ok(32768, -1, 32768, 50000) &&
               !aligned_resume_ok(32768, 0, 0, 50000),
           "aligned_resume_ok bounds");
    // aligned_branch_boundary's rules
    expect(aligned_branch_boundary(43340, 0, 24576, 0, 43379, 1024) == 24576, "classifier 43K: back to 24576");
    expect(aligned_branch_boundary(43340, 2500, 24576, 0, 43379, 1024) == 27076, "from the root at 2500");
    expect(aligned_branch_boundary(20000, 0, 24576, 0, 43379, 1024) == -1, "before the first grid point");
    expect(aligned_branch_boundary(43340, 0, 24576, 24000, 43379, 1024) == -1, "within the margin of the read start");
    expect(aligned_branch_boundary(43340, 0, 24576, 0, 43379, 0) == -1, "margin 0: off");
    expect(aligned_branch_boundary(43340, 0, 0, 0, 43379, 1024) == -1, "grid 0 (chunk-major): none");
    expect(aligned_branch_boundary(43379, 0, 24576, 0, 43379, 1024) == -1, "at the segment end");
    expect(aligned_branch_boundary(60000, 24576, 32768, 24576, 70000, 1024) == 57344, "a read from an aligned checkpoint");
    // ---- 5. read_segment_of: [last cut at or before p, first cut after p)
    {
        auto s = read_segment_of(43340, 0, {2500, 43370}, 43378);
        expect(s.from == 2500 && s.end == 43370, "root before, turn after");
        s = read_segment_of(43340, 0, {-1, 43370}, 43378);
        expect(s.from == 0 && s.end == 43370, "no root");
        s = read_segment_of(43340, 0, {2500, 30000}, 43378);
        expect(s.from == 30000 && s.end == 43378, "the turn before the branch point: its own segment");
        s = read_segment_of(30000, 0, {2500, 30000}, 43378);
        expect(s.from == 30000 && aligned_branch_boundary(30000, s.from, 24576, 0, s.end, 1024) == -1,
               "a cut at the branch point: taken anyway, no aligned one");
        s = read_segment_of(43340, 24576, {-1, 43370}, 43378);
        expect(s.from == 24576 && s.end == 43370, "a read from an aligned checkpoint: its anchor");
        s = read_segment_of(43340, 30000, {2500, 43400}, 43378);
        expect(s.from == 30000 && s.end == 43378, "cuts outside (start, last) ignored");
    }
    // ---- 6a. read_turn_root == the inline code generate.cpp had (p55/final fea17c92, serve loop "turn_at" / "root_at")
    {
        auto old_code = [](const std::vector<int64_t>& ids, int64_t n, int64_t resume, int64_t read_from, bool resumed_from0,
                           int req_ckpt, int64_t prompt_cache, int64_t turn_token, int64_t tail_role_token,
                           int64_t prompt_cache_root) {
            int64_t turn_at = -1;
            if (prompt_cache > 0 && turn_token >= 0)
                for (int64_t i = n - 1; i > resume; --i)
                    if (ids[(size_t) i] == turn_token) { turn_at = i; break; }
            if (turn_at > 0 && tail_role_token >= 0)
                for (int64_t i = turn_at - 1; i > resume; --i)
                    if (ids[(size_t) i] == turn_token) {
                        if (ids[(size_t) i + 1] == tail_role_token) turn_at = i;
                        break;
                    }
            const int64_t last_turn = turn_at;
            if (!req_ckpt) turn_at = -1;
            int64_t root_at = -1;
            if (prompt_cache > 0 && turn_token >= 0 && prompt_cache_root > 0 && (read_from == 0 || resumed_from0))
                for (int64_t i = 1; i < last_turn; ++i)
                    if (ids[(size_t) i] == turn_token) {
                        if (i >= prompt_cache_root) root_at = i;
                        break;
                    }
            return std::make_pair(turn_at, root_at);
        };
        std::mt19937_64 rng(19);
        const int64_t TT = 248045, ROLE = 77;
        for (int it = 0; it < 20000; ++it) {
            const int64_t n = 2 + (int64_t) (rng() % 300);
            std::vector<int64_t> ids((size_t) n);
            for (auto& x : ids) {
                const uint64_t r = rng() % 100;
                x = r < 6 ? TT : r < 10 ? ROLE : (int64_t) (rng() % 1000);
            }
            const int64_t resume = (int64_t) (rng() % (uint64_t) n), read_from = rng() % 3 == 0 ? 0 : resume;
            const bool r0 = rng() % 4 == 0;
            const int ck = (int) (rng() % 4 != 0);
            const int64_t pc = rng() % 8 == 0 ? 0 : 12, tt = rng() % 16 == 0 ? -1 : TT;
            const int64_t role = rng() % 2 ? ROLE : -1, root_min = (int64_t) (rng() % 4) * 40;
            const auto want = old_code(ids, n, resume, read_from, r0, ck, pc, tt, role, root_min);
            const auto got = read_turn_root(ids.data(), n, resume, read_from == 0 || r0, ck != 0, pc > 0, tt, role, root_min);
            expect(got.turn == want.first && got.root == want.second, "read_turn_root == the code it replaced", it,
                   got.turn, want.first);
        }
    }
    // ---- 6b. aligned_resume_check: branch_resume_ok's rule
    expect(aligned_resume_check(27528, 2952, 24576, 48740, 48740, 48740, 24576), "on the grid, the same segment and grid");
    expect(!aligned_resume_check(27528, 2952, 24576, 52449, 52449, 52449, 32768), "the grid moved (24K -> 32K)");
    expect(!aligned_resume_check(27528, 2952, 24576, 20000, 48740, 48740, 24576), "a cut of the read from the anchor before it");
    expect(!aligned_resume_check(27528, 2952, 24576, 48740, 48740, 48000, 24576), "the read from it ends elsewhere");
    expect(!aligned_resume_check(27529, 2952, 24576, 48740, 48740, 48740, 24576), "off the grid");
    expect(!aligned_resume_check(27528, -1, 24576, 48740, 48740, 48740, 24576), "no anchor (a mark off the grid)");
    expect(!aligned_resume_check(27528, 2952, 0, 48740, 48740, 48740, 24576), "no grid kept");
    // ---- 6c. lm_plan_grid: the knobs and the loan's chunk (H2)
    {
        using strata::prefill::LmKnobs;
        using strata::prefill::lm_plan_grid;
        LmKnobs k;   // nothing set: the on-by-default 32K target, even split
        expect(lm_plan_grid(45788, 8192, k) == 24576 && lm_plan_grid(27240, 8192, k) == 32768, "defaults at 8192");
        expect(lm_plan_grid(45788, 2304, k) % 2304 == 0 && lm_plan_grid(45788, 2304, k) != 24576,
               "a short lend's chunk (2304) gives another grid: the loan must be the segment's");
        expect(lm_plan_grid(10000, 8192, k) == 0, "under two chunks: chunk-major");
        k.tokens = 16384;
        k.even = false;
        expect(lm_plan_grid(45788, 8192, k) == 16384 && lm_plan_grid(10000, 8192, k) == 16384 &&
                   lm_plan_grid(8191, 8192, k) == 0,
               "LM_TOKENS=16384 LM_EVEN=0: a fixed 16K grid from one chunk on");
        k = LmKnobs{};
        k.explicit_on = true;   // STRATA_PREFILL_LAYER_MAJOR=1: the 8192 target
        expect(lm_plan_grid(45788, 8192, k) == 16384, "LAYER_MAJOR=1: the 8192 target, two chunks minimum");
        k = LmKnobs{};
        k.force = true;
        expect(lm_plan_grid(5000, 8192, k) == 8192, "LM_FORCE: any length");
        expect(lm_plan_grid(45788, 0, LmKnobs{}) == 0, "no chunk: no grid");
    }
    // ---- 6d. ConversationCache::best with the usable filter (STRATA_BRANCH_CHECKPOINT_ALIGN)
    {
        using namespace strata::core;
        ConversationCache cache(1 << 20, 4);
        SavedConversation sv;
        sv.live.ids = {1, 2, 3, 4, 5, 6, 7, 8};
        sv.live.gdn.resize(64, 7);
        sv.cvec = true;
        ConversationCheckpoint c1, c2;
        c1.ids = {1, 2, 3};
        c1.gdn.resize(64, 1);
        c2.ids = {1, 2, 3, 4, 5};
        c2.gdn.resize(64, 2);
        c2.branch = true;
        c2.branch_anchor = 0;
        c2.branch_grid = 0;   // (an off-grid branch checkpoint)
        sv.checkpoints.push_back(c1);
        sv.checkpoints.push_back(c2);
        expect(cache.put(std::move(sv)), "park");
        const std::vector<int64_t> prompt = {1, 2, 3, 4, 5, 9, 9};
        expect(cache.best(prompt, {}, true).tokens == 5, "best: the branch checkpoint without a filter");
        const auto m = cache.best(prompt, {}, true, [](const ConversationCheckpoint& c) { return !c.branch || c.branch_grid > 0; });
        expect(m.tokens == 3 && !m.live, "best(usable): the filtered branch checkpoint left out");
        expect(cache.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 1}, {}, true,
                          [](const ConversationCheckpoint&) { return false; }).tokens == 8,
               "best(usable): the parked live state always counts");
    }
    if (fails) {
        std::fprintf(stderr, "branch_align_test: %d of %d checks failed\n", fails, checks);
        return 1;
    }
    std::printf("branch_align_test: %d checks passed (%lld aligned cuts, %lld of %lld later prompts go on from them)\n",
                checks, (long long) n_cut, (long long) n_resume_ok, (long long) n_resume);
    return 0;
}
