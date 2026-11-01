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
//   5. read_segment_of picks the segment a read cuts around the branch point.
#include "strata/prefill/lm_plan.hpp"
#include "strata/program/slot_checkpoints.hpp"

#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {
using strata::prefill::lm_super_chunk;
using strata::program::aligned_branch_boundary;
using strata::program::aligned_resume_ok;
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
    if (fails) {
        std::fprintf(stderr, "branch_align_test: %d of %d checks failed\n", fails, checks);
        return 1;
    }
    std::printf("branch_align_test: %d checks passed (%lld aligned cuts, %lld of %lld later prompts go on from them)\n",
                checks, (long long) n_cut, (long long) n_resume_ok, (long long) n_resume);
    return 0;
}
