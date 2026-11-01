#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace strata::program {

// STRATA_SLOT_CHECKPOINTS: which of the main session's checkpoints a batch slot keeps with the conversation it takes
// (BADM) or keeps (park=B): the deepest `keep` that are a strict start of `held`, shallowest first, one per length
// (the first of equal ones, as the single deepest one was picked before).
template <class Checkpoint>
std::vector<const Checkpoint*> deepest_prefix_checkpoints(const std::vector<Checkpoint>& checks,
                                                         const std::vector<int32_t>& held, std::size_t keep) {
    std::vector<const Checkpoint*> out;
    for (const Checkpoint& c : checks)
        if (c.ids.size() < held.size() && std::equal(c.ids.begin(), c.ids.end(), held.begin())) out.push_back(&c);
    std::stable_sort(out.begin(), out.end(),
                     [](const Checkpoint* a, const Checkpoint* b) { return a->ids.size() < b->ids.size(); });
    out.erase(std::unique(out.begin(), out.end(),
                          [](const Checkpoint* a, const Checkpoint* b) { return a->ids.size() == b->ids.size(); }),
              out.end());
    if (out.size() > keep) out.erase(out.begin(), out.end() - (std::ptrdiff_t) keep);
    return out;
}

// The ones a slot keeps of the strict starts of `held` (one per length), shallowest first: the deepest (the turn
// checkpoint); with `branch` (STRATA_BRANCH_CHECKPOINT) the deepest `branch` one - where a client that rewrites the end
// of its history leaves it next, also when a periodic checkpoint lies between it and the turn - whatever `keep` is;
// then the deepest others up to `keep` (STRATA_SLOT_CHECKPOINTS) in all.  So keep 1: the turn and the branch one when
// there is one; keep 2: two, the second the branch one or else the next deepest (a periodic one).
template <class Checkpoint>
std::vector<const Checkpoint*> slot_checkpoint_picks(const std::vector<Checkpoint>& checks,
                                                     const std::vector<int32_t>& held, std::size_t keep, bool branch) {
    const std::vector<const Checkpoint*> all = deepest_prefix_checkpoints(checks, held, checks.size());
    std::vector<char> take(all.size(), 0);
    std::size_t n = 0;
    if (keep > 0 && !all.empty()) {
        take.back() = 1;
        ++n;
    }
    if (branch)
        for (std::size_t i = all.size() - (n > 0 ? 1 : 0); i-- > 0;)
            if (all[i]->branch) {
                take[i] = 1;
                ++n;
                break;
            }
    for (std::size_t i = all.size(); n < keep && i-- > 0;)
        if (!take[i]) {
            take[i] = 1;
            ++n;
        }
    std::vector<const Checkpoint*> out;
    for (std::size_t i = 0; i < all.size(); ++i)
        if (take[i]) out.push_back(all[i]);
    return out;
}

// How many leading tokens a prompt and a held conversation share.
inline int64_t shared_prefix(const std::vector<int64_t>& prompt, const std::vector<int32_t>& held) {
    const std::size_t n = std::min(prompt.size(), held.size());
    std::size_t i = 0;
    while (i < n && prompt[i] == (int64_t) held[i]) ++i;
    return (int64_t) i;
}

// STRATA_BRANCH_CHECKPOINT=<margin>: the read stops for a checkpoint at `shared`, where the prompt leaves a
// conversation the engine holds (the client rewrote what came after it), so the next prompt that branches from this
// one later still finds a state to continue from.  -1 for none: only `margin` tokens or more past where the read
// starts, and before the turn boundary (whose checkpoint is taken anyway) or, without one, the last prompt token.
inline int64_t branch_checkpoint_boundary(int64_t shared, int64_t read_from, int64_t turn_at, int64_t n,
                                          int64_t margin) {
    const int64_t end = turn_at > 0 ? turn_at : n - 1;
    if (margin <= 0 || shared <= 0 || shared - read_from < margin || shared >= end) return -1;
    return shared;
}

// STRATA_BRANCH_CHECKPOINT_ALIGN=1: the branch point moved back onto the grid a read of its segment cuts at anyway.  The
// layer-major walk reads the segment [anchor, seg_end) - between two of the read's own cuts (where it starts, the root,
// the turn boundary, the last prompt token) - in super-chunks of `grid` tokens from `anchor`, so a stop at anchor +
// k grid ends a super-chunk the read ends there anyway: with every part of the segment read in that super-chunk
// (Prefill::pin_super_chunk) the read with the stop computes the bits of the read without it, and a read that goes on
// from the checkpoint those of the read from the anchor.  -1 for none: k >= 1, at least `margin` past `read_from`
// (where this read starts; a checkpoint at it is the one restored) and before `seg_end`.
inline int64_t aligned_branch_boundary(int64_t shared, int64_t anchor, int64_t grid, int64_t read_from, int64_t seg_end,
                                       int64_t margin) {
    if (margin <= 0 || grid <= 0 || anchor < 0 || shared <= anchor || shared >= seg_end) return -1;
    const int64_t at = anchor + (shared - anchor) / grid * grid;
    if (at <= anchor || at - read_from < margin || at >= seg_end) return -1;
    return at;
}

// A read that goes on from an aligned branch checkpoint at `at` (taken on the grid of the read from `anchor`) computes
// the bits of the read from the anchor only where `at` lies on the grid of THAT read: `grid` is the super-chunk the read
// of this prompt's segment [anchor, seg_end) takes, and it changes with the segment's length (the even split: a
// transcript that grows past 48K from its anchor goes from 24K to 32K super-chunks).
inline bool aligned_resume_ok(int64_t at, int64_t anchor, int64_t grid, int64_t seg_end) {
    return anchor >= 0 && grid > 0 && at > anchor && at < seg_end && (at - anchor) % grid == 0;
}

// The read's own cuts in a prompt `ids` of n tokens read on from `from` (generate.cpp's serve loop; one function for the
// read and for STRATA_BRANCH_CHECKPOINT_ALIGN's "what would a read from there cut"): `turn` = the last turn token past
// `from` (#458: moved to the turn before it when that one opens with --tail-role-token), -1 with ckpt=0 or none; `root`
// = the first turn token from position 1 on when it is at least `root_min` tokens in (--prompt-cache-root), only for a
// read that starts at token 0 (`root_ok`) and before the last turn token.  `cache_on`: --prompt-cache > 0.
struct TurnRoot {
    int64_t turn = -1, root = -1;
};
template<class Tok>
inline TurnRoot read_turn_root(const Tok* ids, int64_t n, int64_t from, bool root_ok, bool ckpt, bool cache_on,
                               int64_t turn_token, int64_t tail_role_token, int64_t root_min) {
    TurnRoot r;
    if (cache_on && turn_token >= 0)
        for (int64_t i = n - 1; i > from; --i)
            if ((int64_t) ids[i] == turn_token) { r.turn = i; break; }
    if (r.turn > 0 && tail_role_token >= 0)
        for (int64_t i = r.turn - 1; i > from; --i)
            if ((int64_t) ids[i] == turn_token) {
                if ((int64_t) ids[i + 1] == tail_role_token) r.turn = i;
                break;
            }
    const int64_t last_turn = r.turn;   // ckpt=0 drops the split there, the root is still looked for before it
    if (!ckpt) r.turn = -1;
    if (cache_on && turn_token >= 0 && root_min > 0 && root_ok)
        for (int64_t i = 1; i < last_turn; ++i)
            if ((int64_t) ids[i] == turn_token) {
                if (i >= root_min) r.root = i;
                break;
            }
    return r;
}

// STRATA_BRANCH_CHECKPOINT_ALIGN: may a read go on from a checkpoint at `at` taken on the grid `ck_grid` of a read from
// `anchor` (a branch checkpoint, or a periodic mark inside such a segment)?  `end_from_anchor` / `end_from_at`: the end of
// the segment holding `at` in a read of THIS prompt from the anchor / from `at` (the first of that read's own cuts past
// at: its turn boundary, root, last token), `first_cut_from_anchor` the first cut of the read from the anchor past the
// anchor, `grid_now` the super-chunk that read takes for [anchor, end_from_anchor).  Yes only when that read reads
// [anchor, end) as one segment (no cut of its own in (anchor, at]), ending where the read from `at` ends, on the same
// grid as the checkpoint's, and `at` lies on it: then the rest, read in that grid, gives the bits of the read from the
// anchor.
inline bool aligned_resume_check(int64_t at, int64_t anchor, int64_t ck_grid, int64_t first_cut_from_anchor,
                                 int64_t end_from_anchor, int64_t end_from_at, int64_t grid_now) {
    if (anchor < 0 || ck_grid <= 0 || first_cut_from_anchor <= at || end_from_anchor != end_from_at) return false;
    return grid_now == ck_grid && aligned_resume_ok(at, anchor, grid_now, end_from_anchor);
}

// The segment of a read that holds position p: from the last of the read's cuts at or before p (`start`, where the
// read starts, when none) to the first after it (`last`, the last prompt token, when none).  Cuts outside (start, last)
// (-1: absent) are ignored.  A cut AT p makes p the segment's start, so no aligned point lies before it in the segment.
struct ReadSegment {
    int64_t from = -1, end = -1;
};
inline ReadSegment read_segment_of(int64_t p, int64_t start, std::initializer_list<int64_t> cuts, int64_t last) {
    ReadSegment s{start, last};
    for (const int64_t c : cuts) {
        if (c <= start || c >= last) continue;
        if (c <= p && c > s.from) s.from = c;
        if (c > p && c < s.end) s.end = c;
    }
    return s;
}

} // namespace strata::program
