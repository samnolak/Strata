// include/strata/prefill/stage_prefetch.hpp - STRATA_STAGE_PREFETCH (opt-in): a layer split's next stage gets its first
// expert copies while the stage before it still reads the chunk.  Header-only and free of CUDA so a CPU test can check
// the rules (tests/prefill/stage_prefetch_test.cpp).
//
// A chunk runs the stages one after the other (run_impl hands its rows on at the chunk's end), and each stage streams its
// non-resident experts over its own PCIe link.  While stage k computes, stage k+1's link is idle and so are its staging
// slots (its previous chunk is done).  With the flag stage k issues, at the start of its chunk, the copies stage k+1 makes
// first for that chunk - on stage k+1's copy stream, into stage k+1's own slots, recording the same "copied" events:
//   - the streamed walk (a chunk of at least stream_all_min() tokens): the first `ring` entries of stage k+1's stream,
//     which does not depend on the routing (every non-resident expert of every layer in id order);
//   - the staged walk (a shorter chunk): the first staging slots' worth of stage k+1's first layer's non-resident experts
//     in id order - the experts that layer's routing then picks, unless it skips one (at >= 256 tokens x top-10 of 512 a
//     skipped expert is rare: P = (1 - 10/512)^T).
// Stage k+1 then takes the issued entries when they are exactly the ones it would have issued itself (same expert, same
// page-locked blob, same slot), else it waits for them and issues everything itself.  The same bytes in the same slots
// behind the same events: the same results.  Only page-locked blobs are issued ahead (a DMA; nothing for the stager).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::prefill::stage_prefetch {

/// STRATA_STAGE_PREFETCH: on when the text is a number above 0 (unset, empty or 0: off).
inline bool enabled_from(const char* v) { return v != nullptr && std::atoi(v) > 0; }

/// STRATA_STAGE_PREFETCH_WALK_MIN: the shortest chunk the staged walk's prefetch is tried at (default 256 tokens; 0 turns
/// the staged walk's part off, the streamed walk's stays).
inline int64_t walk_min_from(const char* v) {
    if (v == nullptr || *v == '\0') return 256;
    const long long n = std::atoll(v);
    return n < 0 ? 256 : (int64_t) n;
}

/// One stream entry: the layer, the expert, its blob and whether the blob is page-locked (a DMA without a host copy).
struct Ent {
    int32_t l = -1, e = -1;
    const void* blob = nullptr;
    bool pinned = false;
};

/// The entries a previous stage may issue ahead: the leading run of `seq` (this stage's stream order) whose blobs are
/// page-locked, at most `limit` (the slots) of them.  The first entry that needs the stager (or has no blob) ends it.
inline size_t plan_prefix(const Ent* seq, size_t n_seq, size_t limit) {
    size_t n = 0;
    while (n < n_seq && n < limit && seq[n].pinned && seq[n].blob != nullptr) ++n;
    return n;
}

/// Whether the `n_issued` entries issued ahead are exactly the first ones of this stage's own stream (same layer, expert
/// and blob, page-locked): all of them are taken or none.
inline bool prefix_matches(const Ent* issued, size_t n_issued, const Ent* seq, size_t n_seq) {
    if (n_issued > n_seq) return false;
    for (size_t i = 0; i < n_issued; ++i)
        if (issued[i].l != seq[i].l || issued[i].e != seq[i].e || issued[i].blob != seq[i].blob || !seq[i].pinned ||
            !issued[i].pinned)
            return false;
    return true;
}

/// The staged walk's side: the layer stages its non-resident routed experts in id order, the i-th into slot i % stage_n.
/// Slot s < n holds the expert issued ahead as entry s.  take(s, e, blob) is asked for each staging in order: true when
/// slot s already holds that blob (the walk skips its own copy); the first miss ends the prefetch for good, and the walk
/// copies over the remaining slots itself (its copies queue behind the early ones on the same copy stream).
struct WalkCursor {
    std::vector<Ent> ents;   ///< what was issued ahead, slot by slot
    size_t next = 0;         ///< the slot the next staging must be to match
    bool live = false;       ///< false: nothing (more) to take
    size_t taken = 0, missed = 0;

    void arm(std::vector<Ent> issued) {
        ents = std::move(issued);
        next = 0;
        live = !ents.empty();
    }
    bool take(size_t slot, int32_t l, int32_t e, const void* blob, bool pinned) {
        if (!live) return false;
        if (slot == next && next < ents.size() && ents[next].l == l && ents[next].e == e && ents[next].blob == blob &&
            pinned) {
            ++next;
            ++taken;
            if (next == ents.size()) live = false;
            return true;
        }
        live = false;
        ++missed;
        return false;
    }
    /// The slots issued ahead that the walk did not take (their copies may still be in flight: wait before the slots'
    /// memory is anything else's).
    size_t left() const { return ents.size() - next; }
};

/// The issue loop of stage_prefetch_issue, checked by its own calls' return codes (not the thread's last CUDA error, which
/// may hold another part's pending error): for slot i < n, wait(i) when live(i) (the slot's release event), copy(i) (the
/// DMA), record(i) (its "copied" event) - each returns 0 on success, else the error code it got.  The first failing step
/// ends the loop.  Returns how many slots had all their steps succeed; `*first_err` gets the failing code (0: none) and
/// `*reached` (if given) the slots a step was tried for (the failing one included: its copy may be queued).
template <class Live, class Wait, class Copy, class Record>
inline size_t issue_checked(size_t n, Live&& live, Wait&& wait, Copy&& copy, Record&& record, int* first_err,
                            size_t* reached = nullptr) {
    int e = 0;
    size_t i = 0;
    for (; i < n; ++i) {
        if (live(i) && (e = (int) wait(i)) != 0) break;
        if ((e = (int) copy(i)) != 0) break;
        if ((e = (int) record(i)) != 0) break;
    }
    if (first_err != nullptr) *first_err = e;
    if (reached != nullptr) *reached = e != 0 ? i + 1 : i;
    return i;
}

}  // namespace strata::prefill::stage_prefetch
