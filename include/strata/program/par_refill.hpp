// include/strata/program/par_refill.hpp - STRATA_REFILL_PAR: the lent cache slots of every layer-split stage refilled at
// once, each stage by its own threads (header-only: the scheduling; the copies are the caller's callbacks, so the test
// runs it without a GPU).
//
// Why: a batch admission's prompt read borrows the tail of every stage's expert cache and gives it back afterwards
// (`refill` in generate.cpp).  With the GGUF read in place (--mmap-experts, no RAM copy) each lent slot's blob is first
// assembled from the mapping in a pageable staging buffer on the serve thread and then copied by the driver, which
// stages a pageable source itself and returns only when the copy is done - so the "queue every stage, then wait"
// scheme (#340) ran stage after stage at ~8.4 GB/s (E11: 594 ms of a 2.8 s admission, 368 slots per stage 190 ms
// apart).  Here every stage gets `per_part` threads; each gathers its share of the stage's blobs into its own
// page-locked ring buffers (the page-cache reads of all threads run at once) and queues the copy from there, which
// is a real asynchronous DMA, on the stage's device.  The bytes and the slots are the ones `refill` writes; only
// the route and the order of the copies differ.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace strata::program {

struct ParRefillItem {
    int32_t layer = 0, expert = 0, slot = 0;
    int64_t bytes = 0;
    bool pinned = false;   ///< the source's blob is already page-locked (resident RAM): copied from there, no gather
};

/// STRATA_REFILL_PAR=N: N gather threads per stage (0 / unset: off - the serial refill); clamped to [0, 16].
inline int par_refill_threads(const char* env) {
    if (env == nullptr || env[0] == '\0') return 0;
    const int v = std::atoi(env);
    return std::clamp(v, 0, 16);
}

/// Thread t of a part with n items and `threads` threads takes items t, t + T, t + 2T, ... (T = the threads the part
/// really gets: at most one per item).  Interleaved, so every thread works through the whole stage and they end
/// together.
inline int par_refill_part_threads(size_t n_items, int threads) {
    if (n_items == 0 || threads <= 0) return 0;
    return (int) std::min<size_t>((size_t) threads, n_items);
}

/// Runs every part's items at once (part = one stage's lent slots, on its own device).  Callbacks, called on the
/// worker threads (part p, thread t, ring buffer r):
///   begin(p, t)            - first, once per thread (make the device current, have the ring); false: failure
///   wait(p, t, r)          - before buffer r is filled again: its previous copy must have left it; false: failure
///   gather(p, t, r, item)  - the item's blob into buffer r (never called for a pinned item); false: failure
///   issue(p, t, r, item)   - queue the copy into item.slot from buffer r, or from the page-locked source when r < 0
///                            (a pinned item); false: failure
///   end(p, t)              - last, once per thread, also after a failure (wait for its copies); false: failure
/// Returns false when any callback failed; the other threads stop at their next item.  `ring` >= 1 buffers per thread.
template <class Begin, class Wait, class Gather, class Issue, class End>
bool par_refill_run(const std::vector<std::vector<ParRefillItem>>& parts, int per_part, int ring, Begin&& begin,
                    Wait&& wait, Gather&& gather, Issue&& issue, End&& end) {
    if (ring < 1) ring = 1;
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    for (size_t p = 0; p < parts.size(); ++p) {
        const int nt = par_refill_part_threads(parts[p].size(), per_part);
        for (int t = 0; t < nt; ++t) {
            threads.emplace_back([&, p, t, nt] {
                const int pi = (int) p;
                if (begin(pi, t)) {
                    int64_t used = 0;   // buffers this thread has filled so far (round robin over its ring)
                    for (size_t k = (size_t) t; k < parts[p].size(); k += (size_t) nt) {
                        if (failed.load(std::memory_order_relaxed)) break;
                        const ParRefillItem& it = parts[p][k];
                        if (it.pinned) {
                            if (!issue(pi, t, -1, it)) { failed.store(true); break; }
                            continue;
                        }
                        const int r = (int) (used % ring);
                        if ((used >= ring && !wait(pi, t, r)) || !gather(pi, t, r, it) || !issue(pi, t, r, it)) {
                            failed.store(true);
                            break;
                        }
                        ++used;
                    }
                } else {
                    failed.store(true);
                }
                if (!end(pi, t)) failed.store(true);
            });
        }
    }
    for (std::thread& th : threads) th.join();
    return !failed.load();
}

}  // namespace strata::program
