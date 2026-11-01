// include/strata/program/idle_park.hpp - STRATA_IDLE_PARK (opt-in): an idle batch slot's conversation goes to the
// conversation cache while the engine has nothing else to do, instead of on the admission path that overwrites the slot.
// Header-only and free of CUDA so a CPU test can check the rules (tests/program/idle_park_test.cpp).
//
// A batch admission that overwrites an idle slot first parks what the slot holds (park_slot: its K/V and GDN state into
// host memory, ~0.4 s for a 100K+ history and ~1 s at 300K) - on the critical path of that admission.  An agent's slot is
// idle while its tool runs (1-16 s), so the park can happen then.  The tracker below watches each slot's idle state by a
// key of what it holds (it changes with every turn) and says which slot to park next: the one idle the longest, once it
// has been idle for at least `min_ms` (an agent that comes straight back to its slot would make a park wasted work), and
// each held state at most once.  The park itself is park_slot's, with one difference: an idle park never evicts another
// parked conversation (it is skipped when the image does not fit beside them), so it can only add to the cache.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::program::idle_park {

/// STRATA_IDLE_PARK: 0 off (unset, empty, 0 or not a number), 1 while no slot decodes and no command waits, 2 also between
/// batch windows (the decoding slots wait for the park).
inline int mode_from(const char* v) {
    if (v == nullptr || *v == '\0') return 0;
    const int n = std::atoi(v);
    return n <= 0 ? 0 : n >= 2 ? 2 : 1;
}

/// STRATA_IDLE_PARK_MS: how long a slot must have been idle before it is parked (default 500 ms; negative: the default).
inline int64_t min_ms_from(const char* v) {
    if (v == nullptr || *v == '\0') return 500;
    const long long n = std::atoll(v);
    return n < 0 ? 500 : (int64_t) n;
}

/// What a slot holds, as a key: 0 when it is not a candidate (decoding, empty, pictures, nothing cached, too short or
/// already held elsewhere - the caller decides), else a value that changes with the tokens it holds.
inline uint64_t slot_key(size_t n_ids, int32_t last_id, bool cvec) {
    if (n_ids == 0) return 0;
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t) n_ids;
    h = (h ^ (uint64_t) (uint32_t) last_id) * 0xbf58476d1ce4e5b9ull;
    h = (h ^ (cvec ? 1u : 2u)) * 0x94d049bb133111ebull;
    h ^= h >> 31;
    return h == 0 ? 1 : h;
}

struct Tracker {
    std::vector<uint64_t> key;    ///< the last key seen per slot (0: not a candidate)
    std::vector<double> since;    ///< when that key was first seen (ms, the caller's clock)
    std::vector<char> done;       ///< the key's park was tried (parked, held already, or skipped)

    /// Slot b holds `k` at `now_ms`: a new key restarts its idle time.
    void see(size_t b, uint64_t k, double now_ms) {
        if (b >= key.size()) {
            key.resize(b + 1, 0);
            since.resize(b + 1, 0.0);
            done.resize(b + 1, 0);
        }
        if (key[b] != k) {
            key[b] = k;
            since[b] = now_ms;
            done[b] = 0;
        }
    }
    /// The slot to park now: the candidate idle the longest, idle for at least `min_ms`, not tried yet; -1 when none, with
    /// `*due_ms` (if given) the time the next one becomes due (-1: none will without a change).
    int pick(double now_ms, double min_ms, double* due_ms = nullptr) const {
        int best = -1;
        double due = -1.0;
        for (size_t b = 0; b < key.size(); ++b) {
            if (key[b] == 0 || done[b]) continue;
            const double at = since[b] + min_ms;
            if (at <= now_ms) {
                if (best < 0 || since[b] < since[(size_t) best]) best = (int) b;
            } else if (due < 0.0 || at < due) {
                due = at;
            }
        }
        if (due_ms != nullptr) *due_ms = best >= 0 ? now_ms : due;
        return best;
    }
    /// The park of slot b's current key was tried: not again until the slot holds something else.
    void tried(size_t b) {
        if (b < done.size()) done[b] = 1;
    }
};

}  // namespace strata::program::idle_park
