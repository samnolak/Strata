// include/strata/program/idle_park.hpp - STRATA_IDLE_PARK (opt-in): an idle batch slot's conversation goes to the
// conversation cache while the engine has nothing else to do, instead of on the admission path that overwrites the slot.
// Header-only and free of CUDA so a CPU test can check the rules (tests/program/idle_park_test.cpp).
//
// A batch admission that overwrites an idle slot first parks what the slot holds (park_slot: its K/V and GDN state into
// host memory, ~0.4 s for a 100K+ history and ~1 s at 300K) - on the critical path of that admission.  An agent's slot is
// idle while its tool runs (1-16 s), so the park can happen then.  The tracker below watches each slot's idle state by a
// key of what it holds (it changes with every turn) and says which slot to park next: the one idle the longest, once it
// has been idle for at least `min_ms` (an agent that comes straight back to its slot would make a park wasted work), and
// each held state at most once.  The park itself is park_slot's, with these differences: an idle park never evicts another
// parked conversation (it is skipped when the image does not fit beside them, and drops the copies it supersedes only
// when it stores the image), so it can only add to the cache; it leaves STRATA_IDLE_PARK_FREE_MIB more RAM free than the
// conversation cache's floor; and it stops between its copies when a request comes in (core/capture_abort.hpp), so the
// request waits for at most one K/V layer's copy, not the whole park.
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

/// STRATA_IDLE_PARK_FREE_MIB: the free RAM an idle park leaves above --conversation-cache-min-free-mib (default 4096 MiB;
/// negative: the default).  An idle park's copy may be one nobody restores (the agent comes back to its own slot), so it
/// takes RAM only with this much more to spare than an admission's park would.
inline int64_t free_mib_from(const char* v) {
    if (v == nullptr || *v == '\0') return 4096;
    const long long n = std::atoll(v);
    return n < 0 ? 4096 : (int64_t) n;
}

/// STRATA_IDLE_PARK_WATCHDOG_S: how long an idle park may go without a beat before the watchdog ends the engine (default
/// 600 s, 10x the request watchdog's default: a restart costs far more than a slow park; it is an allowance on top of
/// STRATA_WATCHDOG_S, see core/progress.hpp progress_allow); 0: an idle park is not watched (as before the review fix);
/// negative or not a number: the default.
inline int64_t watchdog_s_from(const char* v) {
    if (v == nullptr || *v == '\0') return 600;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || n < 0) return 600;
    return (int64_t) n;
}

/// Whether an idle park may start at all: the RAM available now (none read: no) leaves `floor_mib + free_mib` free.  The
/// capture's own admission (its size against the same floor) follows; this only spares the work of sizing and splitting
/// the image when the host has no room anyway.
inline bool ram_allows(bool have_reading, uint64_t available_bytes, int64_t floor_mib, int64_t free_mib) {
    if (!have_reading) return false;
    const uint64_t need = ((uint64_t) (floor_mib > 0 ? floor_mib : 0) + (uint64_t) (free_mib > 0 ? free_mib : 0)) << 20;
    return available_bytes >= need;
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
    /// The park of slot b was stopped part-way (a request came in): it may be tried again for the same key, after a new
    /// `min_ms` wait from `now_ms` (the request that stopped it most likely uses the engine for a while).
    void retry(size_t b, double now_ms) {
        if (b >= done.size()) return;
        done[b] = 0;
        since[b] = now_ms;
    }
};

}  // namespace strata::program::idle_park
