// include/strata/kernels/kv_host_pieces.hpp - kv-lean (STRATA_KV_HOST_LAZY=1): copies that respect the pieces a lazy
// host K/V copy is page-locked in.  Header-only and free of CUDA so a CPU test can check it (tests/core).
//
// A lazy host copy (layer.cpp) is one address range per array that is cudaHostRegister'ed in pieces as the context grows.
// The kernels read and write across the pieces freely (every page is mapped), but a cudaMemcpy* names ONE allocation:
// a copy whose range spans two registrations fails with "invalid argument" (seen as "kv_stream: stage: invalid
// argument" on the first prompt read past the first piece).  Every registration starts and ends on a multiple of the
// array's piece size counted from the array's start, so a copy cut at those multiples never spans two of them.
//
// The engine records each lazy array here when it builds it (at start-up, before any request); the copy paths of host
// K/V (kv_stream.cu's stage/unstage/ring restore, the conversation snapshot copies) cut their copies with
// kv_host_copy_pieces.  With no lazy array recorded (the default) a copy is one call, exactly as before.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::kernels {

struct KvHostPieceRun {
    uintptr_t start = 0;   ///< the array's first byte
    uintptr_t end = 0;     ///< one past its reserved last byte
    uint64_t piece = 0;    ///< bytes of one registration piece (registrations start and end on multiples of it)
};

/// The recorded lazy arrays, sorted by start (one per array of every lazy host copy).
inline std::vector<KvHostPieceRun>& kv_host_piece_runs() {
    static std::vector<KvHostPieceRun> runs;
    return runs;
}

/// Records an array [start, end) registered in pieces of `piece` bytes from `start`.  Start-up only (not thread-safe
/// against concurrent copies); ignored for an empty range or a zero piece.
inline void kv_host_piece_record(const void* start, const void* end, uint64_t piece) {
    const uintptr_t s = (uintptr_t) start, e = (uintptr_t) end;
    if (e <= s || piece == 0) return;
    auto& runs = kv_host_piece_runs();
    const KvHostPieceRun r{s, e, piece};
    runs.insert(std::upper_bound(runs.begin(), runs.end(), r,
                                 [](const KvHostPieceRun& a, const KvHostPieceRun& b) { return a.start < b.start; }),
                r);
}

/// Bytes from `p` (at most `n`) that stay inside one piece of a recorded array; `n` when `p` is in none.
inline size_t kv_host_piece_room(const void* p, size_t n) {
    const auto& runs = kv_host_piece_runs();
    if (runs.empty() || n == 0) return n;
    const uintptr_t a = (uintptr_t) p;
    auto it = std::upper_bound(runs.begin(), runs.end(), a,
                               [](uintptr_t v, const KvHostPieceRun& r) { return v < r.start; });
    if (it == runs.begin()) return n;
    --it;
    if (a >= it->end) return n;
    const uint64_t off = (uint64_t) (a - it->start);
    const uint64_t next = (off / it->piece + 1) * it->piece;   // the next piece boundary past `p`
    return (size_t) std::min<uint64_t>((uint64_t) n, next - off);
}

/// Calls `copy(dst, src, k)` for consecutive parts of [0, n) none of which spans a piece boundary of a recorded array on
/// either side; returns the first non-zero result (0 = all done).  One call when neither side is in a recorded array.
template <class Copy>
inline int kv_host_copy_pieces(void* dst, const void* src, size_t n, Copy&& copy) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    while (n > 0) {
        const size_t k = std::min(kv_host_piece_room(d, n), kv_host_piece_room(s, n));
        if (const int e = (int) copy((void*) d, (const void*) s, k); e != 0) return e;
        d += k;
        s += k;
        n -= k;
    }
    return 0;
}

/// The lazy growth's plan for one array: the registration [from, to) that takes it from `reg_end` to cover `cells`
/// cells of `bytes_per_cell`, its end rounded up to `page` and held inside [start, reserved_end).  from == to: nothing
/// to register.  (layer.cpp's lazy_grow; tests/core/kv_host_pieces_test.cpp checks the piece invariant on it.)
struct KvHostGrow {
    uintptr_t from = 0, to = 0;
};
inline KvHostGrow kv_host_grow_plan(uintptr_t start, uintptr_t reg_end, uintptr_t reserved_end, int64_t cells,
                                    uint64_t bytes_per_cell, uint64_t page) {
    const uintptr_t want = start + (uintptr_t) ((uint64_t) cells * bytes_per_cell);
    uintptr_t end = (want + page - 1) / page * page;
    end = std::min(end, reserved_end);
    if (end <= reg_end) return {reg_end, reg_end};
    return {reg_end, end};
}

}  // namespace strata::kernels
