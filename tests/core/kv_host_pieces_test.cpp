// tests/core/kv_host_pieces_test.cpp - kv-lean: the lazy host K/V's registration plan and the copy cutting
// (include/strata/kernels/kv_host_pieces.hpp).  CPU only, no CUDA.
//
// The failure it guards: a lazy host copy is cudaHostRegister'ed in pieces as the context grows, and a cudaMemcpy* whose
// range spans two registrations fails ("kv_stream: stage: invalid argument" on the first prompt read past the first
// piece).  Here the registrations are simulated exactly as layer.cpp's lazy_grow makes them (kv_host_grow_plan, steps
// of `step` cells, the end held at the reserved end), and every copy kv_host_copy_pieces makes must lie inside ONE
// registration on each side that is a lazy array.
#include "strata/kernels/kv_host_pieces.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
int fails = 0;
long checks = 0;
#define CHECK(c)                                                                               \
    do {                                                                                       \
        ++checks;                                                                              \
        if (!(c)) {                                                                            \
            ++fails;                                                                           \
            if (fails < 20) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
        }                                                                                      \
    } while (0)

using strata::kernels::kv_host_copy_pieces;
using strata::kernels::kv_host_grow_plan;
using strata::kernels::kv_host_piece_record;
using strata::kernels::kv_host_piece_room;
using strata::kernels::kv_host_piece_runs;

struct Reg { uintptr_t a, b; };

// One simulated lazy array: its registrations as lazy_grow would make them.
struct Array {
    uintptr_t start, reserved_end;
    uint64_t bytes_per_cell;
    int64_t max_cells;
    uintptr_t reg_end;
    std::vector<Reg> regs;
    void grow(int64_t cells, uint64_t page) {
        cells = std::min(cells, max_cells);
        const auto g = kv_host_grow_plan(start, reg_end, reserved_end, cells, bytes_per_cell, page);
        if (g.to > g.from) { regs.push_back({g.from, g.to}); reg_end = g.to; }
    }
    int which(uintptr_t p) const {   // the registration holding byte p, or -1
        for (size_t i = 0; i < regs.size(); ++i) if (p >= regs[i].a && p < regs[i].b) return (int) i;
        return -1;
    }
};

// layer.cpp's growth targets: init (a whole number of steps), then whole steps past what is asked, at most max_cells
int64_t grow_target(int64_t have, int64_t cells, int64_t step, int64_t max_cells) {
    if (cells <= have) return have;
    return std::min(max_cells, (std::max(cells, have + step) + step - 1) / step * step);
}

void run_case(uint64_t rows_per_cell, uint64_t row_bytes, int64_t max_cells, int64_t step, uint64_t page, unsigned seed) {
    kv_host_piece_runs().clear();
    const uint64_t A = 2ull << 20;
    const uint64_t bpc = rows_per_cell * row_bytes;
    const uint64_t reserved = ((uint64_t) max_cells * bpc + A - 1) / A * A;
    // two arrays (a lazy source and a lazy destination of the same layout, as a slot copy has) plus device memory
    const uintptr_t base_src = (uintptr_t) 0x7f0000000000ull, base_dst = base_src + reserved + A * 4;
    Array src{base_src, base_src + reserved, bpc, max_cells, base_src, {}};
    Array dst{base_dst, base_dst + reserved, bpc, max_cells, base_dst, {}};
    kv_host_piece_record((const void*) src.start, (const void*) src.reserved_end, (uint64_t) step * bpc);
    kv_host_piece_record((const void*) dst.start, (const void*) dst.reserved_end, (uint64_t) step * bpc);
    CHECK(kv_host_piece_runs().size() == 2);
    std::mt19937_64 rng(seed);
    int64_t have_s = 0, have_d = 0;
    have_s = grow_target(0, step, step, max_cells);   // the initial registration (one step)
    src.grow(have_s, page);
    have_d = grow_target(0, step, step, max_cells);
    dst.grow(have_d, page);
    for (int round = 0; round < 400; ++round) {
        // a request reaches further (sometimes far: a long prompt read in one go, as the pre-ensure does)
        const int64_t ask = std::min<int64_t>(max_cells, (int64_t) (rng() % (uint64_t) max_cells) + 1);
        have_s = grow_target(have_s, ask, step, max_cells);
        src.grow(have_s, page);
        have_d = grow_target(have_d, ask, step, max_cells);
        dst.grow(have_d, page);
        // every registration starts on a piece boundary of its array
        for (const Reg& r : src.regs) CHECK((r.a - src.start) % ((uint64_t) step * bpc) == 0);
        for (const Reg& r : src.regs) CHECK(r.b <= src.reserved_end && r.b % page == 0);
        // copies of cell ranges inside what both have registered: lazy -> lazy (slot copy), lazy -> device (stage),
        // device -> lazy (unstage / restore from a VRAM slot), each cut so that no part spans two registrations
        const int64_t have = std::min(have_s, have_d);
        for (int c = 0; c < 20; ++c) {
            const int64_t c0 = (int64_t) (rng() % (uint64_t) have), c1 = c0 + 1 + (int64_t) (rng() % (uint64_t) (have - c0));
            const uint64_t off = (uint64_t) c0 * bpc, n = (uint64_t) (c1 - c0) * bpc;
            const int kind = (int) (rng() % 3);
            const uintptr_t dev = (uintptr_t) 0x100000000ull;   // device memory: not a recorded array
            const uintptr_t s = kind == 2 ? dev + off : src.start + off;
            const uintptr_t d = kind == 1 ? dev + off : dst.start + off;
            uint64_t covered = 0;
            int parts = 0;
            const int e = kv_host_copy_pieces((void*) d, (const void*) s, (size_t) n, [&](void* pd, const void* ps, size_t k) {
                const uintptr_t xd = (uintptr_t) pd, xs = (uintptr_t) ps;
                CHECK(k > 0);
                CHECK(xs == s + covered && xd == d + covered);   // consecutive, in order
                if (kind != 2) CHECK(src.which(xs) >= 0 && src.which(xs) == src.which(xs + k - 1));
                if (kind != 1) CHECK(dst.which(xd) >= 0 && dst.which(xd) == dst.which(xd + k - 1));
                covered += k;
                ++parts;
                return 0;
            });
            CHECK(e == 0);
            CHECK(covered == n);   // the whole range, nothing more
            CHECK(parts >= 1);
        }
    }
    // the whole context registered: one more ask changes nothing
    have_s = grow_target(have_s, max_cells, step, max_cells);
    src.grow(have_s, page);
    const size_t nregs = src.regs.size();
    src.grow(max_cells, page);
    CHECK(src.regs.size() == nregs);
    CHECK(src.reg_end <= src.reserved_end);
    CHECK(src.reg_end >= src.start + (uintptr_t) ((uint64_t) max_cells * bpc));
}
}  // namespace

int main() {
    // no recorded array: a copy is ONE call, unchanged (the default, STRATA_KV_HOST_LAZY unset)
    kv_host_piece_runs().clear();
    {
        int calls = 0;
        size_t got = 0;
        char a[64], b[64];
        CHECK(kv_host_copy_pieces(a, b, sizeof a, [&](void*, const void*, size_t k) { ++calls; got = k; return 0; }) == 0);
        CHECK(calls == 1 && got == sizeof a);
        CHECK(kv_host_piece_room(a, 12345) == 12345);
    }
    // an error from a part stops the copy and is returned
    {
        kv_host_piece_runs().clear();
        kv_host_piece_record((const void*) 0x1000000, (const void*) 0x2000000, 0x1000);
        int calls = 0;
        CHECK(kv_host_copy_pieces((void*) 0x1000800, (const void*) 0x5000000, 0x3000, [&](void*, const void*, size_t) {
                  return ++calls == 2 ? 7 : 0;
              }) == 7);
        CHECK(calls == 2);
        // room: up to the next boundary inside the array, the whole length outside it
        CHECK(kv_host_piece_room((const void*) 0x1000800, 0x10000) == 0x800);
        CHECK(kv_host_piece_room((const void*) 0x1001000, 0x10000) == 0x1000);
        CHECK(kv_host_piece_room((const void*) 0x1000800, 0x100) == 0x100);
        CHECK(kv_host_piece_room((const void*) 0x2000000, 0x10000) == 0x10000);   // one past the end: not in it
        CHECK(kv_host_piece_room((const void*) 0x0fff000, 0x10000) == 0x10000);   // before it
    }
    // the shapes the engine uses: k8v4 (K codes 256 B/row, K scales 8 B/row, rotated Q4_0 V 144 B/row; 2 rows a cell),
    // int8 / f16 rows, the default 16384-cell step and a small one, max contexts that are and are not whole steps
    const uint64_t rb[] = {256, 8, 144, 512};
    const int64_t ctx[] = {131072, 262144, 393216, 524288, 200003};
    const int64_t steps[] = {16384, 1024};
    unsigned seed = 1;
    for (const uint64_t b : rb)
        for (const int64_t m : ctx)
            for (const int64_t st : steps) run_case(2, b, m, st, 4096, seed++);
    if (fails == 0) std::printf("kv_host_pieces_test: %ld checks passed\n", checks);
    else std::printf("kv_host_pieces_test: %d of %ld checks FAILED\n", fails, checks);
    return fails == 0 ? 0 : 1;
}
