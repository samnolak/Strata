// par_refill.hpp: STRATA_REFILL_PAR's scheduling - every item exactly once into its slot, a ring buffer never refilled
// before its copy has left it, pinned items without a gather, a failure stops the run (synthetic: no model, no GPU; the
// "device" copies run late, at the buffer's next wait or the thread's end, as an asynchronous DMA would).
#include "strata/program/par_refill.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

using strata::program::par_refill_part_threads;
using strata::program::par_refill_run;
using strata::program::par_refill_threads;
using strata::program::ParRefillItem;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

namespace {
constexpr int kBlob = 64;
uint8_t byte_of(const ParRefillItem& it, int k) { return (uint8_t) (it.layer * 31 + it.expert * 7 + k); }

struct Fake {
    int parts, per_part, ring;
    std::vector<std::vector<uint8_t>> slots;   // [part][slot * kBlob]
    std::vector<std::vector<uint8_t>> bufs;    // [(part * per_part + t) * ring + r][kBlob]
    struct Pending { bool live = false; int slot = -1; };
    std::vector<Pending> pend;                 // per buffer: the copy queued from it, not landed yet
    std::vector<int> begun, ended, gathers, pinned_issues;
    std::mutex mu;
    int fail_layer = -1;                       // gather fails for this layer
    Fake(int p, int pp, int r, int nslots)
        : parts(p), per_part(pp), ring(r), slots((size_t) p, std::vector<uint8_t>((size_t) nslots * kBlob, 0)),
          bufs((size_t) p * pp * r, std::vector<uint8_t>(kBlob, 0)), pend((size_t) p * pp * r),
          begun((size_t) p * pp, 0), ended((size_t) p * pp, 0), gathers((size_t) p, 0), pinned_issues((size_t) p, 0) {}
    size_t bi(int p, int t, int r) const { return ((size_t) p * per_part + t) * ring + r; }
    void land(int p, int t, int r) {   // the late copy: whatever the buffer holds NOW goes into the slot
        Pending& q = pend[bi(p, t, r)];
        if (!q.live) return;
        std::memcpy(&slots[(size_t) p][(size_t) q.slot * kBlob], bufs[bi(p, t, r)].data(), kBlob);
        q.live = false;
    }
};

bool run(Fake& f, const std::vector<std::vector<ParRefillItem>>& items) {
    return par_refill_run(
        items, f.per_part, f.ring,
        [&](int p, int t) { f.begun[(size_t) p * f.per_part + t]++; return true; },
        [&](int p, int t, int r) { f.land(p, t, r); return true; },
        [&](int p, int t, int r, const ParRefillItem& it) {
            if (it.layer == f.fail_layer) return false;
            CHECK(!f.pend[f.bi(p, t, r)].live);   // the ring discipline: never refilled before its copy left
            for (int k = 0; k < kBlob; ++k) f.bufs[f.bi(p, t, r)][(size_t) k] = byte_of(it, k);
            std::lock_guard<std::mutex> lk(f.mu);
            f.gathers[(size_t) p]++;
            return true;
        },
        [&](int p, int t, int r, const ParRefillItem& it) {
            if (r < 0) {   // pinned: straight from the "resident copy"
                for (int k = 0; k < kBlob; ++k) f.slots[(size_t) p][(size_t) it.slot * kBlob + k] = byte_of(it, k);
                std::lock_guard<std::mutex> lk(f.mu);
                f.pinned_issues[(size_t) p]++;
                return true;
            }
            Fake::Pending& q = f.pend[f.bi(p, t, r)];
            q.live = true;
            q.slot = it.slot;
            return true;
        },
        [&](int p, int t) {
            for (int r = 0; r < f.ring; ++r) f.land(p, t, r);
            f.ended[(size_t) p * f.per_part + t]++;
            return true;
        });
}
}  // namespace

int main() {
    CHECK(par_refill_threads(nullptr) == 0);
    CHECK(par_refill_threads("") == 0);
    CHECK(par_refill_threads("0") == 0);
    CHECK(par_refill_threads("3") == 3);
    CHECK(par_refill_threads("99") == 16);
    CHECK(par_refill_threads("-2") == 0);
    CHECK(par_refill_part_threads(0, 4) == 0);
    CHECK(par_refill_part_threads(2, 4) == 2);
    CHECK(par_refill_part_threads(100, 4) == 4);
    CHECK(par_refill_part_threads(100, 0) == 0);

    // four stages: empty, one item, a few, many (slots shuffled, every 5th item pinned)
    const int nslots = 1200;
    for (int per_part : {1, 3, 4}) {
        for (int ring : {1, 2, 4}) {
            Fake f(4, per_part, ring, nslots);
            std::vector<std::vector<ParRefillItem>> items(4);
            const int counts[4] = {0, 1, 7, 1000};
            for (int p = 0; p < 4; ++p)
                for (int i = 0; i < counts[p]; ++i) {
                    ParRefillItem it;
                    it.layer = p * 12 + i % 12;
                    it.expert = i;
                    it.slot = (i * 7919) % nslots;   // distinct for i < nslots (7919 is prime, coprime with 1200)
                    it.bytes = kBlob;
                    it.pinned = i % 5 == 4;
                    items[(size_t) p].push_back(it);
                }
            CHECK(run(f, items));
            for (int p = 0; p < 4; ++p) {
                int pinned = 0;
                for (const ParRefillItem& it : items[(size_t) p]) {
                    pinned += it.pinned;
                    bool same = true;
                    for (int k = 0; k < kBlob; ++k) same = same && f.slots[(size_t) p][(size_t) it.slot * kBlob + k] == byte_of(it, k);
                    CHECK(same);
                }
                CHECK(f.gathers[(size_t) p] == counts[p] - pinned);
                CHECK(f.pinned_issues[(size_t) p] == pinned);
                const int nt = par_refill_part_threads((size_t) counts[p], per_part);
                for (int t = 0; t < per_part; ++t) {
                    CHECK(f.begun[(size_t) p * per_part + t] == (t < nt ? 1 : 0));
                    CHECK(f.ended[(size_t) p * per_part + t] == (t < nt ? 1 : 0));
                }
            }
        }
    }

    // a failed gather: false, every started thread still ends (its copies waited for)
    {
        Fake f(2, 3, 4, nslots);
        f.fail_layer = 5;
        std::vector<std::vector<ParRefillItem>> items(2);
        for (int i = 0; i < 200; ++i) {
            items[0].push_back({i % 10, i, i, kBlob, false});
            items[1].push_back({100 + i % 10, i, i, kBlob, false});
        }
        CHECK(!run(f, items));
        for (int k = 0; k < 6; ++k) CHECK(f.begun[(size_t) k] == f.ended[(size_t) k]);
    }

    if (fails == 0) std::printf("par_refill_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
