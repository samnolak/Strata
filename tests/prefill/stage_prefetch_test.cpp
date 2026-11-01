// STRATA_STAGE_PREFETCH's rules (include/strata/prefill/stage_prefetch.hpp): what a stage may issue ahead into the next
// stage's slots, when the next stage takes it, and the staged walk's slot-by-slot cursor.  No GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/prefill/stage_prefetch.hpp"

using namespace strata::prefill::stage_prefetch;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static const char kBlob[64] = {};
static const void* B(int i) { return kBlob + i; }

int main() {
    // the switches
    CHECK(!enabled_from(nullptr));
    CHECK(!enabled_from(""));
    CHECK(!enabled_from("0"));
    CHECK(enabled_from("1"));
    CHECK(enabled_from("2"));
    CHECK(!enabled_from("-1"));
    CHECK(walk_min_from(nullptr) == 256);
    CHECK(walk_min_from("") == 256);
    CHECK(walk_min_from("0") == 0);
    CHECK(walk_min_from("512") == 512);
    CHECK(walk_min_from("-3") == 256);

    // plan_prefix: the leading run of page-locked blobs, at most `limit`
    {
        std::vector<Ent> seq = {{3, 1, B(1), true}, {3, 4, B(2), true}, {3, 9, B(3), false}, {3, 11, B(4), true}};
        CHECK(plan_prefix(seq.data(), seq.size(), 96) == 2);   // the stager's entry ends it
        CHECK(plan_prefix(seq.data(), seq.size(), 1) == 1);    // the ring
        CHECK(plan_prefix(seq.data(), seq.size(), 0) == 0);
        CHECK(plan_prefix(seq.data(), 0, 96) == 0);
        seq[0].blob = nullptr;                                  // no blob: nothing
        CHECK(plan_prefix(seq.data(), seq.size(), 96) == 0);
        std::vector<Ent> all(200, Ent{5, 0, B(0), true});
        for (size_t i = 0; i < all.size(); ++i) all[i].e = (int32_t) i;
        CHECK(plan_prefix(all.data(), all.size(), 96) == 96);
        CHECK(plan_prefix(all.data(), 50, 96) == 50);
    }

    // prefix_matches: all or nothing, by layer, expert, blob and page-locking
    {
        std::vector<Ent> seq = {{10, 0, B(1), true}, {10, 2, B(2), true}, {10, 3, B(3), true}, {11, 0, B(4), false}};
        std::vector<Ent> issued = {{10, 0, B(1), true}, {10, 2, B(2), true}};
        CHECK(prefix_matches(issued.data(), issued.size(), seq.data(), seq.size()));
        CHECK(prefix_matches(issued.data(), 0, seq.data(), seq.size()));       // nothing issued: trivially
        CHECK(!prefix_matches(issued.data(), issued.size(), seq.data(), 1));   // the stream is shorter
        std::vector<Ent> wrong_e = {{10, 0, B(1), true}, {10, 3, B(2), true}};
        CHECK(!prefix_matches(wrong_e.data(), wrong_e.size(), seq.data(), seq.size()));
        std::vector<Ent> wrong_l = {{9, 0, B(1), true}};
        CHECK(!prefix_matches(wrong_l.data(), wrong_l.size(), seq.data(), seq.size()));
        std::vector<Ent> wrong_b = {{10, 0, B(7), true}};                       // the blob moved (another source)
        CHECK(!prefix_matches(wrong_b.data(), wrong_b.size(), seq.data(), seq.size()));
        std::vector<Ent> unpinned = {{10, 0, B(1), true}, {10, 2, B(2), true}, {10, 3, B(3), true}, {11, 0, B(4), true}};
        CHECK(!prefix_matches(unpinned.data(), unpinned.size(), seq.data(), seq.size()));   // the stream's entry 3 needs the stager
        // a resident expert of the issuing side's view that the taking side streams (the residency changed between): the
        // streams differ from that point, so nothing is taken
        std::vector<Ent> seq2 = {{10, 0, B(1), true}, {10, 1, B(9), true}, {10, 2, B(2), true}};
        CHECK(!prefix_matches(issued.data(), issued.size(), seq2.data(), seq2.size()));
    }

    // WalkCursor: the staged walk takes slot after slot while the experts match, the first miss ends it
    {
        WalkCursor w;
        CHECK(!w.live);
        CHECK(!w.take(0, 0, 1, B(1), true));   // not armed: nothing
        w.arm({{4, 2, B(1), true}, {4, 7, B(2), true}, {4, 9, B(3), true}});
        CHECK(w.live);
        CHECK(w.take(0, 4, 2, B(1), true));
        CHECK(w.take(1, 4, 7, B(2), true));
        CHECK(w.take(2, 4, 9, B(3), true));
        CHECK(!w.live);                        // every slot taken
        CHECK(w.left() == 0);
        CHECK(w.taken == 3 && w.missed == 0);
        CHECK(!w.take(3, 4, 11, B(4), true));  // past them: the walk's own copies
    }
    {
        // the routing skipped expert 7: the walk stages 9 into slot 1 - a miss; slots 1 and 2 are the walk's again
        WalkCursor w;
        w.arm({{4, 2, B(1), true}, {4, 7, B(2), true}, {4, 9, B(3), true}});
        CHECK(w.take(0, 4, 2, B(1), true));
        CHECK(!w.take(1, 4, 9, B(3), true));
        CHECK(!w.live);
        CHECK(w.left() == 2);
        CHECK(w.taken == 1 && w.missed == 1);
        CHECK(!w.take(2, 4, 9, B(3), true));   // no second chance after a miss (slot 2's bytes are not trusted)
        CHECK(w.missed == 1);
    }
    {
        // a slot out of step, another layer, an unpinned blob, another blob: each one is a miss
        WalkCursor a; a.arm({{4, 2, B(1), true}}); CHECK(!a.take(1, 4, 2, B(1), true)); CHECK(a.left() == 1);
        WalkCursor b; b.arm({{4, 2, B(1), true}}); CHECK(!b.take(0, 5, 2, B(1), true));
        WalkCursor c; c.arm({{4, 2, B(1), true}}); CHECK(!c.take(0, 4, 2, B(1), false));
        WalkCursor d; d.arm({{4, 2, B(1), true}}); CHECK(!d.take(0, 4, 2, B(2), true));
        WalkCursor e; e.arm({}); CHECK(!e.live); CHECK(!e.take(0, 4, 2, B(1), true)); CHECK(e.missed == 0);
    }
    {
        // a layer with fewer stagings than the slots issued: the rest stay untaken (the run waits for their copies)
        WalkCursor w;
        std::vector<Ent> v;
        for (int i = 0; i < 32; ++i) v.push_back({7, (int32_t) (3 * i), B(i % 60), true});
        w.arm(v);
        for (int i = 0; i < 20; ++i) CHECK(w.take((size_t) i, 7, 3 * i, B(i % 60), true));
        CHECK(w.live && w.left() == 12);
        CHECK(!w.take(0, 8, 0, B(0), true));   // the next layer's first staging (slot 0 again): the end
        CHECK(!w.live && w.left() == 12);
    }
    // issue_checked: the loop's own return codes decide; the first failing step ends it
    {
        std::vector<int> calls;
        auto live = [&](size_t i) { return i == 1; };
        auto ok = [&](int tag) { return [&calls, tag](size_t i) { calls.push_back(tag * 100 + (int) i); return 0; }; };
        int err = -1;
        size_t reached = 99;
        CHECK(issue_checked(3, live, ok(1), ok(2), ok(3), &err, &reached) == 3 && err == 0 && reached == 3);
        CHECK((calls == std::vector<int>{200, 300, 101, 201, 301, 202, 302}));   // the wait only for the live slot
        calls.clear();
        auto copy_fails_at_1 = [&](size_t i) { calls.push_back(200 + (int) i); return i == 1 ? 11 : 0; };
        CHECK(issue_checked(3, live, ok(1), copy_fails_at_1, ok(3), &err, &reached) == 1 && err == 11 && reached == 2);
        CHECK((calls == std::vector<int>{200, 300, 101, 201}));                  // nothing after the failing copy
        calls.clear();
        auto wait_fails = [&](size_t) { return 4; };
        CHECK(issue_checked(3, live, wait_fails, ok(2), ok(3), &err, &reached) == 1 && err == 4 && reached == 2);
        auto record_fails_at_0 = [&](size_t i) { return i == 0 ? 7 : 0; };
        CHECK(issue_checked(3, live, ok(1), ok(2), record_fails_at_0, &err, &reached) == 0 && err == 7 && reached == 1);
        CHECK(issue_checked(0, live, ok(1), ok(2), ok(3), &err, &reached) == 0 && err == 0 && reached == 0);
        CHECK(issue_checked(2, live, ok(1), ok(2), ok(3), nullptr) == 2);         // the outputs are optional
    }
    if (failed) { std::fprintf(stderr, "stage_prefetch_test: %d failure(s)\n", failed); return 1; }
    std::printf("stage_prefetch_test: PASS\n");
    return 0;
}
