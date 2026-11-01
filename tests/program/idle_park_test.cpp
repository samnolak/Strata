// STRATA_IDLE_PARK's rules (include/strata/program/idle_park.hpp): which idle slot is parked when.  No GPU, no model.
#include <cstdio>
#include <cstdlib>

#include "strata/program/idle_park.hpp"
#include "strata/core/capture_abort.hpp"
#include "strata/program/park_admit.hpp"

#include <string>

using namespace strata::program::idle_park;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    CHECK(mode_from(nullptr) == 0);
    CHECK(mode_from("") == 0);
    CHECK(mode_from("0") == 0);
    CHECK(mode_from("x") == 0);
    CHECK(mode_from("1") == 1);
    CHECK(mode_from("2") == 2);
    CHECK(mode_from("7") == 2);
    CHECK(mode_from("-1") == 0);
    CHECK(min_ms_from(nullptr) == 500);
    CHECK(min_ms_from("0") == 0);
    CHECK(min_ms_from("1500") == 1500);
    CHECK(min_ms_from("-5") == 500);

    CHECK(slot_key(0, 5, true) == 0);
    CHECK(slot_key(10, 5, true) != 0);
    CHECK(slot_key(10, 5, true) != slot_key(11, 5, true));   // one more token: a new key
    CHECK(slot_key(10, 5, true) != slot_key(10, 6, true));
    CHECK(slot_key(10, 5, true) != slot_key(10, 5, false));
    CHECK(slot_key(10, 5, true) == slot_key(10, 5, true));

    {
        Tracker t;
        double due = 0;
        CHECK(t.pick(0, 500, &due) == -1 && due < 0);   // no slots
        t.see(0, 0, 0);                                  // slot 0 decodes
        t.see(1, slot_key(1000, 7, true), 100);          // slot 1 idle from 100
        t.see(2, slot_key(2000, 9, true), 300);          // slot 2 idle from 300
        CHECK(t.pick(200, 500, &due) == -1 && due == 600);   // slot 1 is due at 600
        CHECK(t.pick(600, 500, &due) == 1 && due == 600);
        CHECK(t.pick(900, 500) == 1);                    // both due: the one idle the longest
        t.tried(1);
        CHECK(t.pick(900, 500) == 2);
        t.tried(2);
        CHECK(t.pick(5000, 500, &due) == -1 && due < 0);   // each held state once
        t.see(1, slot_key(1000, 7, true), 6000);         // the same state seen again: still tried
        CHECK(t.pick(7000, 500) == -1);
        t.see(1, slot_key(1300, 4, true), 7000);         // the next turn ended in slot 1: a new state, a new wait
        CHECK(t.pick(7200, 500, &due) == -1 && due == 7500);
        CHECK(t.pick(7500, 500) == 1);
        t.see(1, 0, 7600);                               // admitted again (decoding): not a candidate
        CHECK(t.pick(9000, 500) == -1);
        t.see(1, slot_key(1300, 4, true), 9100);         // idle again with what it held before: a new idle spell
        CHECK(t.pick(9600, 500) == 1);
        CHECK(t.pick(9100, 0) == 1);                     // min 0: at once
    }
    {
        Tracker t;   // a slot index past the tracker's size grows it
        t.see(5, slot_key(3, 1, true), 0);
        CHECK(t.key.size() == 6);
        CHECK(t.pick(1000, 500) == 5);
        t.tried(9);   // out of range: ignored
        CHECK(t.pick(1000, 500) == 5);
    }
    // STRATA_IDLE_PARK_FREE_MIB and the RAM gate before an idle park
    CHECK(free_mib_from(nullptr) == 4096);
    CHECK(free_mib_from("") == 4096);
    CHECK(free_mib_from("0") == 0);
    CHECK(free_mib_from("8192") == 8192);
    CHECK(free_mib_from("-1") == 4096);
    CHECK(!ram_allows(false, ~0ull, 0, 0));                                   // no reading: no park
    CHECK(ram_allows(true, (12288ull + 4096) << 20, 12288, 4096));            // exactly the floor plus the headroom
    CHECK(!ram_allows(true, ((12288ull + 4096) << 20) - 1, 12288, 4096));
    CHECK(ram_allows(true, 0, 0, 0));
    CHECK(ram_allows(true, 1ull << 20, -5, -5));                              // negative figures count as 0
    // a park stopped part-way (a request came in) is tried again after a new wait
    {
        Tracker t;
        t.see(0, slot_key(5000, 3, true), 0);
        CHECK(t.pick(600, 500) == 0);
        t.tried(0);
        CHECK(t.pick(700, 500) == -1);
        t.retry(0, 800);
        double due = 0;
        CHECK(t.pick(900, 500, &due) == -1 && due == 1300);
        CHECK(t.pick(1300, 500) == 0);
        t.see(0, slot_key(5000, 3, true), 1400);   // the same state: the retry's clock stays
        CHECK(t.pick(1400, 500) == 0);
        t.retry(7, 0);                              // out of range: ignored
        CHECK(t.key.size() == 1);
    }
    // capture_abort.hpp: a capture asks the thread's scope; none (every other caller): never stops
    {
        using namespace strata::core;
        std::string err;
        CHECK(!capture_abort_requested());
        CHECK(capture_continue(err) && err.empty());
        bool stop = false;
        {
            CaptureAbortScope s(+[](void* c) { return *static_cast<bool*>(c); }, &stop);
            CHECK(capture_continue(err) && err.empty());
            stop = true;
            CHECK(!capture_continue(err) && capture_aborted(err));
            err.clear();
            {
                bool inner = false;   // scopes nest: the inner one asks, the outer comes back after
                CaptureAbortScope s2(+[](void* c) { return *static_cast<bool*>(c); }, &inner);
                CHECK(capture_continue(err));
            }
            CHECK(!capture_continue(err));
        }
        err.clear();
        CHECK(capture_continue(err) && err.empty());   // gone with its scope
        CHECK(!capture_aborted("stage sync failed") && !capture_aborted(""));
    }
    CHECK(watchdog_s_from(nullptr) == 600);
    CHECK(watchdog_s_from("") == 600);
    CHECK(watchdog_s_from("0") == 0);
    CHECK(watchdog_s_from("90") == 90);
    CHECK(watchdog_s_from("-1") == 600);
    CHECK(watchdog_s_from("x") == 600);
    // a slot skipped for RAM: retry() (not left tried) - due again after a new wait
    {
        Tracker t;
        t.see(0, slot_key(4000, 2, true), 0);
        CHECK(t.pick(500, 500) == 0);
        t.tried(0);
        t.retry(0, 500);   // what idle_park_one does when ram_allows says no
        CHECK(t.pick(600, 500) == -1);
        CHECK(t.pick(1000, 500) == 0);
    }
    // park_admit.hpp in park_source's order: another agent's conversation parked first (oldest), this slot's previous
    // turn after it, both cache slots taken; the slot's next turn is parked while idle
    {
        using namespace strata::core;
        namespace pa = strata::program::park_admit;
        auto image = [](std::initializer_list<int32_t> ids) {
            SavedConversation s; s.live.ids = ids; s.live.gdn.resize(64, 7); s.cvec = true; return s; };
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto fill = [&](ConversationCache& c) {
            SavedConversation other = image({9, 8, 7, 6});
            SavedConversation prev = image({1, 2, 3, 4, 60, 70});
            prev.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
            c.put(std::move(other));
            c.put(std::move(prev));
        };
        const std::vector<int32_t> next = {1, 2, 3, 4, 60, 70, 80, 81};
        const std::vector<ConversationCheckpoint> chain = {cp({1, 2, 3, 4, 60})};
        const std::vector<int32_t> other_ids = {9, 8, 7, 6};
        SavedConversation probe = image({1, 2, 3, 4, 60, 70});
        const size_t one = probe.bytes();

        // the review's repro: before the fix make_room ran here and evicted the other agent before the capture
        {
            ConversationCache c(1 << 20, 2);
            fill(c);
            CHECK(pa::before_capture(c, one, 0, true, next, {}, chain, true) == pa::Room::ok);
            CHECK(c.size() == 2 && c.evictions() == 0 && c.superseded() == 0 && c.holds(other_ids, {}, true));
            // ... the capture is stopped (a request came in) or refused by RAM: the cache is exactly as it was
            CHECK(c.size() == 2 && c.holds(std::vector<int32_t>{1, 2, 3, 4, 60, 70}, {}, true));
        }
        // the capture ends: before_put, then put() - the superseded copy goes, the other agent stays, nothing evicted
        {
            ConversationCache c(1 << 20, 2);
            fill(c);
            CHECK(pa::before_capture(c, one, 0, true, next, {}, chain, true) == pa::Room::ok);
            SavedConversation img = image({1, 2, 3, 4, 60, 70, 80, 81});
            img.checkpoints = chain;
            CHECK(pa::before_put(c, img, 0, true));
            CHECK(c.put(std::move(img)));
            CHECK(c.size() == 2 && c.evictions() == 0 && c.superseded() == 1 && c.holds(other_ids, {}, true));
            CHECK(c.holds(next, {}, true));
        }
        // an unrelated idle slot: no room without an eviction - refused, nothing changed
        {
            ConversationCache c(1 << 20, 2);
            fill(c);
            const std::vector<int32_t> unrelated = {5, 5, 5, 5, 5};
            CHECK(pa::before_capture(c, one, 0, true, unrelated, {}, {cp({5, 5})}, true) == pa::Room::no_room_idle);
            CHECK(c.size() == 2 && c.evictions() == 0 && c.holds(other_ids, {}, true));
        }
        // the captured image came out larger than its estimate and no longer fits: before_put refuses, cache untouched
        {
            SavedConversation big = image({1, 2, 3, 4, 60, 70, 80, 81});
            big.checkpoints = chain;
            big.live.gdn.resize(4096);
            ConversationCache sizing(1 << 20, 8);
            fill(sizing);
            ConversationCache c(sizing.bytes() + one + one / 2, 8);
            fill(c);   // two entries; room for the estimate, not for the larger image
            CHECK(pa::before_capture(c, one, 0, true, next, {}, chain, true) == pa::Room::ok);
            CHECK(!pa::before_put(c, big, 0, true));
            CHECK(c.size() == 2 && c.evictions() == 0 && c.superseded() == 0);
        }
        // an admission's park (not idle) keeps the old order: make_room evicts the oldest up front, before_put says yes
        {
            ConversationCache c(1 << 20, 2);
            fill(c);
            CHECK(pa::before_capture(c, one, 0, false, next, {}, chain, true) == pa::Room::ok);
            CHECK(c.size() == 1 && c.evictions() == 1 && !c.holds(other_ids, {}, true));
            CHECK(pa::before_put(c, probe, 0, false));
            ConversationCache tiny(8, 2);
            CHECK(pa::before_capture(tiny, one, 0, false, next, {}, chain, true) == pa::Room::no_room);
        }
    }
    if (failed) { std::fprintf(stderr, "idle_park_test: %d failure(s)\n", failed); return 1; }
    std::printf("idle_park_test: PASS\n");
    return 0;
}
