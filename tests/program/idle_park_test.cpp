// STRATA_IDLE_PARK's rules (include/strata/program/idle_park.hpp): which idle slot is parked when.  No GPU, no model.
#include <cstdio>
#include <cstdlib>

#include "strata/program/idle_park.hpp"

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
    if (failed) { std::fprintf(stderr, "idle_park_test: %d failure(s)\n", failed); return 1; }
    std::printf("idle_park_test: PASS\n");
    return 0;
}
