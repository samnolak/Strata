// src/spec/swarm_schedule_test.cpp - SwarmSchedule (STRATA_SWARM_SPEC): the global verify schedule of the pipelined groups
//
//   1. no proposals known: every slot verifies its own token only;
//   2. one group (its trip through the stages is the round): a likely proposal is taken, an unlikely one is not (N = 0);
//   3. a saturated pipeline (more groups than stages: the stages' loads are the round): the swarm's threshold drops the
//      proposals of a slot that a slot alone would verify (DSpark: one threshold, not one per slot);
//   4. groups = stages, one slot each (the heaviest group's trip is the round): a lighter group's rows are free up to it;
//   5. every count stays within its slot's cap, its known probabilities and a group's 8 rows, and is a prefix;
//   6. the cost model: measured counts, a line through them for the others, one point scaled by the prior, a stall capped;
//   7. the calibration: proposals of a probability that never hold stop being verified;
//   8. the mode switch.
#include "strata/spec/swarm_schedule.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using strata::spec::SwarmSchedule;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
// every stage: 3 ms for a window of one row, +1 ms a row (E2's shape at Q6 is 3.3 + 1.1)
void seed(SwarmSchedule& s, double c0, double row) {
    for (int k = 0; k < s.stages(); ++k)
        for (int r = 1; r <= SwarmSchedule::kMaxRows; ++r)
            for (int i = 0; i < 5; ++i) s.observe_stage(k, r, c0 + row * (r - 1));
}
SwarmSchedule::Slot slot(int group, int cap, double p, int n_p) {
    SwarmSchedule::Slot x;
    x.group = group;
    x.cap = cap;
    x.n_p = n_p;
    for (int j = 0; j < n_p && j < SwarmSchedule::kMaxDrafts; ++j) x.p[j] = (float) p;
    return x;
}
}  // namespace

int main() {
    std::printf("swarm_schedule_test\n");
    {
        SwarmSchedule s(4);
        std::vector<SwarmSchedule::Slot> sl = {slot(0, 3, 0.9, 0), slot(1, 3, 0.9, 0)};
        int nd[2] = {7, 7};
        const auto p = s.plan(sl.data(), 2, 2, nd);
        check(nd[0] == 0 && nd[1] == 0 && p.taken == 0, "no probabilities known: no proposals, each slot its token");
    }
    {
        SwarmSchedule s(4);
        seed(s, 3.0, 1.0);
        int nd[1];
        std::vector<SwarmSchedule::Slot> hi = {slot(0, 3, 0.95, 3)};
        s.plan(hi.data(), 1, 1, nd);
        check(nd[0] >= 2, "one group, proposals at 0.95: two or more verified");
        std::vector<SwarmSchedule::Slot> lo = {slot(0, 3, 0.2, 3)};
        s.plan(lo.data(), 1, 1, nd);
        check(nd[0] == 0, "one group, proposals at 0.2: none verified (an honest N = 0)");
        std::vector<SwarmSchedule::Slot> mid = {slot(0, 3, 0.4, 3)};
        s.plan(mid.data(), 1, 1, nd);
        check(nd[0] == 1, "one group, proposals at 0.4: the first only (1.4/16 > 1/12 > 1.56/20)");
    }
    {
        // 2 stages, 4 groups of one slot: each stage serves all four windows in a round (the loads add up)
        SwarmSchedule s(2);
        seed(s, 3.0, 1.0);
        std::vector<SwarmSchedule::Slot> sl = {slot(0, 3, 0.95, 3), slot(1, 3, 0.95, 3), slot(2, 3, 0.4, 3),
                                               slot(3, 3, 0.4, 3)};
        int nd[4];
        const auto p = s.plan(sl.data(), 4, 4, nd);
        check(nd[0] >= 2 && nd[1] >= 2, "saturated pipeline: the likely slots keep their proposals");
        check(nd[2] == 0 && nd[3] == 0, "saturated pipeline: the 0.4 slots verify none (alone they would verify one)");
        int alone[1];
        std::vector<SwarmSchedule::Slot> one = {slot(0, 3, 0.4, 3)};
        SwarmSchedule s1(2);
        seed(s1, 3.0, 1.0);
        s1.plan(one.data(), 1, 1, alone);
        check(alone[0] == 1, "  (the same 0.4 slot alone in a 2-stage pipeline: one proposal)");
        check(p.threshold > 0.4, "saturated pipeline: the swarm's threshold is above 0.4");
    }
    {
        // 4 stages, 4 groups of one slot: a round is the heaviest group's trip, the lighter groups' rows ride along free
        SwarmSchedule s(4);
        seed(s, 3.0, 1.0);
        std::vector<SwarmSchedule::Slot> sl = {slot(0, 3, 0.95, 3), slot(1, 3, 0.95, 3), slot(2, 3, 0.25, 3),
                                               slot(3, 3, 0.25, 3)};
        int nd[4];
        s.plan(sl.data(), 4, 4, nd);
        check(nd[2] >= 1 && nd[3] >= 1, "groups = stages: the 0.25 slots get rows free under the heavy groups' trip");
        int alone[1];
        std::vector<SwarmSchedule::Slot> one = {slot(0, 3, 0.25, 3)};
        s.plan(one.data(), 1, 1, alone);
        check(alone[0] == 0, "  (the same 0.25 slot alone: none)");
    }
    {
        SwarmSchedule s(4);
        seed(s, 1.0, 0.0);   // rows cost nothing: everything allowed is taken
        std::vector<SwarmSchedule::Slot> sl = {slot(0, 2, 0.99, 7), slot(0, 7, 0.99, 7), slot(1, 7, 0.99, 3),
                                               slot(1, 0, 0.99, 7)};
        int nd[4];
        s.plan(sl.data(), 4, 2, nd);
        check(nd[0] == 2, "free rows: a slot stops at its cap (2)");
        check(nd[0] + nd[1] + 2 <= SwarmSchedule::kMaxRows, "free rows: a group's window stays within 8 rows");
        check(nd[2] == 3, "free rows: a slot stops at its known probabilities (3)");
        check(nd[3] == 0, "free rows: cap 0 (no draft ready, a stopping slot) takes none");
        // a slot whose second proposal is far less likely than its first: the prefix holds (never j = 1 without j = 0)
        SwarmSchedule t(1);
        seed(t, 3.0, 1.0);
        SwarmSchedule::Slot x = slot(0, 3, 0.9, 3);
        x.p[1] = 0.05f;
        int n1[1];
        t.plan(&x, 1, 1, n1);
        check(n1[0] == 1, "a likely first proposal, an unlikely second: the first only");
        SwarmSchedule::Slot y = slot(0, 3, 0.9, 3);
        y.p[0] = std::nanf("");
        t.plan(&y, 1, 1, n1);
        check(n1[0] == 0, "a NaN probability: no chance, and nothing after it");
    }
    {
        SwarmSchedule s(2);
        for (int i = 0; i < 10; ++i) { s.observe_stage(0, 1, 3.0); s.observe_stage(0, 3, 3.8); }
        check(std::fabs(s.stage_ms(0, 1) - 3.0) < 1e-9 && std::fabs(s.stage_ms(0, 3) - 3.8) < 1e-9, "cost: measured counts");
        check(std::fabs(s.stage_ms(0, 2) - 3.4) < 1e-6, "cost: between two counts, on their line (3.4)");
        check(std::fabs(s.stage_ms(0, 5) - 4.6) < 1e-6, "cost: past them, on their line (4.6)");
        check(std::fabs(s.stage_ms(1, 2) - 3.4) < 1e-6, "cost: a stage with nothing measured takes every stage's line");
        SwarmSchedule u(1);
        for (int i = 0; i < 4; ++i) u.observe_stage(0, 2, 5.0);
        check(std::fabs(u.stage_ms(0, 4) - 5.0 * 1.75 / 1.25) < 1e-6, "cost: one count measured, scaled by the prior shape");
        u.observe_stage(0, 2, 500.0);   // a capture or a stall
        check(u.stage_ms(0, 2) <= 7.0 + 1e-9, "cost: one stalled window moves the estimate at most kStall times its weight");
        SwarmSchedule v(1);
        for (int i = 0; i < 5; ++i) v.observe_draft_wait(2.0);
        check(std::fabs(v.draft_wait_ms() - 2.0) < 1e-9, "cost: the drafter chain's wait");
        int rows[2] = {1, 1};
        // (nothing measured: a window of one row is 1 unit; the stage serves 2 windows = 2, a group's trip 1 + 2 = 3)
        check(std::fabs(v.round_ms(rows, 2) - 3.0) < 1e-9, "cost: the round is the larger of the stage load and a trip");
        v.observe_stage(0, 1, 4.0);
        check(std::fabs(v.round_ms(rows, 2) - 8.0) < 1e-9, "cost: ... and the stage load once it is the larger");
    }
    {
        SwarmSchedule s(1);
        seed(s, 3.0, 1.0);
        SwarmSchedule::Slot x = slot(0, 3, 0.9, 3);
        int nd[1];
        s.plan(&x, 1, 1, nd);
        check(nd[0] >= 1, "calibration: before any verdict a 0.9 proposal is verified");
        const float p[3] = {0.9f, 0.9f, 0.9f};
        for (int i = 0; i < 400; ++i) s.observe_probs(p, 3, 0);
        s.plan(&x, 1, 1, nd);
        check(s.rate(0.9) < 0.1 && nd[0] == 0, "calibration: 0.9 proposals that never hold stop being verified");
    }
    {
        using strata::spec::swarm_spec_mode;
        check(swarm_spec_mode(nullptr) == 0 && swarm_spec_mode("") == 0 && swarm_spec_mode("0") == 0, "mode: unset / 0 = off");
        check(swarm_spec_mode("1") == 1 && swarm_spec_mode("2") == 2 && swarm_spec_mode("9") == 2, "mode: 1, 2 (above: 2)");
    }
    std::printf(g_fail ? "swarm_schedule_test: %d FAILED\n" : "swarm_schedule_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
