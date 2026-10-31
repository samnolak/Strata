// include/strata/spec/swarm_schedule.hpp - STRATA_SWARM_SPEC (opt-in): which MTP proposals the pipelined slot groups verify
//
// With STRATA_PIPED_MTP=N each slot of a pipelined group carries up to N proposals in its group's window; by default the
// first always, each further one while the draft layer was at least --spec-min-p sure of the one before - a rule per
// slot, tuned for the solo path.  With every stage busy (as many groups in flight as stages) the groups' window costs add
// up on each stage, so the best schedule is a single threshold over all slots, not one per slot: verify proposal (s, j)
// only while its chance to be committed is worth the row it costs at the throughput the whole swarm reaches.  This is
// DeepInfer's MTPVarlen::schedule (DSpark Alg. 1, csrc/engine/mtp_varlen.hpp:150-275, Apache-2.0; the idea, no code)
// over a layer split instead of a data-parallel rank:
//
//   1. every (slot s, proposal j) gets its survival  a(s,j) = prod_{i<=j} rate(p(s,i))   (rate: the measured acceptance of
//      an MTP proposal by its draft probability, DraftPolicy's calibration, shared by every slot: one drafter);
//   2. the pairs are sorted by a (ties: slot, then j - so each slot's proposals stay a prefix);
//   3. from "no proposals" (each slot its own token: one committed token each) the pairs are added one by one and the
//      prefix with the most expected tokens per ms of a round is taken, a round costing
//          T = max( max_k sum_g c_k(n_g),  max_g (sum_k c_k(n_g) + d) )
//      c_k(n): the measured time of a window of n rows on stage k (launch -> done; EMA per row count, other counts
//      interpolated), n_g: group g's rows, d: the drafter chain when it holds a group back.  The first term is a full
//      pipeline (every stage serves every group: the costs add), the second a group's own trip through the stages (a
//      pipeline with fewer groups than stages), so one rule covers both.
//
// It only chooses which proposals are verified: every verified row is judged as before, so greedy (and Philox-sampled)
// tokens are the same as without it.  CPU only; tested in swarm_schedule_test.
#pragma once

#include "strata/spec/draft_policy.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace strata::spec {

class SwarmSchedule {
public:
    static constexpr int kMaxStages = 8;
    static constexpr int kMaxRows = 8;     // a group's window (kVerifyNarrowT)
    static constexpr int kMaxDrafts = 7;   // per slot
    static constexpr int kMaxSlots = 64;

    explicit SwarmSchedule(int n_stages);

    // ---- cost model
    /// A group window of `rows` rows took `ms` on `stage` (from its launch returning to the poll that saw it done).
    void observe_stage(int stage, int rows, double ms);
    /// A group waited `ms` for its slots' drafter chains after its window left the last stage.
    void observe_draft_wait(double ms);
    /// The modelled time of a window of `rows` rows on `stage`: measured, else interpolated from that stage's measured
    /// row counts (a line through them, slope >= 0), else scaled from one by the prior shape 1 + kPriorSlope (rows - 1).
    double stage_ms(int stage, int rows) const;
    double draft_wait_ms() const { return d_n_ > 0 ? d_ : 0.0; }

    // ---- calibration (DraftPolicy::observe_probs / draft_rate: the measured acceptance by draft probability)
    void observe_probs(const float* p, int n_verified, int accepted);
    double rate(double p) const { return cal_.draft_rate(p); }

    // ---- the plan
    struct Slot {
        int group = 0;                     ///< its pipelined group (0 .. n_groups - 1)
        int cap = 0;                       ///< the most proposals it may carry now (0 = none)
        int n_p = 0;                       ///< draft probabilities known (p[0 .. n_p))
        float p[kMaxDrafts] = {};
    };
    struct Plan {
        double tok_per_ms = 0.0;           ///< the modelled throughput of the chosen schedule
        double round_ms = 0.0;
        double tokens = 0.0;
        double threshold = 0.0;            ///< the survival of the last pair taken (1 = none taken)
        int taken = 0;                     ///< proposals taken over all slots
    };
    /// The proposals each slot verifies (`nd[s]`, s < n_slots) under the global schedule.  Slots of every active group take
    /// part (the threshold is the swarm's); the caller applies the counts of the group it launches.
    Plan plan(const Slot* slots, int n_slots, int n_groups, int* nd) const;
    /// The modelled round time for `rows[g]` rows in group g (0 = a group with no window: not counted).
    double round_ms(const int* rows, int n_groups) const;

    /// "stage 1: R1 3.10(n=40) R2 3.52(n=12) ...; draft wait 0.40 ms (n=7)" for the log.
    std::string cost_text() const;
    int stages() const { return n_stages_; }

    static constexpr double kPriorSlope = 0.25;   // a row more, relative to a window of one (E2: AR-Q4 ~0.4 ms on ~3 ms)

private:
    int n_stages_;
    std::array<std::array<double, kMaxRows + 1>, kMaxStages> c_{}, n_{};
    double d_ = 0.0, d_n_ = 0.0;
    DraftPolicy cal_;
};

/// STRATA_SWARM_SPEC: 0 off (the default: exactly the rule above), 1 the global schedule, 2 also each slot's next drafter
/// chain only one step past the proposals it verified this window (the drafter on the last stage's card does less).
int swarm_spec_mode(const char* env);

}  // namespace strata::spec
