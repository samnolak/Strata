// src/spec/swarm_schedule.cpp - see include/strata/spec/swarm_schedule.hpp.
#include "strata/spec/swarm_schedule.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::spec {
namespace {

constexpr double kAlpha = 0.1;      // EMA weight of a new window time (a running mean for the first samples)
constexpr double kStall = 3.0;      // a window time above this many times the known one counts as that much (a capture, a stall)

double shape(int rows) { return 1.0 + SwarmSchedule::kPriorSlope * (double) (rows - 1); }

}  // namespace

SwarmSchedule::SwarmSchedule(int n_stages)
    : n_stages_(std::clamp(n_stages, 1, kMaxStages)), cal_(DraftPolicy::kMaxT) {}

int swarm_spec_mode(const char* env) {
    if (env == nullptr || env[0] == '\0') return 0;
    return std::clamp(std::atoi(env), 0, 2);
}

void SwarmSchedule::observe_stage(int stage, int rows, double ms) {
    if (stage < 0 || stage >= n_stages_ || rows < 1 || rows > kMaxRows || !(ms > 0.0)) return;
    double& c = c_[(size_t) stage][(size_t) rows];
    double& n = n_[(size_t) stage][(size_t) rows];
    if (n >= 3.0) ms = std::min(ms, kStall * c);
    const double a = std::max(kAlpha, 1.0 / (n + 1.0));
    c = n > 0.0 ? (1.0 - a) * c + a * ms : ms;
    n += 1.0;
}

void SwarmSchedule::observe_draft_wait(double ms) {
    if (!(ms > 0.0)) return;
    if (d_n_ >= 3.0) ms = std::min(ms, kStall * d_);
    const double a = std::max(kAlpha, 1.0 / (d_n_ + 1.0));
    d_ = d_n_ > 0.0 ? (1.0 - a) * d_ + a * ms : ms;
    d_n_ += 1.0;
}

double SwarmSchedule::stage_ms(int stage, int rows) const {
    stage = std::clamp(stage, 0, n_stages_ - 1);
    rows = std::clamp(rows, 1, kMaxRows);
    if (n_[(size_t) stage][(size_t) rows] > 0.0) return c_[(size_t) stage][(size_t) rows];
    // the points: this stage's measured row counts, or (none yet) every stage's, as one average stage
    double sw = 0.0, sx = 0.0, sy = 0.0, cmin = 0.0;
    int distinct = 0, only = 0;
    auto gather = [&](int k0, int k1) {
        for (int k = k0; k < k1; ++k)
            for (int r = 1; r <= kMaxRows; ++r) {
                const double n = n_[(size_t) k][(size_t) r];
                if (n <= 0.0) continue;
                const double w = std::min(n, 20.0), y = c_[(size_t) k][(size_t) r];
                sw += w;
                sx += w * r;
                sy += w * y;
                cmin = cmin > 0.0 ? std::min(cmin, y) : y;
            }
    };
    gather(stage, stage + 1);
    int k0 = stage, k1 = stage + 1;
    if (sw <= 0.0) { k0 = 0; k1 = n_stages_; gather(k0, k1); }
    if (sw <= 0.0) return shape(rows);   // nothing measured anywhere: the prior's shape (the plan is scale-free)
    std::array<char, kMaxRows + 1> seen{};
    for (int k = k0; k < k1; ++k)
        for (int r = 1; r <= kMaxRows; ++r)
            if (n_[(size_t) k][(size_t) r] > 0.0 && !seen[(size_t) r]) { seen[(size_t) r] = 1; ++distinct; only = r; }
    const double mx = sx / sw, my = sy / sw;
    if (distinct < 2) return my * shape(rows) / shape(only);
    double sxy = 0.0, sxx = 0.0;
    for (int k = k0; k < k1; ++k)
        for (int r = 1; r <= kMaxRows; ++r) {
            const double n = n_[(size_t) k][(size_t) r];
            if (n <= 0.0) continue;
            const double w = std::min(n, 20.0);
            sxy += w * (r - mx) * (c_[(size_t) k][(size_t) r] - my);
            sxx += w * (r - mx) * (r - mx);
        }
    const double b = sxx > 0.0 ? std::max(0.0, sxy / sxx) : 0.0;
    return std::max(my + b * (rows - mx), 0.25 * cmin);
}

double SwarmSchedule::round_ms(const int* rows, int n_groups) const {
    double full = 0.0, trip = 0.0;
    for (int k = 0; k < n_stages_; ++k) {
        double load = 0.0;
        for (int g = 0; g < n_groups; ++g)
            if (rows[g] > 0) load += stage_ms(k, rows[g]);
        full = std::max(full, load);
    }
    for (int g = 0; g < n_groups; ++g) {
        if (rows[g] <= 0) continue;
        double t = draft_wait_ms();
        for (int k = 0; k < n_stages_; ++k) t += stage_ms(k, rows[g]);
        trip = std::max(trip, t);
    }
    return std::max({full, trip, 1e-9});
}

SwarmSchedule::Plan SwarmSchedule::plan(const Slot* slots, int n_slots, int n_groups, int* nd) const {
    Plan out;
    n_slots = std::clamp(n_slots, 0, kMaxSlots);
    n_groups = std::clamp(n_groups, 1, kMaxSlots);
    struct Cand { double a; int s; int j; };
    std::array<Cand, (size_t) kMaxSlots * kMaxDrafts> cand{};
    int nc = 0;
    std::array<int, kMaxSlots> rows{};
    double tokens = 0.0;
    for (int s = 0; s < n_slots; ++s) {
        nd[s] = 0;
        const int g = std::clamp(slots[s].group, 0, n_groups - 1);
        rows[(size_t) g] += 1;   // the slot's own token: always verified, one committed token
        tokens += 1.0;
        const int lim = std::min({slots[s].cap, slots[s].n_p, kMaxDrafts, kMaxRows - 1});
        double a = 1.0;
        for (int j = 0; j < lim; ++j) {
            a *= std::clamp(rate((double) slots[s].p[j]), 0.0, 1.0);
            cand[(size_t) nc++] = Cand{a, s, j};
        }
    }
    if (n_slots == 0) return out;
    // by survival; ties by slot then position, so a slot's proposals come in order (a is non-increasing in j)
    std::sort(cand.begin(), cand.begin() + nc, [](const Cand& x, const Cand& y) {
        if (x.a != y.a) return x.a > y.a;
        if (x.s != y.s) return x.s < y.s;
        return x.j < y.j;
    });
    double best_v = tokens / round_ms(rows.data(), n_groups);
    out.round_ms = round_ms(rows.data(), n_groups);
    out.tokens = tokens;
    out.tok_per_ms = best_v;
    out.threshold = 1.0;
    std::array<int, kMaxSlots> cnt{};
    std::array<char, kMaxSlots> blocked{};
    std::array<int, (size_t) kMaxSlots * kMaxDrafts> order{};   // the pairs taken, in order
    int n_taken = 0, best_n = 0;
    double run = tokens;
    for (int i = 0; i < nc; ++i) {
        const Cand& c = cand[(size_t) i];
        const int g = std::clamp(slots[c.s].group, 0, n_groups - 1);
        // (a slot's pairs must stay a prefix: one that cannot be taken blocks the rest of that slot)
        if (blocked[(size_t) c.s] || c.j != cnt[(size_t) c.s] || rows[(size_t) g] + 1 > kMaxRows) {
            blocked[(size_t) c.s] = 1;
            continue;
        }
        rows[(size_t) g] += 1;
        cnt[(size_t) c.s] += 1;
        run += c.a;
        order[(size_t) n_taken++] = i;
        const double rm = round_ms(rows.data(), n_groups);
        const double v = run / rm;
        if (v > best_v * (1.0 + 1e-9)) {
            best_v = v;
            best_n = n_taken;
            out.tok_per_ms = v;
            out.round_ms = rm;
            out.tokens = run;
            out.threshold = c.a;
        }
    }
    // the best prefix, then one pass over the pairs after it, in the same order, taking each that still raises the
    // tokens per ms: a round bound by one group's trip through the stages leaves rows of the lighter groups free, which
    // a prefix by survival alone reaches only together with the heavier group's costlier ones
    rows.fill(0);
    cnt.fill(0);
    blocked.fill(0);
    run = tokens;
    for (int s = 0; s < n_slots; ++s) rows[(size_t) std::clamp(slots[s].group, 0, n_groups - 1)] += 1;
    int last = -1;
    for (int t = 0; t < best_n; ++t) {
        const Cand& c = cand[(size_t) order[(size_t) t]];
        rows[(size_t) std::clamp(slots[c.s].group, 0, n_groups - 1)] += 1;
        cnt[(size_t) c.s] += 1;
        run += c.a;
        last = order[(size_t) t];
    }
    for (int i = last + 1; i < nc; ++i) {
        const Cand& c = cand[(size_t) i];
        const int g = std::clamp(slots[c.s].group, 0, n_groups - 1);
        if (blocked[(size_t) c.s] || c.j != cnt[(size_t) c.s] || rows[(size_t) g] + 1 > kMaxRows) {
            blocked[(size_t) c.s] = 1;
            continue;
        }
        rows[(size_t) g] += 1;
        const double rm = round_ms(rows.data(), n_groups);
        const double v = (run + c.a) / rm;
        if (v > best_v * (1.0 + 1e-9)) {
            cnt[(size_t) c.s] += 1;
            run += c.a;
            best_v = v;
            out.tok_per_ms = v;
            out.round_ms = rm;
            out.tokens = run;
            out.threshold = std::min(out.threshold, c.a);
            ++best_n;
        } else {
            rows[(size_t) g] -= 1;
            blocked[(size_t) c.s] = 1;   // (its later pairs are no likelier)
        }
    }
    for (int s = 0; s < n_slots; ++s) nd[s] = cnt[(size_t) s];
    out.taken = best_n;
    return out;
}

std::string SwarmSchedule::cost_text() const {
    std::string out;
    char b[96];
    for (int k = 0; k < n_stages_; ++k) {
        std::snprintf(b, sizeof b, "%sstage %d:", k ? "; " : "", k + 1);
        out += b;
        bool any = false;
        for (int r = 1; r <= kMaxRows; ++r)
            if (n_[(size_t) k][(size_t) r] > 0.0) {
                std::snprintf(b, sizeof b, " R%d %.2f(n=%.0f)", r, c_[(size_t) k][(size_t) r], n_[(size_t) k][(size_t) r]);
                out += b;
                any = true;
            }
        if (!any) out += " -";
    }
    std::snprintf(b, sizeof b, "; drafter chain %.2f ms (n=%.0f)", draft_wait_ms(), d_n_);
    out += b;
    return out;
}

}  // namespace strata::spec
