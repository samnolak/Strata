// include/strata/program/stage_reserve.hpp - what the layer split keeps free on a card when it sizes the expert cache.
//
// This is the arithmetic of `stage_room` in src/program/generate.cpp, moved here unchanged so a host test can pin it
// (tests/program/stage_reserve_test.cpp). `reserve_mib` is exactly the sum that function subtracted from the card's
// free memory before; the default placement does not change.
//
// #3702: on an unequal pair (RTX 3060 + RTX 5060 Ti) the last card was left with 68 MiB after loading although
// 700 + 96 MiB were reserved for it, and the verify windows' graphs (captured on first use, 4-6 MiB each and once 42 MiB)
// then ran it out of memory. `proposed_window_mib` below is a PROPOSAL for a larger window term; nothing calls it
// (it needs the model gate in phaseA-tests/gh027/gates/3702-later-reserve.sh before it can become the default).
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program::stage_reserve {

constexpr int64_t kWindowMib = 96;       ///< the verify windows on a later stage; 75 MiB measured, rounded up
constexpr int64_t kDrafterMib = 1000;    ///< the MTP drafter (839 MiB) + the head, on the last stage only

struct Inputs {
    int64_t base_reserve_mib = 0;   ///< base_reserve_for(): the per-device value when given, else --vram-reserve-later-mib on a later stage when given, else --vram-reserve-mib
    int64_t prompt_buffers_mib = 0; ///< the prompt path's own buffers (0 when they are borrowed from the cache)
    bool later = false;             ///< a stage after the first
    bool drafter = false;           ///< the stage that carries the MTP drafter and the head
    int64_t pipe_window_mib = 0;    ///< --pipeline-windows: a second verifier on every stage
    int64_t pipe_snap_mib = 0;      ///< --pipeline-windows 2 or more: the first stage's second GDN state
    int64_t extra_snap_mib = 0;     ///< --pipeline-windows 3 on more than two stages: a later front stage's snapshots
    int64_t drafter_extra_mib = 0;  ///< p/mtp-q8: the drafter's experts beyond kDrafterMib's Q2_0 (Q8_0: +1,875 MiB), drafter stage only
};

/// What the plan keeps free on the card, in MiB (generate.cpp's `reserve` before the shift).
inline int64_t reserve_mib(const Inputs& in) {
    return in.base_reserve_mib + in.prompt_buffers_mib + (in.later ? kWindowMib : in.pipe_snap_mib) + in.pipe_window_mib +
           (in.drafter ? kDrafterMib + in.drafter_extra_mib : 0) + in.extra_snap_mib;
}

/// The room the cache may use: the free bytes less the reserve, never below 0.
inline int64_t room_bytes(int64_t free_bytes, const Inputs& in) {
    return std::max<int64_t>(free_bytes - (reserve_mib(in) << 20), 0);
}

// ---------------------------------------------------------------------------------------------------- per device
// p55/final2-c (opt-in): one base reserve per CUDA device, `--vram-reserve-per-dev m0,m1,..` or
// STRATA_VRAM_RESERVE_PER_DEV="m0,m1,..", indexed by the CUDA ordinal in the engine's order (CUDA_VISIBLE_DEVICES as the
// server set it).  The two global flags can only give the first card one value and every later card another; a box
// whose display sits on one of the later cards (here CUDA3) wants a large reserve there and small ones elsewhere.
// An empty item or "-" keeps that device's reserve as before, and so does every device past the end of the list.
// Not given (the list empty): base_reserve_for() is the old expression exactly (pinned by stage_reserve_test).

constexpr size_t kPerDevMax = 16;              ///< items accepted in the list (CUDA ordinals 0..15)
constexpr int64_t kPerDevMaxMib = 1 << 20;     ///< 1 TiB in MiB: a larger figure is a typo, not a reserve

/// Parses "3584,2816,2816,6656" (spaces allowed around items; "" or "-" = not given -> -1).  At least one item must
/// give a value.  On failure `err` says what is wrong and `out` is left empty.
inline bool parse_per_dev(const std::string& s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::vector<int64_t> v;
    bool any = false;
    size_t a = 0;
    while (true) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        std::string t = s.substr(a, b - a);
        const size_t f = t.find_first_not_of(" \t");
        t = f == std::string::npos ? std::string() : t.substr(f, t.find_last_not_of(" \t") - f + 1);
        if (t.empty() || t == "-") {
            v.push_back(-1);
        } else {
            if (t.find_first_not_of("0123456789") != std::string::npos || t.size() > 7) {
                err = "item " + std::to_string(v.size()) + " (\"" + t + "\") is not a whole number of MiB";
                return false;
            }
            const int64_t mib = std::stoll(t);
            if (mib > kPerDevMaxMib) {
                err = "item " + std::to_string(v.size()) + " (" + t + " MiB) is larger than " + std::to_string(kPerDevMaxMib);
                return false;
            }
            v.push_back(mib);
            any = true;
        }
        if (v.size() > kPerDevMax) {
            err = "more than " + std::to_string(kPerDevMax) + " items";
            return false;
        }
        if (b == s.size()) break;
        a = b + 1;
    }
    if (!any) {
        err = "no item gives a reserve (\"" + s + "\")";
        return false;
    }
    out = std::move(v);
    return true;
}

/// The value the list gives device `dev`, or -1 (not given, past the end, a negative ordinal).
inline int64_t per_dev_value(const std::vector<int64_t>& per_dev, int dev) {
    return dev >= 0 && (size_t) dev < per_dev.size() ? per_dev[(size_t) dev] : -1;
}

/// The base reserve (Inputs::base_reserve_mib) of device `dev` running a first (`later` false) or a later stage: the
/// list's value for it when it gives one, else the old rule - a later stage --vram-reserve-later-mib when that was
/// given (>= 0), else --vram-reserve-mib.
inline int64_t base_reserve_for(int dev, bool later, int64_t vram_reserve_mib, int64_t vram_reserve_later_mib,
                                const std::vector<int64_t>& per_dev) {
    const int64_t mine = per_dev_value(per_dev, dev);
    if (mine >= 0) return mine;
    return later && vram_reserve_later_mib >= 0 ? vram_reserve_later_mib : vram_reserve_mib;
}

// generate.cpp's flow, factored here so stage_reserve_test covers it.  At start-up the primary card's item (when the list
// gives one) replaces --vram-reserve-mib, so every first-card sizing path follows it; the flag's own value is kept apart
// (Options::vram_reserve_flag_mib) because it stays the fallback of every LATER device whose item is empty, "-" or past
// the end - not the primary card's item.

/// The start-up step: the primary card's item when the list gives one.  `vram_reserve_mib` / `given` are updated in place
/// and the flag's value before the override is returned (the fallback for later devices), or -1 when nothing was
/// overridden (then `vram_reserve_mib` still is the flag's value, also after the auto paths adapt it).
inline int64_t apply_primary(const std::vector<int64_t>& per_dev, int primary_device, int& vram_reserve_mib, bool& given) {
    const int64_t first = per_dev_value(per_dev, primary_device);
    if (first < 0) return -1;
    const int64_t flag = vram_reserve_mib;
    vram_reserve_mib = (int) first;
    given = true;
    return flag;
}

/// --vram-reserve-mib as the flag (or the default / the auto paths) gave it: `flag_mib` when apply_primary() overrode
/// it (>= 0), else the current `vram_reserve_mib`.
inline int64_t flag_reserve(int64_t vram_reserve_mib, int64_t flag_mib) {
    return flag_mib >= 0 ? flag_mib : vram_reserve_mib;
}

/// The base reserve generate.cpp uses for device `dev` (-1 = the primary card, as stage_room's first stage passes it):
/// the primary card on a first stage keeps the current `vram_reserve_mib` (its item, or the flag); every other device
/// takes base_reserve_for() with the FLAG's value as the fallback, never the primary card's item.  With an empty list
/// this is exactly the old expression (later && later_mib >= 0 ? later_mib : vram_reserve_mib).
inline int64_t device_reserve(int dev, bool later, int primary_device, int64_t vram_reserve_mib, int64_t flag_mib,
                              int64_t vram_reserve_later_mib, const std::vector<int64_t>& per_dev) {
    const int d = dev < 0 ? primary_device : dev;
    if (d == primary_device && !later) return vram_reserve_mib;
    return base_reserve_for(d, later, flag_reserve(vram_reserve_mib, flag_mib), vram_reserve_later_mib, per_dev);
}

// ---------------------------------------------------------------------------------------------------- the proposal
constexpr int64_t kGraphMib = 6;            ///< one window graph, captured on first use (log of #3702: 4, 6, 6)
constexpr int64_t kInstantiateBurstMib = 48; ///< the largest single step seen (42 MiB in #3702): room kept for one more

/// PROPOSAL, unused: the window term of a later stage when the windows of every size 1..max_window_t can still be
/// captured after loading: the first window's arena (kWindowMib already holds it), a graph for each further size, and
/// one burst.
inline int64_t proposed_window_mib(int max_window_t) {
    const int64_t sizes = std::max<int64_t>(max_window_t, 1);
    return kWindowMib + (sizes - 1) * kGraphMib + kInstantiateBurstMib;
}

/// PROPOSAL, unused: reserve_mib with the proposed window term on a later stage.
inline int64_t proposed_reserve_mib(const Inputs& in, int max_window_t) {
    Inputs p = in;
    int64_t r = reserve_mib(p);
    if (in.later) r += proposed_window_mib(max_window_t) - kWindowMib;
    return r;
}

}  // namespace strata::program::stage_reserve
