// tests/program/stage_reserve_test.cpp - #3702: the later-stage reserve arithmetic of the layer split search.
// reserve_mib / room_bytes must equal generate.cpp's `stage_room` expression as it was written there; the proposal
// functions are pinned too (they are not used by the default placement).  p55/final2-c: the per-device reserve
// (--vram-reserve-per-dev / STRATA_VRAM_RESERVE_PER_DEV) - its parser, its lookup, and that an unset list changes nothing.
#include "strata/program/stage_reserve.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace strata::program::stage_reserve;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}

// the expression as it stood in generate.cpp before the move (kWindowMib, kDrafterMib, kPipeSnapMib, kPipeWindowMib)
int64_t legacy_room(int64_t fb, int64_t vram_reserve_mib, int64_t vram_reserve_later_mib, int64_t pf, bool later, bool drafter,
                    int64_t pipeline_windows) {
    const int64_t kWin = 96, kDrafter = 1000;
    const int64_t kPipeWindow = pipeline_windows > 0 ? 160 : 0;
    const int64_t kPipeSnap = pipeline_windows >= 2 ? 96 : 0;
    const int64_t base_reserve = later && vram_reserve_later_mib >= 0 ? vram_reserve_later_mib : vram_reserve_mib;
    const int64_t reserve = (base_reserve + pf + (later ? kWin : kPipeSnap) + kPipeWindow + (drafter ? kDrafter : 0)) << 20;
    return std::max<int64_t>(fb - reserve, 0);
}
}  // namespace

int main() {
    // the same numbers as the old expression over a grid of settings
    long cases = 0;
    for (int64_t fb : {int64_t(0), int64_t(500) << 20, int64_t(3000) << 20, int64_t(15) << 30})
        for (int64_t vr : {int64_t(0), int64_t(700), int64_t(1280)})
            for (int64_t vl : {int64_t(-1), int64_t(0), int64_t(1400)})
                for (int64_t pf : {int64_t(0), int64_t(160 + 4096 * 680 / 1024)})
                    for (int later = 0; later < 2; ++later)
                        for (int dr = 0; dr < 2; ++dr)
                            for (int64_t pw : {int64_t(0), int64_t(1), int64_t(2)}) {
                                Inputs in;
                                in.base_reserve_mib = later && vl >= 0 ? vl : vr;
                                in.prompt_buffers_mib = pf;
                                in.later = later != 0;
                                in.drafter = dr != 0;
                                in.pipe_window_mib = pw > 0 ? 160 : 0;
                                in.pipe_snap_mib = pw >= 2 ? 96 : 0;
                                ++cases;
                                check(room_bytes(fb, in) == legacy_room(fb, vr, vl, pf, later != 0, dr != 0, pw),
                                      "room differs from the old expression (fb " + std::to_string(fb >> 20) + " MiB, reserve " +
                                          std::to_string(vr) + "/" + std::to_string(vl) + ", later " + std::to_string(later) + ")");
                            }
    // the configuration of #3702: default reserves (700), borrowed prompt buffers, CUDA1 = the last card with the drafter
    {
        Inputs in;
        in.base_reserve_mib = 700;
        in.later = true;
        in.drafter = true;
        check(reserve_mib(in) == 700 + 96 + 1000, "the last card of #3702 keeps 1796 MiB planned");
        in.drafter = false;
        check(reserve_mib(in) == 796, "a later card without the drafter keeps 796 MiB planned");
        in.later = false;
        check(reserve_mib(in) == 700, "the first card keeps the base reserve alone");
        in.pipe_snap_mib = 96;
        check(reserve_mib(in) == 796, "the first card books the pipeline's second GDN state");
    }
    // p/mtp-q8: a Q8_0 drafter (rt/experts.txt) books its experts' extra MiB on the drafter's stage only; 0 = the Q2_0 numbers
    {
        Inputs in;
        in.base_reserve_mib = 700;
        in.later = true;
        in.drafter = true;
        in.drafter_extra_mib = 1875;
        check(reserve_mib(in) == 700 + 96 + 1000 + 1875, "a Q8_0 drafter's stage keeps 3671 MiB planned");
        in.drafter = false;
        check(reserve_mib(in) == 796, "the extra drafter MiB never land on a stage without the drafter");
    }
    // the same arithmetic with --vram-reserve-later-mib 1400 (the workaround that worked in #3702)
    {
        Inputs in;
        in.base_reserve_mib = 1400;
        in.later = true;
        check(reserve_mib(in) == 1496, "the workaround reserve: 1400 + 96");
    }
    // the proposal: the window term grows with the number of window sizes (--spec 4 -> windows of 1..6 tokens)
    check(proposed_window_mib(1) == 96 + 48, "proposal: one window size = the arena and the burst");
    check(proposed_window_mib(6) == 96 + 5 * 6 + 48, "proposal: six sizes = 174 MiB");
    check(proposed_window_mib(0) == proposed_window_mib(1), "proposal: no negative sizes");
    {
        Inputs in;
        in.base_reserve_mib = 700;
        in.later = true;
        check(proposed_reserve_mib(in, 6) == 700 + 174, "proposed reserve on a later card, six sizes");
        in.later = false;
        check(proposed_reserve_mib(in, 6) == reserve_mib(in), "the first card is not changed by the proposal");
    }
    // ---- p55/final2-c: the per-device reserve (--vram-reserve-per-dev / STRATA_VRAM_RESERVE_PER_DEV)
    // not given (empty list): base_reserve_for is the old expression on every device, so stage_room's room is too
    {
        const std::vector<int64_t> none;
        for (int dev = -1; dev < 6; ++dev)
            for (int64_t vr : {int64_t(0), int64_t(700), int64_t(4608)})
                for (int64_t vl : {int64_t(-1), int64_t(0), int64_t(1400)})
                    for (int later = 0; later < 2; ++later)
                        for (int64_t fb : {int64_t(0), int64_t(3000) << 20, int64_t(26) << 30}) {
                            Inputs in;
                            in.base_reserve_mib = base_reserve_for(dev < 0 ? 0 : dev, later != 0, vr, vl, none);
                            in.later = later != 0;
                            in.pipe_window_mib = 160;
                            ++cases;
                            check(in.base_reserve_mib == (later && vl >= 0 ? vl : vr), "unset list: the old base reserve");
                            check(room_bytes(fb, in) == legacy_room(fb, vr, vl, 0, later != 0, false, 1),
                                  "unset list: the room equals the old expression (dev " + std::to_string(dev) + ")");
                        }
    }
    // parsing
    {
        std::vector<int64_t> v;
        std::string e;
        check(parse_per_dev("3584,2816,2816,6656", v, e) && v == std::vector<int64_t>{3584, 2816, 2816, 6656}, "parse: four items");
        check(parse_per_dev(" 3072 , 2048,2048 ,4608 ", v, e) && v == std::vector<int64_t>{3072, 2048, 2048, 4608}, "parse: spaces");
        check(parse_per_dev("-,2048,,6144", v, e) && v == std::vector<int64_t>{-1, 2048, -1, 6144}, "parse: - and empty keep a device");
        check(parse_per_dev("0", v, e) && v == std::vector<int64_t>{0}, "parse: 0 is a reserve");
        check(parse_per_dev("1048576", v, e) && v.size() == 1, "parse: the largest figure");
        check(!parse_per_dev("1048577", v, e) && v.empty() && !e.empty(), "parse: above 1 TiB refused");
        check(!parse_per_dev("", v, e) && v.empty(), "parse: empty refused");
        check(!parse_per_dev(",,-", v, e) && v.empty(), "parse: nothing given refused");
        check(!parse_per_dev("3072,-5,2048", v, e) && v.empty(), "parse: a negative figure refused");
        check(!parse_per_dev("3072,2k", v, e) && v.empty(), "parse: a unit refused");
        check(!parse_per_dev("3.5", v, e) && v.empty(), "parse: a fraction refused");
        check(!parse_per_dev("99999999999999999999", v, e) && v.empty(), "parse: an overflowing figure refused");
        check(!parse_per_dev("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", v, e) && v.empty(), "parse: 17 items refused");
        check(parse_per_dev("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16", v, e) && v.size() == 16, "parse: 16 items");
    }
    // lookup: the list's value wins on its device; a device without one (-1, past the end) keeps the old rule
    {
        const std::vector<int64_t> pd{3584, 2816, -1, 6656};
        check(base_reserve_for(0, false, 4608, -1, pd) == 3584, "per-dev: CUDA0 its own");
        check(base_reserve_for(1, true, 4608, -1, pd) == 2816, "per-dev: CUDA1 its own");
        check(base_reserve_for(2, true, 4608, -1, pd) == 4608, "per-dev: CUDA2 not given -> --vram-reserve-mib");
        check(base_reserve_for(2, true, 4608, 1800, pd) == 1800, "per-dev: CUDA2 not given -> --vram-reserve-later-mib");
        check(base_reserve_for(3, true, 4608, 1800, pd) == 6656, "per-dev: the list beats --vram-reserve-later-mib");
        check(base_reserve_for(4, true, 4608, -1, pd) == 4608, "per-dev: past the list -> the old rule");
        check(base_reserve_for(-1, false, 4608, -1, pd) == 4608, "per-dev: a negative ordinal -> the old rule");
        check(per_dev_value(pd, 2) == -1 && per_dev_value(pd, 9) == -1 && per_dev_value(pd, 3) == 6656, "per-dev: per_dev_value");
        check(base_reserve_for(1, true, 4608, -1, std::vector<int64_t>{-1, 0}) == 0, "per-dev: 0 is a reserve, not 'unset'");
    }
    // the 4x RTX 4080 32 GB of this fork (display on CUDA3), K=11,24,37 at reserve 4608 measured: CUDA1 26.75 GiB free ->
    // room 22.00 GiB (the 96 MiB windows + 160 MiB --pipeline-windows on top of the base).  With 2816 there the room grows by
    // exactly 4608 - 2816 MiB, and with 6656 on CUDA3 it shrinks by 6656 - 4608: the per-device value moves the base and
    // nothing else.
    {
        const std::vector<int64_t> pd{3584, 2816, 2816, 6656};
        const int64_t fb1 = (int64_t) (26.75 * 1073741824.0);
        Inputs a;
        a.later = true;
        a.pipe_window_mib = 160;
        a.pipe_snap_mib = 96;
        a.base_reserve_mib = base_reserve_for(1, true, 4608, -1, std::vector<int64_t>{});
        Inputs b = a;
        b.base_reserve_mib = base_reserve_for(1, true, 4608, -1, pd);
        check(reserve_mib(a) == 4608 + 96 + 160, "4080 rig: CUDA1 keeps 4864 MiB at reserve 4608");
        check(room_bytes(fb1, b) - room_bytes(fb1, a) == (int64_t) (4608 - 2816) << 20, "4080 rig: CUDA1 gains 1792 MiB");
        Inputs c = a;
        c.base_reserve_mib = base_reserve_for(3, true, 4608, -1, pd);
        check(room_bytes(fb1, a) - room_bytes(fb1, c) == (int64_t) (6656 - 4608) << 20, "4080 rig: CUDA3 gives 2048 MiB");
        Inputs d = a;
        d.later = false;
        d.base_reserve_mib = base_reserve_for(0, false, 4608, -1, pd);
        check(reserve_mib(d) == 3584 + 96 + 160, "4080 rig: CUDA0 first stage: base + the second GDN state + the second verifier");
        // a reserve larger than the free memory leaves no room, never a negative one
        Inputs z = a;
        z.base_reserve_mib = base_reserve_for(0, true, 0, -1, std::vector<int64_t>{40000});
        check(room_bytes(fb1, z) == 0, "4080 rig: an oversized reserve -> room 0");
    }
    // ---- generate.cpp's flow (apply_primary at start-up, then device_reserve in the start-up log, stage_room and the
    // serve-time LOW check): a later device without an item falls back to the FLAG's --vram-reserve-mib, never to the
    // primary card's item (review: 13058d48 overwrote o.vram_reserve_mib with CUDA0's item and stage_room then used it)
    {
        // the reviewer's repro: "3584,-,-,6656" with --vram-reserve-mib 4608: CUDA1/2 keep 4608 (the log said 4608,
        // stage_room used 3584)
        std::vector<int64_t> pd;
        std::string e;
        check(parse_per_dev("3584,-,-,6656", pd, e), "flow: parse the repro list");
        int vr = 4608;
        bool given = true;
        const int64_t flag = apply_primary(pd, 0, vr, given);
        check(vr == 3584 && given && flag == 4608, "flow: CUDA0's item replaces --vram-reserve-mib, the flag is kept");
        check(device_reserve(-1, false, 0, vr, flag, -1, pd) == 3584, "flow: stage_room(-1) = CUDA0's item");
        check(device_reserve(0, false, 0, vr, flag, -1, pd) == 3584, "flow: CUDA0 = its item");
        check(device_reserve(1, true, 0, vr, flag, -1, pd) == 4608, "flow: CUDA1 '-' -> the flag 4608, not CUDA0's 3584");
        check(device_reserve(2, true, 0, vr, flag, -1, pd) == 4608, "flow: CUDA2 '-' -> the flag 4608");
        check(device_reserve(3, true, 0, vr, flag, -1, pd) == 6656, "flow: CUDA3 its item");
        check(device_reserve(5, true, 0, vr, flag, -1, pd) == 4608, "flow: past the end -> the flag");
        check(device_reserve(1, true, 0, vr, flag, 1800, pd) == 1800, "flow: CUDA1 '-' -> --vram-reserve-later-mib when given");
        // an item that is not given for the primary card: nothing is overridden, the auto paths may still adapt it
        int vr2 = 700;
        bool given2 = false;
        const int64_t flag2 = apply_primary(std::vector<int64_t>{-1, 2816}, 0, vr2, given2);
        check(flag2 == -1 && vr2 == 700 && !given2, "flow: no CUDA0 item -> nothing replaced, not 'given'");
        vr2 = 512;   // e.g. the expert cache auto path adapted the reserve (generate.cpp ~7250)
        check(device_reserve(-1, false, 0, vr2, flag2, -1, std::vector<int64_t>{-1, 2816}) == 512, "flow: adapted CUDA0 kept");
        check(device_reserve(2, true, 0, vr2, flag2, -1, std::vector<int64_t>{-1, 2816}) == 512,
              "flow: no override -> later devices follow the (adapted) --vram-reserve-mib as before");
        check(device_reserve(1, true, 0, vr2, flag2, -1, std::vector<int64_t>{-1, 2816}) == 2816, "flow: CUDA1 its item");
        // STRATA_PRIMARY_DEVICE=2: the primary's item is item 2, and stage_room's dev -1 means CUDA2, not CUDA0
        int vr3 = 4608;
        bool given3 = false;
        const std::vector<int64_t> pd3{1000, 2000, 3000, -1};
        const int64_t flag3 = apply_primary(pd3, 2, vr3, given3);
        check(vr3 == 3000 && flag3 == 4608 && given3, "flow: primary CUDA2 -> its item 3000");
        check(device_reserve(-1, false, 2, vr3, flag3, -1, pd3) == 3000, "flow: primary CUDA2: stage_room(-1) = item 2");
        check(device_reserve(3, true, 2, vr3, flag3, -1, pd3) == 4608, "flow: primary CUDA2: CUDA3 '-' -> the flag");
        check(device_reserve(0, true, 2, vr3, flag3, -1, pd3) == 1000, "flow: primary CUDA2: CUDA0 as a later stage = item 0");
        // the grid: for every list / flag / later flag / primary, device_reserve = the item when given, else the old rule
        // with the flag's value (and with no list at all, exactly the old expression with the current vram_reserve_mib)
        const std::vector<std::vector<int64_t>> lists{{}, {3584}, {3584, -1, -1, 6656}, {-1, 2816, 2816, 6656},
                                                      {3584, 2816, 2816, 6656}, {0, -1}, {-1, -1, 5000}};
        for (const auto& L : lists)
            for (int prim = 0; prim < 3; ++prim)
                for (int fl : {0, 700, 4608})
                    for (int64_t vl : {int64_t(-1), int64_t(0), int64_t(1400)})
                        for (int dev = -1; dev < 6; ++dev)
                            for (int later = 0; later < 2; ++later) {
                                if (dev >= 0 && dev == prim && later) continue;   // the primary card is never a later stage
                                int v = fl;
                                bool gv = false;
                                const int64_t fm = apply_primary(L, prim, v, gv);
                                const int d = dev < 0 ? prim : dev;
                                const int64_t item = d < (int) L.size() ? L[(size_t) d] : -1;
                                const int64_t want = item >= 0 ? item : (later && vl >= 0 ? vl : fl);
                                ++cases;
                                check(device_reserve(dev, later != 0, prim, v, fm, vl, L) == want,
                                      "flow grid: dev " + std::to_string(dev) + " primary " + std::to_string(prim) + " flag " +
                                          std::to_string(fl) + " later " + std::to_string(later));
                                if (L.empty())
                                    check(device_reserve(dev, later != 0, prim, v, fm, vl, L) == (later && vl >= 0 ? vl : v),
                                          "flow grid: unset list = the old expression");
                            }
    }
    std::printf("stage_reserve_test: %ld cases, %s\n", cases, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
