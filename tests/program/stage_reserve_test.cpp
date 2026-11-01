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
    std::printf("stage_reserve_test: %ld cases, %s\n", cases, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
