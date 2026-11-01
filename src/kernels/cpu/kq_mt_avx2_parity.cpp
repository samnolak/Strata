// src/kernels/cpu/kq_mt_avx2_parity.cpp - the multi-token AVX-2 Q6_K / Q8_0 expert rows (kq_mt_avx2.cpp) against
// ggml-cpu's single-token dot (the engine's default path for these formats), bit for bit, on random ggml-quantized
// weights: 1..16 tokens, whole and partial row ranges (odd starts and lengths), gate/up with SwiGLU and plain rows.
// Then through the engine's entry points (native_gu_rows / native_down_rows), which take the new rows only with
// STRATA_CPU_AVX2_MT=1: run it with and without that variable, both must print PASS.
//
//     kq_mt_avx2_parity                      parity (exit 0 = PASS, 1 = FAIL, 77 = no AVX2: skipped)
//     kq_mt_avx2_parity --bench [--mb 2048] [--reps 3] [--nt 1,2,3,4,6,8,12,16]
//                                            one thread, ggml's per-token dot vs the new rows: us per expert,
//                                            rows/s and GB/s of weight bytes, weights streamed from DRAM (cold) and
//                                            one L2/L3-resident expert (hot).  Pin it: taskset -c 36-47 nice -n 19.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_mt_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

const ggml_type_traits_cpu* traits(int t) { return ggml_get_type_traits_cpu((ggml_type) t); }

std::vector<uint8_t> quant_rows(int type, int rows, int n, std::mt19937& rng, float sigma) {
    std::normal_distribution<float> nd(0.f, sigma);
    std::vector<float> src((size_t) rows * n);
    for (auto& v : src) v = nd(rng);
    // a few large weights per row, as real experts have (they set the block scales)
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < 4; ++k) src[(size_t) r * n + pick(rng)] *= 8.f;
    std::vector<uint8_t> dst(ggml_row_size((ggml_type) type, n) * (size_t) rows);
    ggml_quantize_chunk((ggml_type) type, src.data(), dst.data(), 0, rows, n, nullptr);
    return dst;
}

std::vector<std::vector<uint8_t>> quant_acts(int act_type, int nt, int n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> a((size_t) nt);
    std::vector<float> x((size_t) n);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng);
        x[(size_t) (t * 37) % n] = 12.f;   // an outlier channel
        a[(size_t) t].resize(ggml_row_size((ggml_type) act_type, n));
        traits(act_type)->from_float(x.data(), a[(size_t) t].data(), n);
    }
    return a;
}

// ggml's per-token dot, the way native_gu_rows / native_down_rows run it by default
void ref_gu(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
            float* const* ff, int r0, int r1) {
    const ggml_vec_dot_t dot = traits(type)->vec_dot;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, blob + (size_t) r * gu_row, 0, act[t], 0, 1);
            dot(n, &u, 0, blob + up_off + (size_t) r * gu_row, 0, act[t], 0, 1);
            g *= 1.f;
            u *= 1.f;
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
}
void ref_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
              int r0, int r1) {
    const ggml_vec_dot_t dot = traits(type)->vec_dot;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, w + (size_t) r * row_bytes, 0, act[t], 0, 1);
            out[t][r] = s;
        }
}

int fails = 0;
long checks = 0;

void cmp(const char* what, const std::vector<std::vector<float>>& got, const std::vector<std::vector<float>>& want,
         int nt, int r0, int r1) {
    ++checks;
    for (int t = 0; t < nt; ++t)
        if (std::memcmp(got[(size_t) t].data() + r0, want[(size_t) t].data() + r0, sizeof(float) * (size_t) (r1 - r0)) != 0) {
            int r = r0;
            while (std::memcmp(&got[(size_t) t][(size_t) r], &want[(size_t) t][(size_t) r], 4) == 0) ++r;
            if (fails < 20)
                std::printf("FAIL %s nt=%d t=%d rows [%d,%d): row %d got %.9g want %.9g\n", what, nt, t, r0, r1, r,
                            got[(size_t) t][(size_t) r], want[(size_t) t][(size_t) r]);
            ++fails;
            return;
        }
}

// one geometry: n_embd x n_ff expert with gate/up type gu and down type dn (Flash-Next: 2560 x 640)
void parity_geometry(int gu, int dn, int n_embd, int n_ff, std::mt19937& rng) {
    const int gu_act = traits(gu)->vec_dot_type, d_act = traits(dn)->vec_dot_type;
    const size_t gu_row = ggml_row_size((ggml_type) gu, n_embd), d_row = ggml_row_size((ggml_type) dn, n_ff);
    std::vector<uint8_t> gate = quant_rows(gu, n_ff, n_embd, rng, 0.02f), up = quant_rows(gu, n_ff, n_embd, rng, 0.02f);
    std::vector<uint8_t> blob(gate.size() + up.size());
    std::memcpy(blob.data(), gate.data(), gate.size());
    std::memcpy(blob.data() + gate.size(), up.data(), up.size());
    const size_t up_off = gate.size();
    std::vector<uint8_t> down = quant_rows(dn, n_embd, n_ff, rng, 0.02f);
    constexpr int NTMAX = 16;
    auto a = quant_acts(gu_act, NTMAX, n_embd, rng);
    auto h = quant_acts(d_act, NTMAX, n_ff, rng);
    const void* ap[NTMAX];
    const void* hp[NTMAX];
    for (int t = 0; t < NTMAX; ++t) { ap[t] = a[(size_t) t].data(); hp[t] = h[(size_t) t].data(); }
    std::vector<std::vector<float>> want_ff(NTMAX, std::vector<float>((size_t) n_ff, -7.f)),
        got_ff(NTMAX, std::vector<float>((size_t) n_ff, -9.f)), want_o(NTMAX, std::vector<float>((size_t) n_embd, -7.f)),
        got_o(NTMAX, std::vector<float>((size_t) n_embd, -9.f));
    float* wf[NTMAX];
    float* gf[NTMAX];
    float* wo[NTMAX];
    float* go[NTMAX];
    for (int t = 0; t < NTMAX; ++t) {
        wf[t] = want_ff[(size_t) t].data(); gf[t] = got_ff[(size_t) t].data();
        wo[t] = want_o[(size_t) t].data(); go[t] = got_o[(size_t) t].data();
    }
    ref_gu(gu, blob.data(), gu_row, up_off, n_embd, ap, NTMAX, wf, 0, n_ff);
    ref_rows(dn, down.data(), d_row, n_ff, hp, NTMAX, wo, 0, n_embd);
    char what[96];
    const int gu_ranges[][2] = {{0, n_ff}, {1, n_ff - 2}, {3, 4}, {5, std::min(38, n_ff)}, {n_ff - 7, n_ff}};
    const int d_ranges[][2] = {{0, n_embd}, {1, n_embd - 2}, {3, 4}, {7, std::min(70, n_embd)}, {n_embd - 5, n_embd}};
    for (int nt = 1; nt <= NTMAX; ++nt) {
        for (const auto& rr : gu_ranges) {
            for (auto& v : got_ff) std::fill(v.begin(), v.end(), -9.f);
            cpu::kqmt_gu_rows(gu, blob.data(), gu_row, up_off, n_embd, ap, nt, gf, rr[0], rr[1]);
            std::snprintf(what, sizeof what, "gu %s %dx%d", ggml_type_name((ggml_type) gu), n_embd, n_ff);
            cmp(what, got_ff, want_ff, nt, rr[0], rr[1]);
        }
        for (const auto& rr : d_ranges) {
            for (auto& v : got_o) std::fill(v.begin(), v.end(), -9.f);
            cpu::kqmt_rows(dn, down.data(), d_row, n_ff, hp, nt, go, rr[0], rr[1]);
            std::snprintf(what, sizeof what, "down %s %dx%d", ggml_type_name((ggml_type) dn), n_ff, n_embd);
            cmp(what, got_o, want_o, nt, rr[0], rr[1]);
        }
        // a token's rows do not depend on which other tokens share the pass: the same token at another position
        if (nt >= 2) {
            const void* ap2[NTMAX];
            float* gf2[NTMAX];
            for (int t = 0; t < nt; ++t) { ap2[t] = ap[(t + 1) % nt]; gf2[t] = got_ff[(size_t) ((t + 1) % nt)].data(); }
            for (auto& v : got_ff) std::fill(v.begin(), v.end(), -9.f);
            cpu::kqmt_gu_rows(gu, blob.data(), gu_row, up_off, n_embd, ap2, nt, gf2, 0, n_ff);
            std::snprintf(what, sizeof what, "gu-rotated %s", ggml_type_name((ggml_type) gu));
            cmp(what, got_ff, want_ff, nt, 0, n_ff);
        }
    }
    // the engine entry points (STRATA_CPU_AVX2_MT decides which rows they take; both must equal ggml's)
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, dn, n_embd, n_ff, f, err)) { std::printf("FAIL native_fmt: %s\n", err.c_str()); ++fails; return; }
    std::vector<uint8_t> eblob(f.bytes);
    std::memcpy(eblob.data(), blob.data(), blob.size());
    std::memcpy(eblob.data() + f.down_off, down.data(), down.size());
    for (int nt = 1; nt <= 8; ++nt) {   // the pool's groups hold at most MAXT = 8 tokens
        for (auto& v : got_ff) std::fill(v.begin(), v.end(), -9.f);
        for (auto& v : got_o) std::fill(v.begin(), v.end(), -9.f);
        cpu::native_gu_rows(f, eblob.data(), ap, nt, gf, 0, n_ff);
        cpu::native_down_rows(f, eblob.data(), hp, nt, go, 0, n_embd);
        std::snprintf(what, sizeof what, "native_gu_rows %s", ggml_type_name((ggml_type) gu));
        cmp(what, got_ff, want_ff, nt, 0, n_ff);
        std::snprintf(what, sizeof what, "native_down_rows %s", ggml_type_name((ggml_type) dn));
        cmp(what, got_o, want_o, nt, 0, n_embd);
        for (auto& v : got_ff) std::fill(v.begin(), v.end(), -9.f);
        cpu::native_gu_rows_invariant(f, eblob.data(), ap, nt, gf, 7, n_ff - 3);
        std::snprintf(what, sizeof what, "native_gu_rows_invariant %s", ggml_type_name((ggml_type) gu));
        cmp(what, got_ff, want_ff, nt, 7, n_ff - 3);
    }
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::string str(s);
    size_t at = 0;
    while (at <= str.size()) {
        const size_t c = str.find(',', at);
        v.push_back(std::atoi(str.c_str() + at));
        if (c == std::string::npos) break;
        at = c + 1;
    }
    return v;
}

// one thread: `rows` x `n` matrices of `type` laid out back to back in `w` (`count` of them); every pass walks all of
// them (cold: far more than the caches) or the first one again and again (hot)
struct BenchCase {
    const char* name;
    int type;
    bool gu;        // gate/up pair (+SwiGLU) or plain rows
    int n, rows;    // dot length, rows per matrix
};

void bench(int mb, int reps, const std::vector<int>& nts) {
    std::mt19937 rng(99);
    const BenchCase cases[] = {{"Q6_K gate/up 2560->640x2", 14, true, 2560, 640},
                               {"Q8_0 down   640->2560", 8, false, 640, 2560},
                               {"Q8_0 gate/up 2560->640x2", 8, true, 2560, 640}};
    std::printf("bench: one thread, %d MB of weights per cold pass, best of %d; us per expert matrix set, Mrows/s (row x token dots), GB/s of weight bytes\n",
                mb, reps);
    std::printf("%-26s %3s %5s | %10s %8s %7s | %10s %8s %7s | %6s\n", "case", "nt", "mode", "ggml us", "Mrow/s", "GB/s",
                "new us", "Mrow/s", "GB/s", "x");
    for (const BenchCase& bc : cases) {
        const int act_type = traits(bc.type)->vec_dot_type;
        const size_t row_bytes = ggml_row_size((ggml_type) bc.type, bc.n);
        const size_t mat_bytes = row_bytes * (size_t) bc.rows * (bc.gu ? 2 : 1);
        const size_t count = std::max<size_t>(2, ((size_t) mb << 20) / mat_bytes);
        std::vector<uint8_t> one = quant_rows(bc.type, bc.rows * (bc.gu ? 2 : 1), bc.n, rng, 0.02f);
        std::vector<uint8_t> w(mat_bytes * count);
        for (size_t c = 0; c < count; ++c) std::memcpy(w.data() + c * mat_bytes, one.data(), mat_bytes);
        auto a = quant_acts(act_type, 16, bc.n, rng);
        const void* ap[16];
        for (int t = 0; t < 16; ++t) ap[t] = a[(size_t) t].data();
        std::vector<std::vector<float>> o(16, std::vector<float>((size_t) bc.rows));
        float* op[16];
        for (int t = 0; t < 16; ++t) op[t] = o[(size_t) t].data();
        for (const int nt : nts) {
            for (int hot = 0; hot < 2; ++hot) {
                const size_t passes = hot ? std::max<size_t>(4, count) : count;
                double best[2] = {1e30, 1e30};
                for (int rep = 0; rep < reps; ++rep)
                    for (int impl = 0; impl < 2; ++impl) {
                        const double t0 = now_s();
                        for (size_t c = 0; c < passes; ++c) {
                            const uint8_t* m = w.data() + (hot ? 0 : c * mat_bytes);
                            if (bc.gu) {
                                if (impl == 0) ref_gu(bc.type, m, row_bytes, row_bytes * bc.rows, bc.n, ap, nt, op, 0, bc.rows);
                                else cpu::kqmt_gu_rows(bc.type, m, row_bytes, row_bytes * bc.rows, bc.n, ap, nt, op, 0, bc.rows);
                            } else {
                                if (impl == 0) ref_rows(bc.type, m, row_bytes, bc.n, ap, nt, op, 0, bc.rows);
                                else cpu::kqmt_rows(bc.type, m, row_bytes, bc.n, ap, nt, op, 0, bc.rows);
                            }
                        }
                        best[impl] = std::min(best[impl], (now_s() - t0) / (double) passes);
                    }
                const double dots = (double) bc.rows * (bc.gu ? 2 : 1) * nt;
                std::printf("%-26s %3d %5s | %10.1f %8.2f %7.2f | %10.1f %8.2f %7.2f | %5.2fx\n", bc.name, nt, hot ? "hot" : "cold",
                            best[0] * 1e6, dots / best[0] * 1e-6, (double) mat_bytes / best[0] * 1e-9, best[1] * 1e6,
                            dots / best[1] * 1e-6, (double) mat_bytes / best[1] * 1e-9, best[0] / best[1]);
                std::fflush(stdout);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!cpu::cpu_avx2_ok()) {
        std::printf("kq_mt_avx2_parity: SKIP (no AVX2)\n");
        return 77;
    }
    ggml_cpu_init();
    bool do_bench = false;
    int mb = 2048, reps = 3;
    std::vector<int> nts = {1, 2, 3, 4, 6, 8, 12, 16};
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--bench") do_bench = true;
        else if (s == "--mb" && i + 1 < argc) mb = std::atoi(argv[++i]);
        else if (s == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (s == "--nt" && i + 1 < argc) nts = parse_list(argv[++i]);
    }
    if (do_bench) {
        bench(mb, reps, nts);
        return 0;
    }
    if (!(ggml_cpu_has_avx2() && ggml_cpu_has_fma() && !ggml_cpu_has_avx512())) {
        std::printf("kq_mt_avx2_parity: SKIP (this ggml-cpu is not the AVX2/FMA build the rows reproduce)\n");
        return 77;
    }
    const char* mt = std::getenv("STRATA_CPU_AVX2_MT");
    std::printf("kq_mt_avx2_parity: STRATA_CPU_AVX2_MT=%s\n", mt ? mt : "(unset)");
    std::mt19937 rng(20261101);
    for (int seed = 0; seed < 3; ++seed) {
        parity_geometry(14, 8, 2560, 640, rng);   // UD-Q6_K_XL: Q6_K gate/up, Q8_0 down
        parity_geometry(8, 8, 2560, 640, rng);    // its one Q8_0 layer
    }
    parity_geometry(14, 8, 512, 96, rng);         // small and odd geometries: 2 Q6_K blocks, 3 Q8_0 blocks
    parity_geometry(14, 8, 768, 160, rng);
    parity_geometry(8, 8, 96, 32, rng);
    std::printf("kq_mt_avx2_parity: %ld checks, %d failed: %s\n", checks, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
