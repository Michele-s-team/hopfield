#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <omp.h>

using namespace std;

#include "gsl_rng.h"
#include "gsl_math.h"
#include "gsl_sf_log.h"
#include "gsl_randist.h"

#include "main.hpp"
#include "int.hpp"

#include "hopfield_model.hpp"
#include "hopfield_nobits.hpp"
#include "hopfield_bits.hpp"

SimulationIO IO;

using BitsNeighbors = HopfieldBits;   // bitwise, neighbor-coupling formulation
using BitsOverlaps  = HopfieldBits;   // bitwise, shifted-overlap formulation

// Seeds shared by all models of a given job (same seed => same draws)
static inline unsigned long seed_graph(int run) { return 1000 + run; }
static inline unsigned long seed_init (int run) { return  123 + run; }
static inline unsigned long seed_evol (int run) { return   42 + run; }

// Per-repetition offsets (rep 0 keeps the original seeds)
static inline unsigned long seed_init_rep(int run, int rep) { return seed_init(run) + 1000003UL * rep; }
static inline unsigned long seed_evol_rep(int run, int rep) { return seed_evol(run) +    7919UL * rep; }

// Copy the graph (neighbors, non_neighbors, degrees) from src to dst
template <class Src, class Dst>
void copy_graph(const Src& src, Dst& dst) {
    dst.initNeighbors(src.getNeighbors());
    dst.initNonNeighbors(src.getNonNeighbors());
    dst.initDegrees(src.getDegrees());
}

// Time a callable (seconds). The RNG is reseeded right before the run.
template <class F>
double timed_run(gsl_rng* ran, unsigned long seed, F&& fn) {
    gsl_rng_set(ran, seed);
    auto t0 = chrono::steady_clock::now();
    fn();
    auto t1 = chrono::steady_clock::now();
    return chrono::duration<double>(t1 - t0).count();
}

// Repeat a measurement n times. For repetition i:
//   - reset(i)   restores the initial state number i (NOT timed; it also
//                records the "before evolution" snapshot),
//   - seed_of(i) gives the evolution seed of repetition i,
//   - fn         is the timed work (setup + sweeps),
//   - post(i)    records the "after evolution" snapshot (NOT timed).
// Each repetition has its own init and seed, and the same i is used for
// all 6 methods of a job, so the comparison stays paired.
template <class Reset, class SeedOf, class F, class Post>
vector<double> repeat_timed(int n, gsl_rng* ran, SeedOf&& seed_of,
                            Reset&& reset, F&& fn, Post&& post) {
    vector<double> t;
    t.reserve(n);
    for (int i = 0; i < n; ++i) {
        reset(i);
        t.push_back(timed_run(ran, seed_of(i), fn));
        post(i);
    }
    return t;
}

struct Stats { double mean, min, max, median; };

Stats compute_stats(vector<double> v) {
    sort(v.begin(), v.end());
    const size_t n = v.size();
    Stats s;
    s.min    = v.front();
    s.max    = v.back();
    s.mean   = accumulate(v.begin(), v.end(), 0.0) / n;
    s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return s;
}

void write_stats(ostream& o, const Stats& s) {
    o << s.mean << "," << s.min << "," << s.max << "," << s.median;
}

int main() {
    // ── Scanned parameters ────────────────────────────────────────────────────
    vector<int>    sizes        = {16//, 32, 64
        };                                        // 3
    vector<double> alpha_vals   = {0.02, 0.05, 0.1, 0.2, 0.3};                         // 5
    vector<double> temperatures = {0.2, 0.5, 1.0, 2.0, 5.0};                           // 5
    vector<double> rho_vals     = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};  // 10

    // Number of jobs = 3 * 5 * 5 * 10 = 750

    const int N_sweeps = 1 << 10;
    const string out_path = "../../results/Hopfield/timing_comparison.csv";

    struct Job { int L; double alpha, temperature, rho; int run; };
    vector<Job> jobs;
    int run = 0;
    for (int L : sizes)
        for (double alpha : alpha_vals)
            for (double T : temperatures)
                for (double rho : rho_vals)
                    jobs.push_back({L, alpha, T, rho, ++run});

    const int nb_runs = jobs.size();

    // Header: four NoBits methods and two Bits methods; 4 stats per method
    ofstream out(out_path);
    out << "N,rho,alpha,P,beta,k_eff,n_sweeps,n_rep";
    for (const char* m : {"classic_indep_neighbors", "classic_indep_overlaps",
                          "classic_shared_neighbors", "classic_shared_overlaps",
                          "bitwise_neighbors", "bitwise_overlaps"})
        for (const char* s : {"mean", "min", "max", "median"})
            out << ",t_" << m << "_" << s;
    out << ",speedup_neighbors_indep,speedup_overlaps_indep"
           ",speedup_neighbors_shared,speedup_overlaps_shared"
           ",speedup_ratio_overlaps_over_neighbors_indep"
           ",speedup_ratio_overlaps_over_neighbors_shared\n";
    out << setprecision(10);
    out.flush();

    auto start_all = chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int job_id = 0; job_id < nb_runs; ++job_id) {

        const Job& job = jobs[job_id];
        const int    N    = job.L * job.L;
        const double beta = 1.0 / job.temperature;
        const int    P    = static_cast<int>(round(N * job.alpha));
        const double rho  = job.rho;
        const int    n_rep = 7;

        const bool save    = false;   // no disk I/O, it would bias the timings
        const int  freq    = 1;
        const int  burn_in = 0;

        gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

        #pragma omp critical(print_info)
        {
            cout << "\n============ Hopfield 2D Model ============\n";
            cout << "  Run:         " << job.run << "/" << nb_runs << "\n";
            cout << "  Lattice:     " << job.L << " x " << job.L << " = " << N << "\n";
            cout << "  Alpha:       " << job.alpha << " (P = " << P << ")\n";
            cout << "  Temperature: " << job.temperature << " (beta = " << beta << ")\n";
            cout << "  Rho:         " << rho << "\n";
            cout << "  Sweeps:      " << N_sweeps << "\n";
            cout << "  Repetitions: " << n_rep << "\n";
            cout << "  Thread:      " << omp_get_thread_num() << "\n";
            cout << "============================================\n" << flush;
        }

        BitsNeighbors  bits_nb(N, beta, N_sweeps, P);
        BitsOverlaps   bits_ov(N, beta, N_sweeps, P);
        HopfieldNoBits classic_cn_i(N, beta, N_sweeps, P);
        HopfieldNoBits classic_co_i(N, beta, N_sweeps, P);
        HopfieldNoBits classic_cn_s(N, beta, N_sweeps, P);
        HopfieldNoBits classic_co_s(N, beta, N_sweeps, P);

        // Config types (decayed, in case the getters return references)
        using CfgT     = decay_t<decltype(bits_nb.getSpinsConfig())>;
        using CfgBitsT = decay_t<decltype(bits_nb.getSpinsConfigBits())>;

        // check_equality_configs takes const vector<int>&
        static_assert(is_same<CfgT, vector<int>>::value,
                      "getSpinsConfig() is expected to return vector<int>");

        // ── Graph: built once, copied to the other models ────────────────────
        gsl_rng_set(ran, seed_graph(job.run));
        bits_nb.initNetwork_RegularRandom(rho, ran);
        copy_graph(bits_nb, bits_ov);
        copy_graph(bits_nb, classic_cn_i);
        copy_graph(bits_nb, classic_co_i);
        copy_graph(bits_nb, classic_cn_s);
        copy_graph(bits_nb, classic_co_s);

        const int k_eff = bits_nb.getDegrees()[0];

        // ── Patterns: drawn once (same disorder for all reps) ────────────────
        // Rep 0 uses the original sequence (spins, canonical, patterns).
        gsl_rng_set(ran, seed_init_rep(job.run, 0));
        bits_nb.initSpinsBits(ran);
        bits_nb.toCanonical();
        bits_nb.initPatternsBits(ran);

        auto patterns      = bits_nb.getPatternsBitsToCanonical();
        auto patterns_bits = bits_nb.getPatternsBits();

        bits_ov.initPatternsFromConfigBits(patterns_bits);
        classic_cn_i.initPatternsFromConfig(patterns);
        classic_co_i.initPatternsFromConfig(patterns);
        classic_cn_s.initPatternsFromConfig(patterns);
        classic_co_s.initPatternsFromConfig(patterns);

        // ── One initial spin state per repetition ────────────────────────────
        // Drawn on bits_nb, stored in both representations. Not timed.
        vector<CfgT>     init_cfg(n_rep);
        vector<CfgBitsT> init_bits(n_rep);

        init_cfg[0]  = bits_nb.getSpinsConfig();
        init_bits[0] = bits_nb.getSpinsConfigBits();
        for (int i = 1; i < n_rep; ++i) {
            gsl_rng_set(ran, seed_init_rep(job.run, i));
            bits_nb.initSpinsBits(ran);
            bits_nb.toCanonical();
            init_cfg[i]  = bits_nb.getSpinsConfig();
            init_bits[i] = bits_nb.getSpinsConfigBits();
        }

        // ── Snapshots for the consistency checks (all untimed) ───────────────
        // Each benchmark has its own model instance.
        vector<CfgT> start_cn_i(n_rep), start_co_i(n_rep), end_cn_i(n_rep), end_co_i(n_rep);
        vector<CfgT> start_cn_s(n_rep), start_co_s(n_rep), end_cn_s(n_rep), end_co_s(n_rep);
        vector<CfgT> start_bn_cfg(n_rep), start_bo_cfg(n_rep), end_bn_cfg(n_rep), end_bo_cfg(n_rep);

        // Reset helpers: restore the same initial state for repetition i.
        auto reset_cn_i = [&](int i) {
            classic_cn_i.initSpinsFromConfig(init_cfg[i]);
            start_cn_i[i] = classic_cn_i.getSpinsConfig();
        };
        auto reset_co_i = [&](int i) {
            classic_co_i.initSpinsFromConfig(init_cfg[i]);
            start_co_i[i] = classic_co_i.getSpinsConfig();
        };
        auto reset_cn_s = [&](int i) {
            classic_cn_s.initSpinsFromConfig(init_cfg[i]);
            start_cn_s[i] = classic_cn_s.getSpinsConfig();
        };
        auto reset_co_s = [&](int i) {
            classic_co_s.initSpinsFromConfig(init_cfg[i]);
            start_co_s[i] = classic_co_s.getSpinsConfig();
        };
        auto reset_nb = [&](int i) {
            bits_nb.initSpinsFromConfigBits(init_bits[i]);
            bits_nb.toCanonical();
            start_bn_cfg[i] = bits_nb.getSpinsConfig();
        };
        auto reset_ov = [&](int i) {
            bits_ov.initSpinsFromConfigBits(init_bits[i]);
            bits_ov.toCanonical();
            start_bo_cfg[i] = bits_ov.getSpinsConfig();
        };

        // Convert bitwise states to canonical form before reading snapshots.
        auto post_cn_i = [&](int i) { end_cn_i[i] = classic_cn_i.getSpinsConfig(); };
        auto post_co_i = [&](int i) { end_co_i[i] = classic_co_i.getSpinsConfig(); };
        auto post_cn_s = [&](int i) { end_cn_s[i] = classic_cn_s.getSpinsConfig(); };
        auto post_co_s = [&](int i) { end_co_s[i] = classic_co_s.getSpinsConfig(); };
        auto post_nb = [&](int i) {
            bits_nb.toCanonical();
            end_bn_cfg[i] = bits_nb.getSpinsConfig();
        };
        auto post_ov = [&](int i) {
            bits_ov.toCanonical();
            end_bo_cfg[i] = bits_ov.getSpinsConfig();
        };

                // ── Consistency checks (outside every timed region) ──────────────────
        const int prefix_width = 14;
        const int col_width    = 2;

        // The helper compares its second and third configurations.
        // The first argument is the reference initial configuration.
        auto check_pair = [&](const char* stage, const char* label, int rep,
                      const CfgT& a, const CfgT& b) {
            if (!IO.check_equality_configs(a, b, N)) {
                #pragma omp critical(print_check)
                {
                    cout << "[FAILED] run " << job.run
                        << " rep " << rep
                        << " | " << stage
                        << " | " << label << "\n";
                }
            }
        };

        for (int i = 0; i < n_rep; ++i) {
            // Initialize all six models and record their initial configurations.
            reset_cn_i(i);
            reset_co_i(i);
            reset_cn_s(i);
            reset_co_s(i);
            reset_nb(i);
            reset_ov(i);

            // BEFORE evolution: five comparisons are sufficient to verify
            // that all six model instances start from the same configuration.
            check_pair("before", "NoBits indep neighbors vs indep overlaps",
                    i, start_cn_i[i], start_co_i[i]);

            check_pair("before", "NoBits independent neighbors vs shared neighbors",
                    i, start_cn_i[i], start_cn_s[i]);

            check_pair("before", "NoBits shared neighbors vs shared overlaps",
                    i, start_cn_s[i], start_co_s[i]);

            check_pair("before", "NoBits shared overlaps vs Bits overlaps",
                    i, start_co_s[i], start_bo_cfg[i]);

            check_pair("before", "Bits overlaps vs Bits neighbors",
                    i, start_bo_cfg[i], start_bn_cfg[i]);
        }

        auto seed_of = [&](int i) { return seed_evol_rep(job.run, i); };

        // ── 6 measurements, run in series, each repeated n_rep times ─────────
        // NoBits: IndependentRNG, then SharedRNG; Bits: neighbors and overlaps.
        const Stats s_cn_i = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_cn_i, [&] {
            classic_cn_i.runSweepsIndependentRNG_neighbors(ran, save, freq);
        }, post_cn_i));
        const Stats s_co_i = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_co_i, [&] {
            classic_co_i.runSweepsIndependentRNG_overlaps(ran, save, freq);
        }, post_co_i));
        const Stats s_cn_s = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_cn_s, [&] {
            classic_cn_s.runSweepsSharedRNG_neighbors(ran, save, freq);
        }, post_cn_s));
        const Stats s_co_s = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_co_s, [&] {
            classic_co_s.runSweepsSharedRNG_overlaps(ran, save, freq);
        }, post_co_s));
        const Stats s_bn = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_nb, [&] {
            bits_nb.runSweeps_neighbors(ran, save, freq, burn_in);
        }, post_nb));
        const Stats s_bo = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_ov, [&] {
            bits_ov.runSweeps_overlaps(ran, save, freq, burn_in);
        }, post_ov));

        for (int i = 0; i < n_rep; ++i) {
            check_pair("after", "NoBits independent: neighbors vs overlaps",
                    i, end_cn_i[i], end_co_i[i]);

            check_pair("after", "Shared group: NoBits neighbors vs NoBits overlaps",
                    i, end_cn_s[i], end_co_s[i]);

            check_pair("after", "Shared group: NoBits overlaps vs Bits overlaps",
                    i, end_co_s[i], end_bo_cfg[i]);

            check_pair("after", "Shared group: Bits overlaps vs Bits neighbors",
                    i, end_bo_cfg[i], end_bn_cfg[i]);
        }

        // Speedups from medians (robust to outliers). The two bitwise timings
        // are the common references for the independent and shared NoBits runs.
        const double speedup_neighbors_indep = s_cn_i.median / s_bn.median;
        const double speedup_overlaps_indep  = s_co_i.median / s_bo.median;
        const double speedup_neighbors_shared = s_cn_s.median / s_bn.median;
        const double speedup_overlaps_shared  = s_co_s.median / s_bo.median;

        // Exactly two final ratios: S_ov / S_nb, one for each NoBits RNG mode.
        const double speedup_ratio_indep = speedup_overlaps_indep / speedup_neighbors_indep;
        const double speedup_ratio_shared = speedup_overlaps_shared / speedup_neighbors_shared;

        #pragma omp critical(write_csv)
        {
            out << N << "," << rho << "," << job.alpha << "," << P << "," << beta << ","
                << k_eff << "," << N_sweeps << "," << n_rep << ",";
            write_stats(out, s_cn_i); out << ",";
            write_stats(out, s_co_i); out << ",";
            write_stats(out, s_cn_s); out << ",";
            write_stats(out, s_co_s); out << ",";
            write_stats(out, s_bn); out << ",";
            write_stats(out, s_bo); out << ",";
            out << speedup_neighbors_indep << "," << speedup_overlaps_indep << ","
                << speedup_neighbors_shared << "," << speedup_overlaps_shared << ","
                << speedup_ratio_indep << "," << speedup_ratio_shared << "\n";
            out.flush();

            cout << "Run " << job.run << "/" << nb_runs
                 << " (alpha=" << job.alpha << ", N=" << N << ", P=" << P
                 << ", k=" << k_eff << ", beta=" << beta << ", rho=" << rho << ")"
                 << " median: NoBits_indep[nb=" << s_cn_i.median << "s, ov=" << s_co_i.median << "s]"
                 << " NoBits_shared[nb=" << s_cn_s.median << "s, ov=" << s_co_s.median << "s]"
                 << " Bits[nb=" << s_bn.median << "s, ov=" << s_bo.median << "s]"
                 << " | ratios: indep=" << speedup_ratio_indep
                 << ", shared=" << speedup_ratio_shared << "\n";
        }

        gsl_rng_free(ran);
    }

    auto end_all = chrono::steady_clock::now();
    cout << "\nTotal wall time: "
         << chrono::duration<double>(end_all - start_all).count() << " s\n";
    return 0;
}