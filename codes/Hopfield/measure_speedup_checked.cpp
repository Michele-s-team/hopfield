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
static inline unsigned long seed_pat  (int run) { return  555 + run; }

// Per-repetition offsets (rep 0 keeps the original seeds / draw sequence)
static inline unsigned long seed_init_rep(int run, int rep) { return seed_init(run) + 1000003UL * rep; }
static inline unsigned long seed_evol_rep(int run, int rep) { return seed_evol(run) +    7919UL * rep; }
static inline unsigned long seed_pat_rep (int run, int rep) { return seed_pat(run)  +  104729UL * rep; }

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
//   - reset(i)   restores the patterns AND the initial state number i (NOT
//                timed; it also records the "before evolution" snapshot),
//   - seed_of(i) gives the evolution seed of repetition i,
//   - fn         is the timed work (setup + sweeps),
//   - post(i)    records the "after evolution" snapshot (NOT timed).
// Each repetition has its own patterns, init and seed, and the same i is used
// for all 6 methods of a job, so the comparison stays paired.
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
    //cout << n << " rep done \n";
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
    vector<int>    sizes        = {/*16, 32,*/ 64};                                    // 3
    vector<double> alpha_vals   = {0.02, 0.05 /*0.1, 0.2 ,0.3*/};                     // 5
    //0.15, 0.25
    vector<double> temperatures = {0.2, 0.5, 1.0, 2.0, 5.0};                           // 5
    vector<double> rho_vals     = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};  // 10

    // Number of jobs = 3 * 5 * 5 * 10 = 750

    const int N_sweeps = 1 << 8;
    const string out_path = "../../results/Hopfield/timing_comparison_bis.csv";

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
    out << "N,rho,alpha,beta,n_sweeps,n_rep";
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
        const int    n_rep = 5;

        const bool save    = false;   // no disk I/O, it would bias the timings
        const int  freq    = 1;
        const int  burn_in = 0;

        gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

        #pragma omp critical(print_info)
        {
            cout << "[Hopfield 2D] run " << job.run << "/" << nb_runs
            << " | L=" << job.L << " (N=" << N << ")"
            << " | alpha=" << job.alpha << " (P=" << P << ")"
            << " | T=" << job.temperature << " (beta=" << beta << ")"
            << " | rho=" << rho
            << " | sweeps=" << N_sweeps
            << " | rep=" << n_rep
            << " | thread=" << omp_get_thread_num() << '\n';
        }

        BitsNeighbors  bits_nb(N, beta, N_sweeps, P);
        BitsOverlaps   bits_ov(N, beta, N_sweeps, P);
        HopfieldNoBits classic_nb_indep (N, beta, N_sweeps, P);
        HopfieldNoBits classic_ov_indep (N, beta, N_sweeps, P);
        HopfieldNoBits classic_nb_shared(N, beta, N_sweeps, P);
        HopfieldNoBits classic_ov_shared(N, beta, N_sweeps, P);

        // Config types (decayed, in case the getters return references)
        using CfgT      = decay_t<decltype(bits_nb.getSpinsConfig())>;
        using CfgBitsT  = decay_t<decltype(bits_nb.getSpinsConfigBits())>;
        using PatT      = decay_t<decltype(bits_nb.getPatternsBitsToCanonical())>;
        using PatBitsT  = decay_t<decltype(bits_nb.getPatternsBits())>;

        // check_equality_configs takes const vector<int>&
        static_assert(is_same<CfgT, vector<int>>::value,
                      "getSpinsConfig() is expected to return vector<int>");

        // ── Graph: built once, copied to the other models ────────────────────
        // (the graph stays fixed across repetitions: random regular graphs are
        //  statistically equivalent and self-averaging at this N)
        gsl_rng_set(ran, seed_graph(job.run));
        bits_nb.initNetwork_RegularRandom(rho, ran);
        copy_graph(bits_nb, bits_ov);
        copy_graph(bits_nb, classic_nb_indep);
        copy_graph(bits_nb, classic_ov_indep);
        copy_graph(bits_nb, classic_nb_shared);
        copy_graph(bits_nb, classic_ov_shared);

        // ── Patterns AND initial spin state: one pair per repetition ─────────
        // Everything is drawn on bits_nb, stored in both representations, and
        // later injected identically into the 6 models. Not timed.
        //   rep 0 : original sequence (seed_init_rep(run,0) -> spins,
        //           toCanonical, patterns drawn from the same stream), so rep 0
        //           reproduces the previous results exactly.
        //   rep i : same spin draw as before (seed_init_rep(run,i)), then the
        //           patterns are drawn from their own stream seed_pat_rep(run,i)
        vector<CfgT>     init_cfg(n_rep);
        vector<CfgBitsT> init_bits(n_rep);
        vector<PatT>     pat_cfg(n_rep);
        vector<PatBitsT> pat_bits(n_rep);

        for (int i = 0; i < n_rep; ++i) {
            gsl_rng_set(ran, seed_init_rep(job.run, i));
            bits_nb.initSpinsBits(ran);
            bits_nb.toCanonical();
            if (i > 0) gsl_rng_set(ran, seed_pat_rep(job.run, i));
            bits_nb.initPatternsBits(ran);

            init_cfg[i]  = bits_nb.getSpinsConfig();
            init_bits[i] = bits_nb.getSpinsConfigBits();
            pat_cfg[i]   = bits_nb.getPatternsBitsToCanonical();
            pat_bits[i]  = bits_nb.getPatternsBits();
        }

        // ── Snapshots for the consistency checks (all untimed) ───────────────
        // Each benchmark has its own model instance.
        vector<CfgT> start_classic_nb_indep(n_rep), start_classic_ov_indep(n_rep),
                     end_classic_nb_indep(n_rep),   end_classic_ov_indep(n_rep);
        vector<CfgT> start_classic_nb_shared(n_rep), start_classic_ov_shared(n_rep),
                     end_classic_nb_shared(n_rep),   end_classic_ov_shared(n_rep);
        vector<CfgT> start_bits_nb(n_rep), start_bits_ov(n_rep),
                     end_bits_nb(n_rep),   end_bits_ov(n_rep);

        // Reset helpers: restore the same patterns and initial state for
        // repetition i. Patterns first, then spins. All untimed.
        auto reset_classic_nb_indep = [&](int i) {
            classic_nb_indep.initPatternsFromConfig(pat_cfg[i]);
            classic_nb_indep.initSpinsFromConfig(init_cfg[i]);
            start_classic_nb_indep[i] = classic_nb_indep.getSpinsConfig();
        };
        auto reset_classic_ov_indep = [&](int i) {
            classic_ov_indep.initPatternsFromConfig(pat_cfg[i]);
            classic_ov_indep.initSpinsFromConfig(init_cfg[i]);
            start_classic_ov_indep[i] = classic_ov_indep.getSpinsConfig();
        };
        auto reset_classic_nb_shared = [&](int i) {
            classic_nb_shared.initPatternsFromConfig(pat_cfg[i]);
            classic_nb_shared.initSpinsFromConfig(init_cfg[i]);
            start_classic_nb_shared[i] = classic_nb_shared.getSpinsConfig();
        };
        auto reset_classic_ov_shared = [&](int i) {
            classic_ov_shared.initPatternsFromConfig(pat_cfg[i]);
            classic_ov_shared.initSpinsFromConfig(init_cfg[i]);
            start_classic_ov_shared[i] = classic_ov_shared.getSpinsConfig();
        };
        auto reset_bits_nb = [&](int i) {
            bits_nb.initPatternsFromConfigBits(pat_bits[i]);
            bits_nb.initSpinsFromConfigBits(init_bits[i]);
            bits_nb.toCanonical();
            start_bits_nb[i] = bits_nb.getSpinsConfig();
        };
        auto reset_bits_ov = [&](int i) {
            bits_ov.initPatternsFromConfigBits(pat_bits[i]);
            bits_ov.initSpinsFromConfigBits(init_bits[i]);
            bits_ov.toCanonical();
            start_bits_ov[i] = bits_ov.getSpinsConfig();
        };

        // Convert bitwise states to canonical form before reading snapshots.
        auto post_classic_nb_indep  = [&](int i) { end_classic_nb_indep[i]  = classic_nb_indep.getSpinsConfig(); };
        auto post_classic_ov_indep  = [&](int i) { end_classic_ov_indep[i]  = classic_ov_indep.getSpinsConfig(); };
        auto post_classic_nb_shared = [&](int i) { end_classic_nb_shared[i] = classic_nb_shared.getSpinsConfig(); };
        auto post_classic_ov_shared = [&](int i) { end_classic_ov_shared[i] = classic_ov_shared.getSpinsConfig(); };
        auto post_bits_nb = [&](int i) {
            bits_nb.toCanonical();
            end_bits_nb[i] = bits_nb.getSpinsConfig();
        };
        auto post_bits_ov = [&](int i) {
            bits_ov.toCanonical();
            end_bits_ov[i] = bits_ov.getSpinsConfig();
        };

        // ── Consistency checks (outside every timed region) ──────────────────
        const int prefix_width = 14;
        const int col_width    = 2;

        // The helper compares its second and third configurations.
        // The first argument is the reference initial configuration.
        auto check_pair = [&](const char* stage, const char* label, int rep,
                              const CfgT& a, const CfgT& b) -> bool {
            const bool ok = IO.check_equality_configs(a, b, N);
            if (!ok) {
                #pragma omp critical(print_check)
                {
                    cout << "[FAILED] run " << job.run
                         << " rep " << rep
                         << " | " << stage
                         << " | " << label << "\n";
                }
            }
            return ok;
        };

        bool all_ok = true;

        for (int i = 0; i < n_rep; ++i) {
            // Initialize all six models and record their initial configurations.
            reset_classic_nb_indep(i);
            reset_classic_ov_indep(i);
            reset_classic_nb_shared(i);
            reset_classic_ov_shared(i);
            reset_bits_nb(i);
            reset_bits_ov(i);

            // BEFORE evolution: five comparisons are sufficient to verify
            // that all six model instances start from the same configuration.
            all_ok &= check_pair("before", "NoBits indep neighbors vs indep overlaps",
                                 i, start_classic_nb_indep[i], start_classic_ov_indep[i]);

            all_ok &= check_pair("before", "NoBits independent neighbors vs shared neighbors",
                                 i, start_classic_nb_indep[i], start_classic_nb_shared[i]);

            all_ok &= check_pair("before", "NoBits shared neighbors vs shared overlaps",
                                 i, start_classic_nb_shared[i], start_classic_ov_shared[i]);

            all_ok &= check_pair("before", "NoBits shared overlaps vs Bits overlaps",
                                 i, start_classic_ov_shared[i], start_bits_ov[i]);

            all_ok &= check_pair("before", "Bits overlaps vs Bits neighbors",
                                 i, start_bits_ov[i], start_bits_nb[i]);
        }

        if (all_ok)
            cout << "=== SUCCESS: all tests passed (before evolution) ===\n";

        auto seed_of = [&](int i) { return seed_evol_rep(job.run, i); };

        cout << "\nRUNNING SIMULATIONS"<< endl;

        // ── 6 measurements, run in series, each repeated n_rep times ─────────
        // NoBits: IndependentRNG, then SharedRNG; Bits: neighbors and overlaps.
        const Stats stats_classic_nb_indep = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_classic_nb_indep, [&] {
            classic_nb_indep.runSweepsIndependentRNG_neighbors(ran, save, freq);
        }, post_classic_nb_indep));
        cout << "1/6"<< endl;
        const Stats stats_classic_ov_indep = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_classic_ov_indep, [&] {
            classic_ov_indep.runSweepsIndependentRNG_overlaps(ran, save, freq);
        }, post_classic_ov_indep));
        cout << "2/6"<< endl;
        const Stats stats_classic_nb_shared = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_classic_nb_shared, [&] {
            classic_nb_shared.runSweepsSharedRNG_neighbors(ran, save, freq);
        }, post_classic_nb_shared));
        cout << "3/6"<< endl;
        const Stats stats_classic_ov_shared = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_classic_ov_shared, [&] {
            classic_ov_shared.runSweepsSharedRNG_overlaps(ran, save, freq);
        }, post_classic_ov_shared));
        cout << "4/6"<< endl; 
        const Stats stats_bits_nb = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_bits_nb, [&] {
            bits_nb.runSweeps_neighbors(ran, save, freq, burn_in);
        }, post_bits_nb));
        cout << "5/6"<< endl;
        const Stats stats_bits_ov = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_bits_ov, [&] {
            bits_ov.runSweeps_overlaps(ran, save, freq, burn_in);
        }, post_bits_ov));
        cout << "6/6"<< endl;

        cout << "SIMULATIONS DONE"<< endl;


        all_ok = true;
        for (int i = 0; i < n_rep; ++i) {
            all_ok &= check_pair("after", "NoBits independent: neighbors vs overlaps",
                                 i, end_classic_nb_indep[i], end_classic_ov_indep[i]);
            all_ok &= check_pair("after", "Shared group: NoBits neighbors vs NoBits overlaps",
                                 i, end_classic_nb_shared[i], end_classic_ov_shared[i]);
            all_ok &= check_pair("after", "Shared group: NoBits overlaps vs Bits overlaps",
                                 i, end_classic_ov_shared[i], end_bits_ov[i]);
            all_ok &= check_pair("after", "Shared group: Bits overlaps vs Bits neighbors",
                                 i, end_bits_ov[i], end_bits_nb[i]);
        }

        if (all_ok)
            cout << "=== SUCCESS: all tests passed (after evolution) ===\n";

        // Speedups from medians (robust to outliers). The two bitwise timings
        // are the common references for the independent and shared NoBits runs.
        const double speedup_neighbors_indep  = stats_classic_nb_indep.median  / stats_bits_nb.median;
        const double speedup_overlaps_indep   = stats_classic_ov_indep.median  / stats_bits_ov.median;
        const double speedup_neighbors_shared = stats_classic_nb_shared.median / stats_bits_nb.median;
        const double speedup_overlaps_shared  = stats_classic_ov_shared.median / stats_bits_ov.median;

        // Exactly two final ratios: S_ov / S_nb, one for each NoBits RNG mode.
        const double speedup_ratio_indep  = speedup_overlaps_indep  / speedup_neighbors_indep;
        const double speedup_ratio_shared = speedup_overlaps_shared / speedup_neighbors_shared;

        #pragma omp critical(write_csv)
        {
            out << N << "," << rho << "," << job.alpha << "," << beta << ","
             << N_sweeps << "," << n_rep << ",";
            write_stats(out, stats_classic_nb_indep);  out << ",";
            write_stats(out, stats_classic_ov_indep);  out << ",";
            write_stats(out, stats_classic_nb_shared); out << ",";
            write_stats(out, stats_classic_ov_shared); out << ",";
            write_stats(out, stats_bits_nb);           out << ",";
            write_stats(out, stats_bits_ov);           out << ",";
            out << speedup_neighbors_indep << "," << speedup_overlaps_indep << ","
                << speedup_neighbors_shared << "," << speedup_overlaps_shared << ","
                << speedup_ratio_indep << "," << speedup_ratio_shared << "\n";
            out.flush();

            cout << "\nRun " << job.run << "/" << nb_runs
                 << " \n(alpha=" << job.alpha << ", N=" << N
                 << ", beta=" << beta << ", rho=" << rho << ")"
                 << " median: NoBits_indep[nb=" << stats_classic_nb_indep.median
                 << "s, ov=" << stats_classic_ov_indep.median << "s]"
                 << " NoBits_shared[nb=" << stats_classic_nb_shared.median
                 << "s, ov=" << stats_classic_ov_shared.median << "s]"
                 << " Bits[nb=" << stats_bits_nb.median
                 << "s, ov=" << stats_bits_ov.median << "s]"
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