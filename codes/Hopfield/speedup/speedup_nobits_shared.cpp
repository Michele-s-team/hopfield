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

// Seeds shared by all models of a given job (same seed => same draws)
static inline unsigned long seed_graph(int run) { return 1000 + run; }
static inline unsigned long seed_init (int run) { return  123 + run; }
static inline unsigned long seed_evol (int run) { return   42 + run; }
static inline unsigned long seed_pat  (int run) { return  555 + run; }

// Per-repetition seeds (rep 0 keeps the original seeds / draw sequence)
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

struct Stats {
    double mean, min, max, median;
};

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

struct Job {
    int L;
    double alpha, temperature, rho;
    int run;
    int n_sweeps;
};

// Benchmark: NoBits neighbors vs NoBits overlaps, Shared RNG.
// Patterns and initial state change with each repetition i, but are shared
// by the two models for a given i. Ratio = t_neighbors / t_overlaps (medians).
// A HopfieldBits object is used only as a (untimed) generator of the graph,
// the patterns and the initial spin states, as in the full benchmark.

// Measurement protocol:
//   - one untimed warm-up run per method (caches, frequency, first touch),
//   - the two methods are interleaved repetition by repetition, and the order
//     alternates (even i: nb then ov, odd i: ov then nb), so that slow drifts
//     of the machine load and order effects affect both methods equally.
//     Use an even n_rep for a perfectly balanced design.

using NoBitsNeighbors = HopfieldNoBits;
using NoBitsOverlaps  = HopfieldNoBits;

int main() {

    cout << unitbuf;

    vector<int>    sizes        = {32};
    vector<double> alpha_vals   = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6};
    vector<double> temperatures = {0.1, 1, 5};
    vector<double> rho_vals     = {0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};

    // Number of sweeps to benchmark
    const vector<int> N_sweeps_vals = {
        1 << 10,
        1 << 11,
        1 << 12,
        1 << 13
    };

    const int n_rep = 5;

    const string out_path = "../../../results/Hopfield/speedup_nobits_shared.csv";

    // Build the list of jobs
    vector<Job> jobs;
    int run = 0;

    for (int L : sizes) {
        for (double alpha : alpha_vals) {
            for (double T : temperatures) {
                for (double rho : rho_vals) {

                    for (int N_sweeps : N_sweeps_vals) {
                        ++run;
                        jobs.push_back({
                            L,
                            alpha,
                            T,
                            rho,
                            run,
                            N_sweeps
                        });
                    }
                }
            }
        }
    }

    const int nb_runs = jobs.size();

    // Open output CSV
    ofstream out(out_path);

    if (!out) {
        cerr << "ERROR: cannot open " << out_path
             << " (does the directory exist?)\n";
        return 1;
    }

    out << "N,rho,alpha,beta,n_sweeps,n_rep";

    for (const char* m : {"classic_shared_neighbors", "classic_shared_overlaps"}) {
        for (const char* s : {"mean", "min", "max", "median"}) {
            out << ",t_" << m << "_" << s;
        }
    }

    out << ",ratio_neighbors_over_overlaps\n";
    out << setprecision(10);
    out.flush();

    auto start_all = chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int job_id = 0; job_id < nb_runs; ++job_id) {

        const Job& job = jobs[job_id];

        const int    N         = job.L * job.L;
        const double beta      = 1.0 / job.temperature;
        const int    P         = static_cast<int>(round(N * job.alpha));
        const double rho       = job.rho;
        const int    N_sweeps  = job.n_sweeps;

        const bool save    = false;
        const int  freq    = 1;

        gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

        #pragma omp critical(print_info)
        {
            cout << "[NoBits shared nb vs ov] run " << job.run << "/" << nb_runs
                 << " | N=" << N
                 << " | alpha=" << job.alpha << " (P=" << P << ")"
                 << " | T=" << job.temperature
                 << " | rho=" << rho
                 << " | N_sweeps=" << N_sweeps
                 << " | thread=" << omp_get_thread_num() << '\n';
        }

        HopfieldBits    gen(N, beta, N_sweeps, P);   // untimed generator only
        NoBitsNeighbors classic_nb(N, beta, N_sweeps, P);
        NoBitsOverlaps  classic_ov(N, beta, N_sweeps, P);

        using CfgT = decay_t<decltype(gen.getSpinsConfig())>;
        using PatT = decay_t<decltype(gen.getPatternsBitsToCanonical())>;

        static_assert(is_same<CfgT, vector<int>>::value,
                      "getSpinsConfig() is expected to return vector<int>");

        // Graph: built once on gen, copied to both models
        gsl_rng_set(ran, seed_graph(job.run));

        gen.initNetwork_RegularRandom(rho, ran);
        copy_graph(gen, classic_nb);
        copy_graph(gen, classic_ov);

        // Patterns + initial state: one pair per repetition (drawn on gen)
        // rep 0 : original sequence
        // rep i : same spin draw, patterns from their own stream seed_pat_rep
        vector<CfgT> init_cfg(n_rep);
        vector<PatT> pat_cfg(n_rep);

        for (int i = 0; i < n_rep; ++i) {

            gsl_rng_set(ran, seed_init_rep(job.run, i));

            gen.initSpinsBits(ran);
            gen.toCanonical();

            if (i > 0) {
                gsl_rng_set(ran, seed_pat_rep(job.run, i));
            }

            gen.initPatternsBits(ran);

            init_cfg[i] = gen.getSpinsConfig();
            pat_cfg[i]  = gen.getPatternsBitsToCanonical();
        }

        vector<CfgT> start_nb(n_rep), start_ov(n_rep);
        vector<CfgT> end_nb(n_rep), end_ov(n_rep);

        auto reset_nb = [&](int i) {

            classic_nb.initPatternsFromConfig(pat_cfg[i]);
            classic_nb.initSpinsFromConfig(init_cfg[i]);

            start_nb[i] = classic_nb.getSpinsConfig();
        };

        auto reset_ov = [&](int i) {

            classic_ov.initPatternsFromConfig(pat_cfg[i]);
            classic_ov.initSpinsFromConfig(init_cfg[i]);

            start_ov[i] = classic_ov.getSpinsConfig();
        };

        auto post_nb = [&](int i) {
            end_nb[i] = classic_nb.getSpinsConfig();
        };

        auto post_ov = [&](int i) {
            end_ov[i] = classic_ov.getSpinsConfig();
        };

        auto check_pair = [&](const char* stage,
                              int rep,
                              const CfgT& a,
                              const CfgT& b) -> bool {

            const bool ok = IO.check_equality_configs(a, b, N);

            if (!ok) {
                #pragma omp critical(print_check)
                {
                    cout << "[FAILED] run " << job.run
                         << " rep " << rep
                         << " | " << stage
                         << " | NoBits shared neighbors vs overlaps\n";
                }
            }

            return ok;
        };

        bool all_ok = true;

        for (int i = 0; i < n_rep; ++i) {

            reset_nb(i);
            reset_ov(i);

            all_ok &= check_pair(
                "before",
                i,
                start_nb[i],
                start_ov[i]
            );
        }

        if (all_ok) {
            cout << "=== INDENTICAL CONFIGS (before evolution) ===\n";
        }

        auto seed_of = [&](int i) {
            return seed_evol_rep(job.run, i);
        };

        // Timed work
        auto run_nb = [&] {
            classic_nb.runSweepsSharedRNG_neighbors(ran, save, freq);
        };

        auto run_ov = [&] {
            classic_ov.runSweepsSharedRNG_overlaps(ran, save, freq);
        };

        // Warm-up: one untimed run of each method
        reset_nb(0);
        timed_run(ran, seed_of(0), run_nb);

        reset_ov(0);
        timed_run(ran, seed_of(0), run_ov);

        vector<double> t_nb, t_ov;

        t_nb.reserve(n_rep);
        t_ov.reserve(n_rep);

        auto measure_nb = [&](int i) {

            reset_nb(i);
            t_nb.push_back(timed_run(ran, seed_of(i), run_nb));
            post_nb(i);
        };

        auto measure_ov = [&](int i) {

            reset_ov(i);
            t_ov.push_back(timed_run(ran, seed_of(i), run_ov));
            post_ov(i);
        };

        // Interleaved, alternating order: same i for both methods (paired)
        for (int i = 0; i < n_rep; ++i) {

            if (i % 2 == 0) {
                measure_nb(i);
                measure_ov(i);
            }
            else {
                measure_ov(i);
                measure_nb(i);
            }
        }

        const Stats st_nb = compute_stats(t_nb);
        const Stats st_ov = compute_stats(t_ov);

        all_ok = true;

        for (int i = 0; i < n_rep; ++i) {
            all_ok &= check_pair(
                "after",
                i,
                end_nb[i],
                end_ov[i]
            );
        }

        if (all_ok) {
            cout << "=== INDENTICAL CONFIGS (after evolution) ===\n";
        }

        const double ratio = st_nb.median / st_ov.median;   // >1 : overlaps faster

        // Write CSV
        #pragma omp critical(write_csv)
        {
            out << N << ","
                << rho << ","
                << job.alpha << ","
                << beta << ","
                << N_sweeps << ","
                << n_rep << ",";

            write_stats(out, st_nb);
            out << ",";

            write_stats(out, st_ov);
            out << ",";

            out << ratio << "\n";
            out.flush();

            cout << "Run " << job.run
            << " | N=" << N
            << " | alpha=" << job.alpha << " (P=" << P << ")"
            << " | T=" << job.temperature
            << " | rho=" << rho
            << " | N_sweeps=" << N_sweeps
            << " | median: nb=" << st_nb.median
            << "s, ov=" << st_ov.median
            << "s | nb/ov=" << ratio << "\n";
        }

        gsl_rng_free(ran);
    }

    auto end_all = chrono::steady_clock::now();

    cout << "\nTotal wall time: "
         << chrono::duration<double>(end_all - start_all).count()
         << " s\n";

    return 0;
}