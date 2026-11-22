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

// Repeat a measurement n times. For repetition i:
//   reset(i)   restores patterns AND initial state i (NOT timed),
//   seed_of(i) gives the evolution seed of repetition i,
//   fn         is the timed work,
//   post(i)    records the "after evolution" snapshot (NOT timed).
// Same i for both methods => paired comparison.
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

struct Job { int L; double alpha, temperature, rho; int run; };

// Benchmark: bitwise neighbors vs bitwise overlaps.
// Patterns and initial state change with each repetition i, but are shared
// by the two models for a given i. Ratio = t_neighbors / t_overlaps (medians).

using BitsNeighbors = HopfieldBits;   // bitwise, neighbor-coupling formulation
using BitsOverlaps  = HopfieldBits;   // bitwise, shifted-overlap formulation

int main() {
    vector<int>    sizes        = {8};
    vector<double> alpha_vals   = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6};
    vector<double> temperatures = {0.1, 1, 5};
    vector<double> rho_vals     = {0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};

    const int N_sweeps = 1 << 12;
    const string out_path = "../../results/Hopfield/speedup_bits.csv";

    vector<Job> jobs;
    int run = 0;
    for (int L : sizes)
        for (double alpha : alpha_vals)
            for (double T : temperatures)
                for (double rho : rho_vals)
                    jobs.push_back({L, alpha, T, rho, ++run});
    const int nb_runs = jobs.size();

    ofstream out(out_path);
    out << "N,rho,alpha,beta,n_sweeps,n_rep";
    for (const char* m : {"bitwise_neighbors", "bitwise_overlaps"})
        for (const char* s : {"mean", "min", "max", "median"})
            out << ",t_" << m << "_" << s;
    out << ",ratio_neighbors_over_overlaps\n";
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

        const bool save    = false;
        const int  freq    = 1;
        const int  burn_in = 0;

        gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

        #pragma omp critical(print_info)
        {
            cout << "[Bits nb vs ov] run " << job.run << "/" << nb_runs
                 << " | N=" << N << " | alpha=" << job.alpha << " (P=" << P << ")"
                 << " | T=" << job.temperature << " | rho=" << rho
                 << " | thread=" << omp_get_thread_num() << '\n';
        }

        BitsNeighbors bits_nb(N, beta, N_sweeps, P);
        BitsOverlaps  bits_ov(N, beta, N_sweeps, P);

        using CfgT     = decay_t<decltype(bits_nb.getSpinsConfig())>;
        using CfgBitsT = decay_t<decltype(bits_nb.getSpinsConfigBits())>;
        using PatBitsT = decay_t<decltype(bits_nb.getPatternsBits())>;
        static_assert(is_same<CfgT, vector<int>>::value,
                      "getSpinsConfig() is expected to return vector<int>");

        // Graph: built once on bits_nb, copied to bits_ov
        gsl_rng_set(ran, seed_graph(job.run));
        bits_nb.initNetwork_RegularRandom(rho, ran);
        copy_graph(bits_nb, bits_ov);

        // Patterns + initial state: one pair per repetition (drawn on bits_nb)
        //   rep 0 : original sequence (patterns follow the spins on the same stream)
        //   rep i : same spin draw, patterns from their own stream seed_pat_rep
        vector<CfgT>     init_cfg(n_rep);
        vector<CfgBitsT> init_bits(n_rep);
        vector<PatBitsT> pat_bits(n_rep);
        for (int i = 0; i < n_rep; ++i) {
            gsl_rng_set(ran, seed_init_rep(job.run, i));
            bits_nb.initSpinsBits(ran);
            bits_nb.toCanonical();
            if (i > 0) gsl_rng_set(ran, seed_pat_rep(job.run, i));
            bits_nb.initPatternsBits(ran);
            init_cfg[i]  = bits_nb.getSpinsConfig();
            init_bits[i] = bits_nb.getSpinsConfigBits();
            pat_bits[i]  = bits_nb.getPatternsBits();
        }

        vector<CfgT> start_nb(n_rep), start_ov(n_rep), end_nb(n_rep), end_ov(n_rep);

        auto reset_nb = [&](int i) {
            bits_nb.initPatternsFromConfigBits(pat_bits[i]);
            bits_nb.initSpinsFromConfigBits(init_bits[i]);
            bits_nb.toCanonical();
            start_nb[i] = bits_nb.getSpinsConfig();
        };
        auto reset_ov = [&](int i) {
            bits_ov.initPatternsFromConfigBits(pat_bits[i]);
            bits_ov.initSpinsFromConfigBits(init_bits[i]);
            bits_ov.toCanonical();
            start_ov[i] = bits_ov.getSpinsConfig();
        };
        auto post_nb = [&](int i) { bits_nb.toCanonical(); end_nb[i] = bits_nb.getSpinsConfig(); };
        auto post_ov = [&](int i) { bits_ov.toCanonical(); end_ov[i] = bits_ov.getSpinsConfig(); };

        auto check_pair = [&](const char* stage, int rep, const CfgT& a, const CfgT& b) -> bool {
            const bool ok = IO.check_equality_configs(a, b, N);
            if (!ok) {
                #pragma omp critical(print_check)
                cout << "[FAILED] run " << job.run << " rep " << rep
                     << " | " << stage << " | Bits neighbors vs Bits overlaps\n";
            }
            return ok;
        };

        bool all_ok = true;
        for (int i = 0; i < n_rep; ++i) {
            reset_nb(i);
            reset_ov(i);
            all_ok &= check_pair("before", i, start_nb[i], start_ov[i]);
        }
        if (all_ok) cout << "=== SUCCESS (before evolution) ===\n";

        auto seed_of = [&](int i) { return seed_evol_rep(job.run, i); };

        const Stats st_nb = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_nb, [&] {
            bits_nb.runSweeps_neighbors(ran, save, freq, burn_in);
        }, post_nb));
        const Stats st_ov = compute_stats(repeat_timed(n_rep, ran, seed_of, reset_ov, [&] {
            bits_ov.runSweeps_overlaps(ran, save, freq, burn_in);
        }, post_ov));

        all_ok = true;
        for (int i = 0; i < n_rep; ++i)
            all_ok &= check_pair("after", i, end_nb[i], end_ov[i]);
        if (all_ok) cout << "=== SUCCESS (after evolution) ===\n";

        const double ratio = st_nb.median / st_ov.median;   // >1 : overlaps faster

        #pragma omp critical(write_csv)
        {
            out << N << "," << rho << "," << job.alpha << "," << beta << ","
                << N_sweeps << "," << n_rep << ",";
            write_stats(out, st_nb); out << ",";
            write_stats(out, st_ov); out << ",";
            out << ratio << "\n";
            out.flush();
            cout << "Run " << job.run << "/" << nb_runs
                 << " median: nb=" << st_nb.median << "s, ov=" << st_ov.median
                 << "s | nb/ov=" << ratio << "\n";
        }

        gsl_rng_free(ran);
    }

    auto end_all = chrono::steady_clock::now();
    cout << "\nTotal wall time: "
         << chrono::duration<double>(end_all - start_all).count() << " s\n";
    return 0;
}
