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

BitSet BitSet_one;

using BitsNeighbors = HopfieldBits;   // bitwise, neighbor-coupling formulation
using BitsOverlaps  = HopfieldBits;   // bitwise, shifted-overlap formulation

// Seeds shared by all models of a given job (same seed => same draws)
static inline unsigned long seed_graph(int run) { return 1000 + run; }
static inline unsigned long seed_init (int run) { return  123 + run; }
static inline unsigned long seed_evol (int run) { return   42 + run; }

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

// Repeat a measurement n times. `reset` restores the same initial state before
// each repetition (it is NOT timed); `fn` is the timed work (setup + sweeps).
// Same seed every time => identical trajectory, so the spread is pure timing noise.
template <class Reset, class F>
vector<double> repeat_timed(int n, gsl_rng* ran, unsigned long seed, Reset&& reset, F&& fn) {
    vector<double> t;
    t.reserve(n);
    for (int i = 0; i < n; ++i) {
        reset();
        t.push_back(timed_run(ran, seed, fn));
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
    vector<int>    sizes        = {16, 32, 64};                                        // 3
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

    // Header: classic = nobits, bitwise = bits; 4 stats per method
    ofstream out(out_path);
    out << "N,rho,alpha,P,beta,k_eff,n_sweeps,n_rep";
    for (const char* m : {"classic_neighbors", "classic_overlaps",
                          "bitwise_neighbors", "bitwise_overlaps"})
        for (const char* s : {"mean", "min", "max", "median"})
            out << ",t_" << m << "_" << s;
    out << ",speedup_neighbors,speedup_overlaps,t_bitwise_overlaps_over_neighbors\n";
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
        HopfieldNoBits classic(N, beta, N_sweeps, P);

        // ── Graph: built once, copied to the two other models ────────────────
        gsl_rng_set(ran, seed_graph(job.run));
        bits_nb.initNetwork_RegularRandom(rho, ran);
        copy_graph(bits_nb, bits_ov);
        copy_graph(bits_nb, classic);

        const int k_eff = bits_nb.getDegrees()[0];

        // ── Spins and patterns: drawn once on bits_nb ────────────────────────
        gsl_rng_set(ran, seed_init(job.run));
        bits_nb.initSpinsBits(ran);
        bits_nb.toCanonical();
        bits_nb.initPatternsBits(ran);

        auto initial_config      = bits_nb.getSpinsConfig();
        auto initial_config_bits = bits_nb.getSpinsConfigBits();
        auto patterns            = bits_nb.getPatternsBitsToCanonical();
        auto patterns_bits       = bits_nb.getPatternsBits();

        bits_ov.initSpinsFromConfigBits(initial_config_bits);
        bits_ov.initPatternsFromConfigBits(patterns_bits);
        classic.initPatternsFromConfig(patterns);

        // Reset helpers: same initial spins before every repetition
        auto reset_classic = [&] { classic.initSpinsFromConfig(initial_config); };
        auto reset_nb      = [&] { bits_nb.initSpinsFromConfigBits(initial_config_bits); };
        auto reset_ov      = [&] { bits_ov.initSpinsFromConfigBits(initial_config_bits); };

        const unsigned long seed = seed_evol(job.run);

        // ── 4 measurements, run in series, each repeated n_rep times ─────────
        const Stats s_cn = compute_stats(repeat_timed(n_rep, ran, seed, reset_classic, [&] {
            classic.runSweepsIndependentRNG_neighbors(ran, save, freq);
        }));
        const Stats s_co = compute_stats(repeat_timed(n_rep, ran, seed, reset_classic, [&] {
            classic.runSweepsIndependentRNG_overlaps(ran, save, freq);
        }));
        const Stats s_bn = compute_stats(repeat_timed(n_rep, ran, seed, reset_nb, [&] {
            bits_nb.runSweeps_neighbors(ran, save, freq, burn_in);
        }));
        const Stats s_bo = compute_stats(repeat_timed(n_rep, ran, seed, reset_ov, [&] {
            bits_ov.runSweeps_overlaps(ran, save, freq, burn_in);
        }));

        // Speedups from medians (robust to outliers)
        const double speedup_neighbors = s_cn.median / s_bn.median;
        const double speedup_overlaps  = s_co.median / s_bo.median;
        const double ratio_bitwise     = s_bo.median / s_bn.median;

        #pragma omp critical(write_csv)
        {
            out << N << "," << rho << "," << job.alpha << "," << P << "," << beta << ","
                << k_eff << "," << N_sweeps << "," << n_rep << ",";
            write_stats(out, s_cn); out << ",";
            write_stats(out, s_co); out << ",";
            write_stats(out, s_bn); out << ",";
            write_stats(out, s_bo); out << ",";
            out << speedup_neighbors << "," << speedup_overlaps << "," << ratio_bitwise << "\n";
            out.flush();

            cout << "Run " << job.run << "/" << nb_runs
                 << " (alpha=" << job.alpha << ", N=" << N << ", P=" << P
                 << ", k=" << k_eff << ", beta=" << beta << ", rho=" << rho << ")"
                 << " median: classic[nb=" << s_cn.median << "s, ov=" << s_co.median << "s]"
                 << " bitwise[nb=" << s_bn.median << "s, ov=" << s_bo.median << "s]\n";
        }

        gsl_rng_free(ran);
    }

    auto end_all = chrono::steady_clock::now();
    cout << "\nTotal wall time: "
         << chrono::duration<double>(end_all - start_all).count() << " s\n";
    return 0;
}