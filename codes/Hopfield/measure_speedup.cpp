#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
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

// The two bitwise versions (replace the aliases if they become separate classes)
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

// Time a callable (seconds). The RNG is reseeded right before the run,
// so every method starts from the same random stream.
template <class F>
double timed_run(gsl_rng* ran, unsigned long seed, F&& fn) {
    gsl_rng_set(ran, seed);
    auto t0 = chrono::steady_clock::now();
    fn();
    auto t1 = chrono::steady_clock::now();
    return chrono::duration<double>(t1 - t0).count();
}

int main() {
    // ── Scanned parameters ────────────────────────────────────────────────────
    vector<int>    sizes        = {16, 32, 64};                                        // 3
    vector<double> alpha_vals   = {0.02, 0.05, 0.1, 0.2, 0.3};                         // 5
    vector<double> temperatures = {0.2, 0.5, 1.0, 2.0, 5.0};                           // 5
    vector<double> rho_vals     = {0.1 /*, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0*/ };  // 1 (10 when fully enabled)

    // Number of jobs = 3 * 5 * 5 * 1 = 75 (750 with the full rho grid)

    const int N_sweeps = 1 << 13;
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

    // Single output file: header written once, before the parallel region
    // Naming: classic = nobits implementation, bitwise = bits implementation
    ofstream out(out_path);
    out << "N,rho,alpha,P,beta,k_eff,n_sweeps,"
           "t_classic_neighbors,t_classic_overlaps,"
           "t_bitwise_neighbors,t_bitwise_overlaps,"
           "speedup_neighbors,speedup_overlaps,"
           "t_bitwise_overlaps_over_neighbors\n";
    out << setprecision(10);
    out.flush();

    auto start_all = chrono::steady_clock::now();

    // One job per (L, alpha, T, rho)
    #pragma omp parallel for schedule(dynamic)
    for (int job_id = 0; job_id < nb_runs; ++job_id) {

        const Job& job = jobs[job_id];
        const int    N    = job.L * job.L;
        const double beta = 1.0 / job.temperature;
        const int    P    = static_cast<int>(round(N * job.alpha));
        const double rho  = job.rho;

        const bool save    = false;   // no disk I/O, it would bias the timings
        const int  freq    = 1;
        const int  burn_in = 0;

        // Each job owns its own RNG
        gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

        BitsNeighbors  bits_nb(N, beta, N_sweeps, P);
        BitsOverlaps   bits_ov(N, beta, N_sweeps, P);
        HopfieldNoBits classic(N, beta, N_sweeps, P);

        // ── Graph: built once (bits_nb), copied to the two other models ──────
        gsl_rng_set(ran, seed_graph(job.run));
        bits_nb.initNetwork_RegularRandom(rho, ran);
        copy_graph(bits_nb, bits_ov);
        copy_graph(bits_nb, classic);

        // Degree actually obtained (may differ from rho*(N-1): rounding, parity)
        const int k_eff = bits_nb.getDegrees()[0];

        // ── Spins and patterns: drawn once on bits_nb ────────────────────────
        gsl_rng_set(ran, seed_init(job.run));
        bits_nb.initSpinsBits(ran);
        bits_nb.toCanonical();
        bits_nb.initPatternsBits(ran);

        // ── Same spins and patterns for the other models ─────────────────────
        auto initial_config      = bits_nb.getSpinsConfig();
        auto initial_config_bits = bits_nb.getSpinsConfigBits();
        auto patterns            = bits_nb.getPatternsBitsToCanonical();
        auto patterns_bits       = bits_nb.getPatternsBits();

        bits_ov.initSpinsFromConfigBits(initial_config_bits);
        bits_ov.initPatternsFromConfigBits(patterns_bits);

        classic.initPatternsFromConfig(patterns);

        // ── Classic neighbors (reference for bitwise neighbors) ──────────────
        classic.initSpinsFromConfig(initial_config);   // fresh initial state
        const double t_classic_neighbors = timed_run(ran, seed_evol(job.run), [&] {
            classic.runSweepsIndependentRNG_neighbors(ran, save, freq);
        });

        // ── Classic overlaps (reference for bitwise overlaps) ────────────────
        classic.initSpinsFromConfig(initial_config);   // reset: same initial state again
        const double t_classic_overlaps = timed_run(ran, seed_evol(job.run), [&] {
            classic.runSweepsIndependentRNG_overlaps(ran, save, freq);   // computes overlaps itself
        });

        // ── Bitwise neighbors (setup time included) ──────────────────────────
        const double t_bitwise_neighbors = timed_run(ran, seed_evol(job.run), [&] {
            bits_nb.runSweeps_neighbors(ran, save, freq, burn_in);
        });

        // ── Bitwise overlaps (setup time included) ───────────────────────────
        const double t_bitwise_overlaps = timed_run(ran, seed_evol(job.run), [&] {
            bits_ov.runSweeps_overlaps(ran, save, freq, burn_in);
        });

        // Each bitwise method vs. its own classic equivalent
        const double speedup_neighbors = t_classic_neighbors / t_bitwise_neighbors;
        const double speedup_overlaps  = t_classic_overlaps  / t_bitwise_overlaps;
        // Bitwise vs. bitwise (> 1 means neighbors is faster than overlaps)
        const double ratio_bitwise = t_bitwise_overlaps / t_bitwise_neighbors;

        // Protected write: one line per job, immediate flush
        #pragma omp critical(write_csv)
        {
            out << N << "," << rho << "," << job.alpha << "," << P << "," << beta << ","
                << k_eff << "," << N_sweeps << ","
                << t_classic_neighbors << "," << t_classic_overlaps << ","
                << t_bitwise_neighbors << "," << t_bitwise_overlaps << ","
                << speedup_neighbors << "," << speedup_overlaps << ","
                << ratio_bitwise << "\n";
            out.flush();

            cout << "Run " << job.run << "/" << nb_runs
                 << " (alpha=" << job.alpha << ", N=" << N << ", P=" << P
                 << ", k=" << k_eff << ", beta=" << beta << ", rho=" << rho << ")"
                 << " classic[nb=" << t_classic_neighbors << "s, ov=" << t_classic_overlaps << "s]"
                 << " bitwise[nb=" << t_bitwise_neighbors << "s, ov=" << t_bitwise_overlaps << "s]\n";
        }

        gsl_rng_free(ran);
    }

    auto end_all = chrono::steady_clock::now();
    cout << "\nTotal wall time: "
         << chrono::duration<double>(end_all - start_all).count() << " s\n";
    return 0;
}