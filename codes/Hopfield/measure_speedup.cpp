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

// The two bits versions to compare (replace the aliases if they become separate classes)
using Bits1 = HopfieldBits;   // neighbors-based version
using Bits2 = HopfieldBits;   // shifted-overlaps version

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
    ofstream out(out_path);
    out << "N,rho,alpha,P,beta,k_eff,n_sweeps,"
       "t_ref,t_1,t_2,speedup_1,speedup_2,speedup_1_over_2\n";
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

        Bits1 bits1(N, beta, N_sweeps, P);
        Bits2 bits2(N, beta, N_sweeps, P);
        HopfieldNoBits nobits(N, beta, N_sweeps, P);

        // ── Graph: built once (bits1), copied to the two other models ────────
        gsl_rng_set(ran, seed_graph(job.run));
        bits1.initNetwork_RegularRandom(rho, ran);
        copy_graph(bits1, bits2);
        copy_graph(bits1, nobits);

        // Degree actually obtained (may differ from rho*(N-1): rounding, parity)
        const int k_eff = bits1.getDegrees()[0];

        // ── Spins and patterns: drawn once on bits1 ──────────────────────────
        gsl_rng_set(ran, seed_init(job.run));
        bits1.initSpinsBits(ran);
        bits1.toCanonical();
        bits1.initPatternsBits(ran);

        // ── Copy to bits2 and nobits: same spins, same patterns ──────────────
        auto initial_config      = bits1.getSpinsConfig();
        auto initial_config_bits = bits1.getSpinsConfigBits();
        auto patterns            = bits1.getPatternsBitsToCanonical();
        auto patterns_bits       = bits1.getPatternsBits();

        bits2.initSpinsFromConfigBits(initial_config_bits);
        bits2.initPatternsFromConfigBits(patterns_bits);

        nobits.initSpinsFromConfig(initial_config);
        nobits.initPatternsFromConfig(patterns);

        // ── Reference: nobits ────────────────────────────────────────────────
        gsl_rng_set(ran, seed_evol(job.run));
        auto t0 = chrono::steady_clock::now();
        nobits.runSweepsSharedRNG(ran, save, freq);
        auto t1 = chrono::steady_clock::now();
        const double t_ref = chrono::duration<double>(t1 - t0).count();

        // ── Version 1: bits couplings + neighbor-based sweeps ────────────────
        // (setup time included in t_1)
        gsl_rng_set(ran, seed_evol(job.run));
        auto t0_1 = chrono::steady_clock::now();
        bits1.initCouplingsBits();
        bits1.runSweeps_neighbors(ran, save, freq, burn_in);
        auto t1_1 = chrono::steady_clock::now();
        const double t_1 = chrono::duration<double>(t1_1 - t0_1).count();

        // ── Version 2: shifted overlaps + overlap-based sweeps ───────────────
        // (setup time included in t_2)
        gsl_rng_set(ran, seed_evol(job.run));
        auto t0_2 = chrono::steady_clock::now();
        bits2.compute_shifted_overlaps();
        bits2.runSweeps_overlaps(ran, save, freq, burn_in);
        auto t1_2 = chrono::steady_clock::now();
        const double t_2 = chrono::duration<double>(t1_2 - t0_2).count();

        const double speedup_1 = t_ref / t_1;
        const double speedup_2 = t_ref / t_2;
        const double ratio     = speedup_1 / speedup_2;   // = t_2 / t_1

        // Protected write: one line per job, immediate flush
        #pragma omp critical(write_csv)
        {
            out << N << "," << rho << "," << job.alpha << "," << P << "," << beta << ","
            << k_eff << "," << N_sweeps << ","
            << t_ref << "," << t_1 << "," << t_2 << ","
            << speedup_1 << "," << speedup_2 << "," << ratio << "\n";

            cout << "Run " << job.run << "/" << nb_runs
                 << " (alpha=" << job.alpha << ", N=" << N << ", P=" << P
                 << ", k=" << k_eff << ", beta=" << beta << ", rho=" << rho << ")"
                 << " ref=" << t_ref << "s, v1=" << t_1 << "s, v2=" << t_2 << "s\n";
        }

        gsl_rng_free(ran);
    }

    auto end_all = chrono::steady_clock::now();
    cout << "\nTotal wall time: "
         << chrono::duration<double>(end_all - start_all).count() << " s\n";
    return 0;
}