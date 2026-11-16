#include <iostream>
#include <iomanip> 
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <strstream>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <list>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
#include <stdint.h>
#include <chrono>
#include <sys/stat.h> // for mkdir
#include <omp.h>

using namespace std;

#include "gsl_rng.h"
#include "gsl_math.h"
#include "gsl_sf_log.h"
#include "gsl_randist.h"

#include "main.hpp"
#include "int.hpp"

#include "ising_model.hpp"
#include "ising_nobits.hpp"
#include "ising_bits.hpp"

#include "spinglass_model.hpp"
#include "spinglass_nobits.hpp"
#include "spinglass_bits.hpp"

#include "hopfield_model.hpp"
#include "hopfield_nobits.hpp"
#include "hopfield_bits.hpp"

//g++ min_working_example_hopfield.cpp src/*.cpp -llapack -lgsl -lgslcblas -lm -O3 -flto -Wno-deprecated -Iinclude -I/usr/include/gsl -DHAVE_INLINE -o main.o


BitSet BitSet_one; // really strange that we need to define this for the operator -= of BitSet 

// =============================================================================
// Alpha sweep — HopfieldBits only
//
// For each value of alpha between 0 and 0.20 (step 0.01), a new base
// directory "results/alpha_XX/" is created, the network and patterns are
// initialized, and the model is evolved and saved into that directory.
// =============================================================================

// ──────────────────────────────────────────────────────────────────────────────
// make_dir
//
void make_dir(const string& path) {
    mkdir(path.c_str(), 0755);
}

// Create a directory if it does not already exist (mkdir fails silently
// with errno = EEXIST if the directory is already present).
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────

int main() {
    // Parameters scanned
    vector<int> sizes = {64};

    vector<double> alpha_vals = {
        0.01, 0.025//, 0.05, 0.1, 0.2, 0.3
    };

    vector<double> temperatures = {
        0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0
    };

    const int N_sweeps = 1 << 13;

    // Build the list of independent simulation jobs
    struct Job {
        int L;
        double alpha;
        double temperature;
        int run;
    };

    vector<Job> jobs;

    int run = 0;
    for (int L : sizes) {
        for (double alpha : alpha_vals) {
            for (double temperature : temperatures) {
                jobs.push_back({L, alpha, temperature, ++run});
            }
        }
    }

    const int nb_runs = jobs.size();

    auto start_all = chrono::steady_clock::now();

    // Parallel execution: one job per (L, alpha, temperature)
    #pragma omp parallel for schedule(dynamic)
    for (int job_id = 0; job_id < nb_runs; ++job_id) {

        const Job& job = jobs[job_id];

        const int L = job.L;
        const int N = L * L;
        const double alpha = job.alpha;
        const double temperature = job.temperature;
        const double beta = 1.0 / temperature;
        const int P = static_cast<int>(round(N * alpha));

        // Each job owns its own RNG
        gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

        // Output folders
        ostringstream folder_name_bits;
        ostringstream folder_name_nobits;

        folder_name_bits << "../../results/Hopfield/indep/bits/"
                         << "alpha_" << fixed << setprecision(3) << alpha
                         << "/N" << N << "/beta"
                         << SimulationBase::format_beta(beta);

        folder_name_nobits << "../../results/Hopfield/indep/nobits/"
                           << "alpha_" << fixed << setprecision(3) << alpha
                           << "/N" << N << "/beta"
                           << SimulationBase::format_beta(beta);

        string base_folder_bits = folder_name_bits.str();
        string base_folder_nobits = folder_name_nobits.str();

        // Initialize models
        HopfieldBits bits(N, beta, N_sweeps, P);
        HopfieldNoBits nobits(N, beta, N_sweeps, P);

        make_dir(base_folder_bits);
        make_dir(base_folder_nobits);

        bits.SetBaseFolder(base_folder_bits.c_str());
        nobits.SetBaseFolder(base_folder_nobits.c_str());

        // Print job information
        #pragma omp critical
        {
            cout << "\n============ Hopfield 2D Model ============\n";
            cout << "  Run:         " << job.run << "/" << nb_runs << "\n";
            cout << "  Lattice:     " << L << " x " << L << " = " << N << "\n";
            cout << "  Alpha:       " << alpha << " (P = " << P << ")\n";
            cout << "  Temperature:" << temperature << "\n";
            cout << "  Sweeps:      " << N_sweeps << "\n";
            cout << "  Thread:      " << omp_get_thread_num() << "\n";
            cout << "============================================\n";
        }

        // Build networks
        bits.initNetwork_FullyConnected();
        nobits.initNetwork_FullyConnected();

        // Initialize spins
        gsl_rng_set(ran, 123 + job.run);

        bits.initSpinsBits(ran);
        bits.toCanonical();

        auto initial_config_bits = bits.getSpinsConfig();
        nobits.initSpinsFromConfig(initial_config_bits);

        // Initialize patterns
        bits.initPatternsBits(ran);
        auto patterns = bits.getPatternsBitsToCanonical();

        nobits.initPatternsFromConfig(patterns);
        bits.SavePatterns(patterns, P);

        // Evolution: HopfieldBits
        gsl_rng_set(ran, 42 + job.run);

        auto start_bits = chrono::steady_clock::now();

        bits.OpenSpinFiles();
        bits.runSweeps(ran, true, 1, 0);
        bits.SaveSpinConfigurations(bits.getNSweeps());
        bits.CloseSpinFiles();

        auto end_bits = chrono::steady_clock::now();

        double time_bits =
            chrono::duration<double>(end_bits - start_bits).count();

        // Evolution: HopfieldNoBits
        gsl_rng_set(ran, 42 + job.run);

        auto start_nobits = chrono::steady_clock::now();

        nobits.OpenSpinFiles();
        nobits.runSweepsSharedRNG(ran, true, 1);
        nobits.SaveSpinConfigurations(nobits.getNSweeps());
        nobits.CloseSpinFiles();

        auto end_nobits = chrono::steady_clock::now();

        double time_nobits =
            chrono::duration<double>(end_nobits - start_nobits).count();

        // Print timings
        #pragma omp critical
        {
            cout << "Run " << job.run << " completed\n";
            cout << "  bits:   " << time_bits << " s\n";
            cout << "  nobits: " << time_nobits << " s\n";
        }

        gsl_rng_free(ran);
    }

    auto end_all = chrono::steady_clock::now();

    double total_time =
        chrono::duration<double>(end_all - start_all).count();

    cout << "\nSimulation complete.\n";
    cout << "Total wall time: " << total_time << " s\n";

    return 0;
}