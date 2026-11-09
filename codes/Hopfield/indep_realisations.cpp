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
    SimulationIO IO;

    // ── Parameters scanned ───────────────────────────────────────────────────
    // Load alpha = P / N (number of patterns per neuron)
    vector<double> alpha_vals = {
        0.01, 0.025, 0.05, 0.1, 0.138, 0.15, 0.2, 0.3
    };
    // Temperatures (beta = 1 / T)
    vector<double> temperatures = {
        0.1, 0.25, 0.5, 0.75,
        1.0, 2.0, 3.0, 4.0, 5.0,
        7.0, 10.0
    };

    // ── Fixed simulation parameters ──────────────────────────────────────────
    const int L        = 32;       // lattice size (L x L spins)
    const int N        = L * L;    // number of neurons
    const int N_sweeps = 1 << 13;  // number of Metropolis sweeps

    // Random number generator (GSL, gfsr4); it is reseeded at each point below
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

    clock_t start_all = clock();
    int k   = 0;  // alpha counter (for display)
    int run = 0;  // index of the current (alpha, beta) point, used in the seeds

    // ── Loop over alpha ──────────────────────────────────────────────────────
    for (double alpha : alpha_vals) {
        int l = 0;  // temperature counter (for display)
        k += 1;

        // ── Loop over temperature ────────────────────────────────────────────
        for (double temperature : temperatures) {
            l += 1;
            run += 1;
            double beta = 1 / temperature;
            int P = static_cast<int>(round(L * L * alpha));  // number of patterns

            // Output folder names (built from alpha, N and beta)
            ostringstream folder_name_bits;
            ostringstream folder_name_nobits;

            // ── Model initialization ─────────────────────────────────────────
            HopfieldBits bits(L * L, beta, N_sweeps, P);
            HopfieldNoBits nobits(L * L, beta, N_sweeps, P);

            folder_name_bits << "../../results/Hopfield/indep/bits/alpha_"
                             << std::fixed << std::setprecision(3) << alpha
                             << "/N" << L * L << "/" << "beta"
                             << SimulationBase::format_beta(beta);
            folder_name_nobits << "../../results/Hopfield/indep/nobits/alpha_"
                               << std::fixed << std::setprecision(3) << alpha
                               << "/N" << L * L << "/" << "beta"
                               << SimulationBase::format_beta(beta);

            string base_folder_bits   = folder_name_bits.str();
            string base_folder_nobits = folder_name_nobits.str();

            // Create the output folders and give them to the models
            make_dir(base_folder_bits);
            make_dir(base_folder_nobits);
            bits.SetBaseFolder(base_folder_bits.c_str());
            nobits.SetBaseFolder(base_folder_nobits.c_str());

            // Run summary
            cout << "\n==============================================\n";
            cout << "Hopfield 2D Model - alpha = " << alpha << " (P = " << P << ")    (" << k << "/" << alpha_vals.size() << ")\n";
            cout << "  Lattice:       " << L << " x " << L << " = " << L * L << " neurons\n";
            cout << "  Temperature:   " << temperature << "    (" << l << "/" << temperatures.size() << ")\n";;
            cout << "  Sweeps:        " << N_sweeps << "\n";
            cout << "  Base folder:   " << base_folder_bits << "\n";
            cout << "==============================================\n\n";

            // Build the network (fully connected) for both models
            bits.initNetwork_FullyConnected();
            nobits.initNetwork_FullyConnected();
            cout << "Network initialized" << endl;

            // ── Initial spin configuration (same for both models) ────────────
            gsl_rng_set(ran, 123 + run);  // seed specific to this (alpha, beta) point
            bits.initSpinsBits(ran);
            bits.toCanonical();  // bit representation {0,1} -> canonical {-1,+1}
            auto initial_config_bits = bits.getSpinsConfig();
            nobits.initSpinsFromConfig(initial_config_bits);
            cout << "Initial Configuration initialized" << endl;
            // Check that both models start from the same configuration
            IO.check_equality_configs(initial_config_bits, nobits.getSpinsConfig(), initial_config_bits, N);

            // ── Patterns (drawn once by bits, then given to nobits) ──────────
            bits.initPatternsBits(ran);
            auto patterns = bits.getPatternsBitsToCanonical();
            nobits.initPatternsFromConfig(patterns);
            bits.SavePatterns(patterns, P);
            cout << "Patterns initialized and saved\n" << endl;

            // ── Evolution: HopfieldBits ──────────────────────────────────────
            gsl_rng_set(ran, 42 + run);  // seed of the dynamics, same for bits and nobits

            /* Initialization on a pattern (optional):
            auto Patterns = bits.getPatternsBits();
            int mu = gsl_rng_uniform_int(ran, P);
            vector<Bits> Config(N);
            for (int spin = 0; spin < N; ++spin)
                Config[spin] = Patterns[spin*P + mu];
            bits.initSpinsFromConfigBits(Config);
            cout << "Bitwise initialization done" << endl;
            */

            cout << "bitwise: " << endl;
            bits.OpenSpinFiles();
            clock_t start_bits = clock();
            bits.runSweeps(ran, /*save=*/true, /*freq=*/0.1, /*burn_in=*/0);
            bits.SaveSpinConfigurations(bits.getNSweeps());
            bits.CloseSpinFiles();
            //bits.toCanonical();
            cout << "evolve_save_bits terminated" << endl;
            clock_t end_bits = clock();
            double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
            cout << "HopfieldBits done for alpha = " << alpha
                 << ". Time: " << clock_bits << " s\n";

            // ── Evolution: HopfieldNoBits (same seed as bits) ────────────────
            cout << "\nnobits independent RNG: " << endl;
            gsl_rng_set(ran, 42 + run);  // exactly the same value as for bits
            nobits.OpenSpinFiles();
            clock_t start_nobits = clock();
            nobits.runSweepsIndependentRNG(ran, /*save=*/true, /*freq=*/0.1);
            nobits.SaveSpinConfigurations(bits.getNSweeps());
            nobits.CloseSpinFiles();
            cout << "evolve_save_nobits terminated" << endl;
            clock_t end_nobits = clock();
            double clock_nobits = double(end_nobits - start_nobits) / CLOCKS_PER_SEC;
            cout << "HopfieldNoBits done for alpha = " << alpha
                 << ". Time: " << clock_nobits << " s\n";

            // Compare the final configurations of the two models (disabled)
            //IO.check_equality_configs(initial_config_bits, nobits.getSpinsConfig(), bits.getSpinsConfig(), N);
        }
    }

    // ── Total time ───────────────────────────────────────────────────────────
    clock_t end_all = clock();
    double clock_all = double(end_all - start_all) / CLOCKS_PER_SEC;
    cout << "\nSimulation Complete. \n"
         << "Total Time: " << clock_all << " s\n";

    gsl_rng_free(ran);  // free the RNG
    return 0;
}