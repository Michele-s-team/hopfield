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

    vector<int> sizes={8, 16, 32, 64};
    // Load alpha = P / N (number of patterns per neuron)
    vector<double> alpha_vals = {
        0.01, 0.025, 0.05, 0.1, 0.138, 0.15, 0.2, 0.3
    };
    // Temperatures (beta = 1 / T)
    vector<double> temperatures = {10,
        0.1, 0.25, 0.5, 0.75,
        1.0, 2.0, 3.0, 4.0, 5.0,
        7.0, 10.0
    };

    const int nb_runs = sizes.size()*alpha_vals.size()*temperatures.size();

    const int N_sweeps = 1 << 13;  // number of Metropolis sweeps

    // Random number generator (GSL, gfsr4); it is reseeded at each point below
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

    clock_t start_all = clock();
    int run = 0;  // index of the current (alpha, beta) point, used in the seeds

    int N;
    int k=0;  // size counter (for display)

    for (double L: sizes){
        k+=1;
        int l   = 0;  // alpha counter (for display)
    // ── Loop over alpha ──────────────────────────────────────────────────────
        for (double alpha : alpha_vals) {
            int m = 0;  // temperature counter (for display)
            l+= 1;

            // ── Loop over temperature ────────────────────────────────────────────
            for (double temperature : temperatures) {
                m += 1;
                run += 1;
                double beta = 1 / temperature;
                int P = static_cast<int>(round(L * L * alpha));  // number of patterns

                // Output folder names (built from alpha, N and beta)
                ostringstream folder_name_bits;

                // ── Model initialization ─────────────────────────────────────────
                HopfieldBits bits(L * L, beta, N_sweeps, P);

                folder_name_bits << "../../results/Hopfield/time_no_escape/"
                                << "alpha_" << std::fixed << std::setprecision(3) << alpha
                                << "/N" << static_cast<int>(L * L) << "/" << "beta"
                                << SimulationBase::format_beta(beta);

                string base_folder_bits   = folder_name_bits.str();

                // Create the output folders and give them to the models
                make_dir(base_folder_bits);
                bits.SetBaseFolder(base_folder_bits.c_str());

                // Run summary
                cout << "\n============ Hopfield 2D Model ============\n";
                cout << "  run " << run <<"/" << nb_runs <<endl;
                cout << "  Lattice:       " << L << " x " << L << " = " << L * L << " neurons   (" << k << "/" << sizes.size() << ")\n";;
                cout << "  alpha = " << alpha << " (P = " << P << ")    (" << l << "/" << alpha_vals.size() << ")\n";
                cout << "  Temperature:   " << temperature << "    (" << m << "/" << temperatures.size() << ")\n";
                cout << "  Sweeps:        " << N_sweeps << "\n";
                cout << "  Base folder:   " << base_folder_bits << "\n";
                cout << "==============================================\n\n";

                // Build the network (fully connected) for both models
                bits.initNetwork_FullyConnected();

                // ── Initial spin configuration (same for both models) ────────────
                gsl_rng_set(ran, 123 + run);  // seed specific to this (alpha, beta) point
                bits.initSpinsBits(ran);
                // Check that both models start from the same configuration

                // ── Patterns (drawn once by bits, then given to nobits) ──────────
                bits.initPatternsBits(ran);
                auto patterns = bits.getPatternsBitsToCanonical();
                bits.SavePatterns(patterns, P);

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

                bits.OpenSpinFiles();
                clock_t start_bits = clock();
                bits.runSweeps(ran, /*save=*/false, /*freq=*/0.1, /*burn_in=*/0);
                bits.SaveSpinConfigurations(bits.getNSweeps());
                bits.CloseSpinFiles();
                //bits.toCanonical();
                clock_t end_bits = clock();
                double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
            }
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