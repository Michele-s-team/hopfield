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

    // ── Fixed simulation parameters ──────────────────────────────────────────
    const int L         = 32;           // lattice size (L×L spins)
    const double beta   = 5;           // inverse temperature
    const int N_sweeps  = 1 << 12;       // number of Metropolis sweeps

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);  // fixed seed for reproducibility

    double step=0.01;
    double alpha_min=0.02;
    double alpha_max=0.18;

    int n_steps = static_cast<int>(round((alpha_max - alpha_min) / step)); // computed from step and alpha_max

    clock_t start_all = clock();

    for (int k = 0; k <= n_steps; k++) {
        double alpha = alpha_min + k * step;
        int P = static_cast<int>(round(L * L * alpha)); // number of patterns

        // Build folder name using alpha value directly
        ostringstream folder_name;

        // ── Model initialization ────────────────────────────────────────────

        HopfieldBits bits(L * L, beta, N_sweeps, P);

        folder_name << "../results/Hopfield/alpha_"
                    << std::fixed << std::setprecision(3)
                    << alpha
                    << "/N"
                    << L*L
                    << "/"
                    << "beta"
                    << SimulationBase::format_beta(beta);
          string base_folder = folder_name.str();

        make_dir(base_folder);

        bits.SetBaseFolder(base_folder.c_str());

        cout << "\n==============================================\n";
        cout << "Hopfield 2D Model - alpha = " << alpha << " (P = " << P << ")    ("<< k+1 <<"/"<<n_steps+1 << ")\n";
        cout << "  Lattice:       " << L << " x " << L << " = " << L*L << " neurons\n";
        cout << "  Temperature:   " << 1.0/beta << "\n";
        cout << "  Sweeps:        " << N_sweeps << "\n";
        cout << "  Base folder:   " << base_folder << "\n";
        cout << "==============================================\n\n";      

        // Build the 2D square lattice with periodic boundary conditions
        bits.initNetworkFullyConnected();
        cout << "Network initialized (Fully Connected)\n";

        // ── Evolution: HopfieldBits ──────────────────────────────────────────
        gsl_rng_set(ran, 42);  // fixed seed for the dynamics
        clock_t start_bits = clock();
        bits.evolve_save_bits(ran, 1);
        clock_t end_bits = clock();
        double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
        cout << "\nHopfieldBits done for alpha = " << alpha
             << ". Time: " << clock_bits << " s\n";
    }
    
    clock_t end_all = clock();
    double clock_all = double(end_all - start_all) / CLOCKS_PER_SEC;

    cout << "\nSimulation Complete. \n"
             << "Total Time: " << clock_all << " s\n";
    
    gsl_rng_free(ran);

    return 0;
}