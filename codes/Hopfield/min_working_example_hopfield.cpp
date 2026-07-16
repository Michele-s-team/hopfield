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
// Minimal working example — correctness and performance check:
//   HopfieldBits vs HopfieldNoBits
//
// Both models are initialized from the same disorder realization (patterns)
// and the same initial spin configuration. They are then evolved with the same
// RNG seed so that each flip attempt draws the same random number in both cases.
// The resulting magnetizations are compared replica by replica.
//
// Purpose: validate that the bitwise implementation (HopfieldBits) produces
// the same physics as the reference scalar implementation (HopfieldNoBits),
// and measure the wall-clock speedup.
//
// No output files are written.
// =============================================================================

// ──────────────────────────────────────────────────────────────────────────────
// check_equality_configs
//
// Compare the spin configurations of the classic and bitwise models, replica
// by replica and site by site. Reports the first mismatch found, or confirms
// that both configurations are identical.
// ──────────────────────────────────────────────────────────────────────────────
void check_equality_configs(const vector<int>& neurons_before,
                   const vector<int>& neurons_nobits,
                   const vector<int>& neurons_bits,
                   int N) {

    bool all_equal = true;

    for (int r = 0; r < n_bits; r++) {
        // Site-by-site comparison for this replica
        for (int i = 0; i < N; i++) {
            if (neurons_nobits[r * N + i] != neurons_bits[r * N + i]) {
                all_equal = false;
                cout << "Mismatch at r=" << r << " i=" << i << endl;
            }
        }
    }

    if (all_equal)
        cout << "OK: classic and bitwise results are identical\n";
    else
        cout << "WARNING: differences detected between classic and bitwise results\n";
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────
int main() {

    // ── Simulation parameters ─────────────────────────────────────────────────
    const int L         = 50;             // linear lattice size (L×L spins)
    const double beta   = 10;             // inverse temperature
    const double alpha  = 0.1;            // alpha = P/N
    const int P         = L * L * alpha;  // number of patterns (small for testing)
    const int N_sweeps  = 1 << 10;        // number of Metropolis sweeps 

    cout << "\n";
    cout << "Hopfield 2D Model - Bits vs NoBits Comparison\n\n";
    cout << "Parameters:\n";
    cout << "  Lattice:       " << L << " x " << L << " = " << L*L << " neurons\n";
    cout << "  alpha:         " << alpha << "\n";
    cout << "  Patterns (P):  " << P << "\n";
    cout << "  Temperature:   " << 1.0/beta << "\n";
    cout << "  Sweeps:        " << N_sweeps << "\n\n";

    // ── Model initialization ──────────────────────────────────────────────────
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);  // fixed seed for reproducibility

    // Construct both models with identical parameters
    HopfieldBits   bits  (L * L, beta, N_sweeps, P);
    //HopfieldNoBits nobits(L * L, beta, N_sweeps, P);

    bits.SetBaseFolder("../results/");

    // Build the 2D square lattice with periodic boundary conditions
    bits.initNetworkFullyConnected();
    //nobits.initNetwork2D_PBC();
    cout << "Network initialized (Fully Connected)\n";

    // Generate patterns once, then copy them to nobits so both models
    // share the exact same pattern realization
    bits.initPatterns(ran);
    vector<vector<vector<int>>> initial_patterns = bits.getPatterns();
    bits.SavePatterns("patterns/",initial_patterns);
    //nobits.initPatternsFromConfig(initial_patterns);
    cout << "Patterns initialized and shared between models\n";

    // Generate the initial spin configuration once and share it across both models
    bits.initSpins(ran);
    vector<int> initial_config = bits.getSpinsConfig();
    //nobits.initSpinsFromConfig(initial_config);
    cout << "Spin configurations initialized and shared\n";

    // Verify that both models start from the same spin state before evolving
    //check_equality_configs(initial_config, nobits.getSpinsConfig(),bits.getSpinsConfig(), L*L);

    // ── Evolution: HopfieldBits ──────────────────────────────────────────────
    gsl_rng_set(ran, 42);  // fixed seed for RNG comparison
    clock_t start_bits = clock();
    bits.evolve_save(ran, 1,"spins/");
    clock_t end_bits = clock();
    double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
    cout << "\nHopfieldBits done. Time: " << clock_bits << " s\n";

    /*

    // ── Evolution: HopfieldNoBits ────────────────────────────────────────────
    gsl_rng_set(ran, 42);  // same seed for fair comparison
    clock_t start_ref = clock();
    nobits.evolve(ran);
    clock_t end_ref = clock();
    double clock_ref = double(end_ref - start_ref) / CLOCKS_PER_SEC;
    cout << "HopfieldNoBits done. Time: " << clock_ref << " s\n";

    // ── Magnetization comparison ──────────────────────────────────────────────
    vector<double> mag_bits(n_bits), mag_nobits(n_bits);
    bits.GetMagnetizations(mag_bits);
    nobits.GetMagnetizations(mag_nobits);

    bool mags_equal = true;
    for (int r = 0; r < n_bits; r++) {
        if (abs(mag_bits[r] - mag_nobits[r]) > 1e-10) {
            mags_equal = false;
            cout << "Difference at r=" << r
                 << ": bits=" << mag_bits[r]
                 << " nobits=" << mag_nobits[r] << endl;
        }
    }
    if (mags_equal) {
        cout << "OK: magnetizations are identical.\n";
    }

    // ── Performance summary ───────────────────────────────────────────────────
    cout << "\n=== Performance Summary ===\n";
    cout << "HopfieldBits time:   " << clock_bits << " s\n";
    cout << "HopfieldNoBits time: " << clock_ref << " s\n";
    cout << "Acceleration factor: " << clock_ref / clock_bits << "x\n";

    */

    gsl_rng_free(ran);

    return 0;
}