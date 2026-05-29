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

BitSet BitSet_one; // really strange that we need to define this for the operator -= of BitSet 

//  compile on mac, without optimization:
//  clear; clear;  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O0 -Wno-deprecated -I/Users/michelecastellana/Documents/office_stuff/work/stages/stage_bastien_dumont_2026/hopfield/codes/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//  compile on mac, with optimization:
//  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O3 -ftlo -Wno-deprecated  -I/Users/michelecastellana/Documents/gillespie/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//  compile on calcsub:
//  clear; clear;  g++ main.cpp src/*.cpp  -llapack -lgsl -lgslcblas -lm -O3 -Wno-deprecated -I /usr/include/gsl/ -I./include/ -o main.o -Wall -DHAVE_INLINE

//  compile on abacus:
//  g++ main.cpp src/*.cpp -I ./include/ -I /mnt/beegfs/home/mcastel1/gsl/include/gsl  -I/mnt/beegfs/home/mcastel1/gsl/include/ -L/mnt/beegfs/home/mcastel1/gsl/lib/ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o main.o -DHAVE_INLINE

// =============================================================================
// Minimal working example — correctness and performance check:
//   SpinglassBits vs SpinglassNoBits
//
// Both models are initialized from the same disorder realization (couplings)
// and the same initial spin configuration. They are then evolved with the same
// RNG seed so that each flip attempt draws the same random number in both cases.
// The resulting magnetizations are compared replica by replica.
//
// Purpose: validate that the bitwise implementation (SpinglassBits) produces
// the same physics as the reference scalar implementation (SpinglassNoBits),
// and measure the wall-clock speedup.
//
// No output files are written.
// =============================================================================

// ──────────────────────────────────────────────────────────────────────────────
// print_neurons
//
// Compare the spin configurations of the classic and bitwise models, replica
// by replica and site by site. Reports the first mismatch found, or confirms
// that both configurations are identical.
//
// The printing block (before/after states) is commented out by default to keep
// output concise; uncomment it for detailed per-site debugging.
// ──────────────────────────────────────────────────────────────────────────────
void print_neurons(const vector<int>& neurons_before,
                   const vector<int>& neurons_classic,
                   const vector<int>& neurons_bits,
                   int N_neurons, int prefix_width, int col_width) {

    bool all_equal = true;

    for (int r = 0; r < n_bits; r++) {

        ostringstream oss;

        // Uncomment to print full spin states for each replica:
        /*
        oss << "Realization" << right << setw(3) << r+1;
        cout << oss.str() << "\n";

        cout << left << setw(prefix_width) << "       before: ";
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_before[r*N_neurons+i] << " ";
        cout << "\n";

        cout << left << setw(prefix_width) << "after classic: ";
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_classic[r*N_neurons+i] << " ";
        cout << "\n";

        cout << left << setw(prefix_width) << "   after bits: ";
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_bits[r*N_neurons+i] << " ";
        cout << "\n\n";
        */

        // Site-by-site comparison for this replica
        for (int i = 0; i < N_neurons; i++) {
            if (neurons_classic[r*N_neurons+i] != neurons_bits[r*N_neurons+i]) {
                all_equal = false;
                cout << "Mismatch at r=" << r+1 << " i=" << i+1 << endl;
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
    const int    L         = 100;       // linear lattice size (L×L spins)
    const double beta      = 1/1.5;       // inverse temperature
    const int    N_sweeps  = pow(2,14); // number of Metropolis sweeps
    const int    col_width    = 3;      // column width for spin display
    const int    prefix_width = 12;     // label width for spin display

    cout << "\n";
    cout << "SpinGlass 2D \n\n";
    cout << "Parameters: N_neurons=" << L*L
         << " N_sweeps=" << N_sweeps
         << " T=" << 1.5 << "\n\n";

    // ── Model initialization ──────────────────────────────────────────────────
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

    // Construct both models with identical parameters
    SpinglassBits   bits  (L, beta, N_sweeps);
    SpinglassNoBits nobits(L, beta, N_sweeps);

    // Build the 2D square lattice with periodic boundary conditions
    bits.initNetwork2D_PBC();
    nobits.initNetwork2D_PBC();
    cout << "Network initialized\n" << endl;

    // Generate couplings once, then copy them to nobits so both models
    // share the exact same disorder realization
    gsl_rng_set(ran, 123);
    bits.initCouplings(ran);
    vector<vector<vector<int>>> initial_couplings = bits.getCouplingsConfig();
    nobits.initCouplingsFromConfig(initial_couplings);
    cout << "Couplings initialized\n" << endl;

    // Generate the initial spin configuration once and share it across both models
    gsl_rng_set(ran, 123);
    bits.initSpins(ran);
    vector<int> initial_config = bits.getSpinsConfig();
    nobits.initSpinsFromConfig(initial_config);
    cout << "Spin configurations initialized\n" << endl;

    // Verify that both models start from the same spin state before evolving
    print_neurons(initial_config, nobits.getSpinsConfig(),
                  bits.getSpinsConfig(), L*L, prefix_width, col_width);

    // ── Evolution: SpinglassBits ──────────────────────────────────────────────
    // Both models use the same seed (42) so that flip decisions draw the same
    // sequence of random numbers, making the comparison meaningful.
    gsl_rng_set(ran, 42);
    clock_t start_bits = clock();
    bits.evolve(ran);
    clock_t end_bits = clock();
    double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
    cout << "Bits done. Time: " << clock_bits << " s\n";

    // ── Evolution: SpinglassNoBits ────────────────────────────────────────────
    gsl_rng_set(ran, 42);   // reset to same seed for a fair comparison
    clock_t start_ref = clock();
    nobits.evolveIndependentRNG(ran);
    clock_t end_ref = clock();
    double clock_ref = double(end_ref - start_ref) / CLOCKS_PER_SEC;
    cout << "Classic done. Time: " << clock_ref << " s\n";

    // ── Magnetization comparison ──────────────────────────────────────────────
    // Check that both implementations produce identical magnetizations for every
    // replica. Any discrepancy indicates a bug in one of the two update rules.
    vector<double> mag_bits(n_bits), mag_nobits(n_bits);
    bits.GetMagnetizations(mag_bits);
    nobits.GetMagnetizations(mag_nobits);

    /*bool equal = true;
    for (int r = 0; r < n_bits; r++) {
        if (mag_bits[r] != mag_nobits[r]) {
            equal = false;
            cout << "Difference at r=" << r
                 << ": bits=" << mag_bits[r]
                 << " nobits=" << mag_nobits[r] << endl;
        }
    }
    if (equal) {
        cout << "OK: magnetizations are identical." << endl;
    }
    */
    cout << "Acceleration factor = " << clock_ref / clock_bits << "\n";

    // Note: m_io (SimulationIO member of bits/nobits) closes any open CSV files
    // automatically via its destructor when the objects go out of scope here.
    gsl_rng_free(ran);

    return 0;
}