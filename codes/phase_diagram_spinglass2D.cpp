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
#include "spinglass_bits.hpp"
#include "spinglass_nobits.hpp"

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
// phase_diagram.cpp
//
// Simulates the 2D Ising model using a bitwise Metropolis algorithm
// (IsingBits) on a square lattice of size L×L with periodic boundary
// conditions, across a range of temperatures.
//
// For each temperature T:
//   - sets the inverse temperature betaJ = 1/T
//   - reinitializes spins from a fixed reference configuration
//   - runs N_sweeps Metropolis sweeps, saving magnetizations at regular
//     intervals to CSV files (one per realization, named L{L}_r{r}.csv)
//
// The temperature grid is defined by several segments with different
// step sizes, with finer resolution near the critical point Tc ≈ 2.269.
//
// An optional classic (non-bitwise) simulation is available for comparison
// and correctness checking (see commented sections).
//
// Output: ../results/magnetizations/L{L}_r{r}.csv  (columns: T, N, m)
// =============================================================================


// ──────────────────────────────────────────────
// Print and compare spin configurations across realizations
// ──────────────────────────────────────────────
void print_neurons(const vector<int>& neurons_before,
                   const vector<int>& neurons_classic,
                   const vector<int>& neurons_bits,
                   int N_neurons, int prefix_width, int col_width) {

    bool all_equal = true;

    for (int r = 0; r < n_bits; r++) {

        ostringstream oss;
        //Uncomment to print the spins when comparing
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

// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────
int main() {

    // ── Parameters ────────────────────────────
    const int    L        = 25;
    const int    N_sweeps = pow(2, 18);
    // ── Temperature Range ───────────────────

    vector<double> temperatures;

    const double T_min    = 0.1;
    const double step_1   = 0.1;
    const double T_1      = 1.5;
    const double step_2   = 0.05;
    const double T_2      = 2.0;
    const double step_3   = 0.02;
    const double T_3      = 2.5;
    const double step_4   = 0.05;
    const double T_4      = 3.0;
    const double step_5   = 0.1;
    const double T_max    = 4.0;


    auto add_segment = [&](double a, double b, double h, bool include_start){
        int n_start = include_start ? 0 : 1;
        int n_end = (int)((b - a)/h + 0.5);

        for (int n = n_start; n <= n_end; ++n)
            temperatures.push_back(a + n*h);
    };
    /*
    add_segment(T_min, T_1, step_1, true);
    add_segment(T_1,  T_2, step_2, false);
    add_segment(T_2,  T_3, step_3, false);
    add_segment(T_3,  T_4, step_4, false);
    add_segment(T_4,  T_max, step_5, false)
    */
    temperatures.push_back(1.5);
    cout << "[main] Parameters: N_spins=" << L*L
         << "  N_sweeps=" << N_sweeps
         << "\nTemperatures: " <<endl;

    for (double T : temperatures) cout <<T <<"  ";
    cout <<" \n\n";

    // ── Model initialization ───────────────────
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

    // Initialize both models with the same lattice size, inverse temperature, and sweep count
    SpinglassBits   bits  (L, 1.0 / T_min, N_sweeps);
    SpinglassNoBits nobits(L, 1.0 / T_min, N_sweeps);

    // Build the 2D periodic boundary condition network for both models
    bits.initNetwork2D_PBC();
    nobits.initNetwork2D_PBC();
    cout << "Network initialized\n" << endl;

    // Generate couplings once with a fixed seed, then share them across both models
    // to ensure identical disorder realizations
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

    // ── Temperature sweep ──────────────────────────────────────────────────────
    // Iterate over the temperature schedule; at each step both models are reset
    // to the same initial spin configuration and run with the same RNG seed,
    // so that any difference in magnetizations reflects algorithmic divergence only.
    for (int i = 0; i < temperatures.size(); ++i) {

        double T = temperatures[i];
        const double beta = 1.0 / T;
        cout << "T=" << T << "  Step " << i+1 << "/" << temperatures.size() << endl;

        // --- Bitwise simulation ---
        bits.setbeta(beta);
        bits.initSpinsFromConfig(initial_config);   // reset to shared initial state
        gsl_rng_set(ran, 123);                      // reset RNG for reproducibility

        auto t_start_bits = chrono::high_resolution_clock::now();
        bits.evolve_save(ran, 0.1, "../results/magnetizations/magnetizations_bits");
        auto t_end_bits = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> dt_bits = t_end_bits - t_start_bits;
        cout << "  > Bits: " << dt_bits.count() / 1000.0 << " s" << endl;

        // --- Classic (scalar) simulation ---
        nobits.setbeta(beta);
        nobits.initSpinsFromConfig(initial_config); // reset to the same initial state
        gsl_rng_set(ran, 123);                      // same seed as bits for fair comparison

        auto t_start_nobits = chrono::high_resolution_clock::now();
        nobits.evolveSharedRNG(ran);
        auto t_end_nobits = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> dt_nobits = t_end_nobits - t_start_nobits;
        cout << "  > NoBits: " << dt_nobits.count() / 1000.0 << " s" << endl;

        // Report the wall-clock speedup of the bitwise approach
        cout << "  > Speedup: " << dt_nobits.count() / dt_bits.count() << "x" << endl;
        cout << "------------------------------------------" << endl;

        // --- Magnetization comparison ---
        // Verify that both implementations produce identical magnetizations for every replica.
        // Any discrepancy indicates a bug in one of the two update rules.
        vector<double> mag_bits(n_bits), mag_nobits(n_bits);
        bits.GetMagnetizations(mag_bits);
        nobits.GetMagnetizations(mag_nobits);

        bool equal = true;
        for (int r = 0; r < n_bits; r++) {
            if (mag_bits[r] != mag_nobits[r]) {
                equal = false;
                cout << "Difference at r=" << r
                    << ": bits=" << mag_bits[r]
                    << " nobits=" << mag_nobits[r] << endl;
            }
        }
        if (equal) cout << "OK: magnetizations are identical." << endl;
    }

    // Release the RNG resource
    gsl_rng_free(ran);
    return 0;
}
