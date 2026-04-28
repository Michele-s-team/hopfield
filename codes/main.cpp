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

BitSet BitSet_one; // really strange that we need to define this for the operator -= of BitSet 

//  compile on mac, without optimization:
//  clear; clear;  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O0 -Wno-deprecated -I/Users/michelecastellana/Documents/office_stuff/work/stages/stage_bastien_dumont_2026/hopfield/codes/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//  compile on mac, with optimization:
//  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated  -I/Users/michelecastellana/Documents/gillespie/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//  compile on calcsub:
//  clear; clear;  g++ main.cpp src/*.cpp  -llapack -lgsl -lgslcblas -lm -O3 -Wno-deprecated -I /usr/include/gsl/ -I./include/ -o main.o -Wall -DHAVE_INLINE

//  compile on abacus:
//  g++ main.cpp src/*.cpp -I ./include/ -I /mnt/beegfs/home/mcastel1/gsl/include/gsl  -I/mnt/beegfs/home/mcastel1/gsl/include/ -L/mnt/beegfs/home/mcastel1/gsl/lib/ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o main.o -DHAVE_INLINE

// ──────────────────────────────────────────────
// Global bit constants (initialized once at startup)
// ──────────────────────────────────────────────

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

    /*

    // ── Parameters ────────────────────────────
    const int    L        = 100;
    const int    N_sweeps = pow(2, 18);
    const double T_min    = 0.05;
    const double T_max    = 5.0;
    const double dT       = 0.05;
    const int    totalSteps = (int)((T_max - T_min) / dT) + 1;

    InitGlobals();

    cout << "[main] Parameters: N_spins=" << L*L
         << "  n_bits=" << n_bits
         << "  N_sweeps=" << N_sweeps << "\n" << endl;

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

    // ── Model initialization ───────────────────
    IsingBits   bits  (L, 1.0 / T_min, N_sweeps);
    IsingNoBits nobits(L, 1.0 / T_min, N_sweeps);

    bits.initConnections2D_PBC();
    nobits.initConnections2D_PBC();
    cout << "Network initialized\n" << endl;

    gsl_rng_set(ran, 123);
    bits.initSpins(ran);
    vector<int> initial_config = bits.getSpinsConfig();
    nobits.initSpinsFromConfig(initial_config);
    cout << "Spin configurations initialized\n" << endl;

    // ── Temperature sweep ──────────────────────
    bits.OpenCSVFiles("../results/test/magnetizations/magnetizations_bits.csv");

    for (int i = 0; i < totalSteps; i++) {
        const double T  = T_min + i * dT;
        const double BJ = 1.0 / T;
        cout << "T=" << T << "  Step " << i+1 << "/" << totalSteps << endl;

        // --- Bitwise simulation ---
        bits.setBJ(BJ);
        bits.initSpinsFromConfig(initial_config);
        gsl_rng_set(ran, 123);

        auto t_start_bits = chrono::high_resolution_clock::now();
        bits.evolve(ran);
        auto t_end_bits = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> dt_bits = t_end_bits - t_start_bits;

        bits.SaveMagnetizations(N_sweeps);
        cout << "  > Bits: " << dt_bits.count() / 1000.0 << " s" << endl;

        /*
        // --- Classic simulation (uncomment to compare) ---
        nobits.setBJ(BJ);
        nobits.initSpinsFromConfig(initial_config);
        gsl_rng_set(ran, 123);

        auto t_stard_nobits = chrono::high_resolution_clock::now();
        nobits.evolve(ran);
        auto t_end_nobits = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> dt_nobits = t_end_nobits - t_stard_nobits;

        cout << "  > NoBits: " << dt_nobits.count() / 1000.0 << " s" << endl;
        cout << "  > Speedup: " << dt_nobits.count() / dt_bits.count() << "x" << endl;
        cout << "------------------------------------------" << endl;

        // --- Magnetization comparison ---
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

    bits.CloseCSVFiles();
    gsl_rng_free(ran);
    */

    // ══════════════════════════════════════════
    // LEGACY: small-scale correctness check
    // (bits vs classic on a small L, few sweeps)
    // ══════════════════════════════════════════

    const int    L_test        = 100;
    const double BJ_test       = 0.01;
    const int    N_sweeps_test = 10000;
    const int    col_width     = 3;
    const int    prefix_width  = 12;

    cout << "[main] Parameters: N_neurons=" << L_test*L_test
         << " N_sweeps=" << N_sweeps_test
         << " Beta*J=" << BJ_test <<"\n";

    gsl_rng* ran_test = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran_test, 123);

    IsingBits bits_test(L_test, BJ_test, N_sweeps_test);
    IsingNoBits classic_test(L_test, BJ_test, N_sweeps_test);

    bits_test.initConnections2D_PBC();
    classic_test.initConnections2D_PBC();
    cout << "Network initialized\n" << endl;
    
    bits_test.initSpins(ran_test);
    vector<int> initial_config = bits_test.getSpinsConfig();    
    classic_test.initSpinsFromConfig(initial_config);
    cout << "Spin configurations initialized\n" << endl;

    print_neurons(initial_config, classic_test.getSpinsConfig(), bits_test.getSpinsConfig(), L_test*L_test, prefix_width, col_width);

    gsl_rng_set(ran_test, 45);
    clock_t start_bits = clock();
    bits_test.evolve(ran_test);
    clock_t end_bits = clock();
    double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
    cout << "Bits done. Time: " << clock_bits << " s\n";

    gsl_rng_set(ran_test, 45); 
    clock_t start_ref = clock();
    classic_test.evolve(ran_test);
    clock_t end_ref = clock();
    double clock_ref = double(end_ref - start_ref) / CLOCKS_PER_SEC;
    cout << "Classic done. Time: " << clock_ref << " s\n";

    print_neurons(initial_config, classic_test.getSpinsConfig(), bits_test.getSpinsConfig(), L_test*L_test, prefix_width, col_width);

    cout << "Acceleration factor = " << clock_ref / clock_bits << "\n";

    gsl_rng_free(ran_test);

    return 0;
}