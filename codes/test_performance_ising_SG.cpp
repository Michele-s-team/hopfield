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
#include "simulation_base.hpp"
#include "ising_model.hpp"
#include "ising_nobits.hpp"
#include "ising_bits.hpp"

#include "spinglass_model.hpp"
#include "spinglass_nobits.hpp"
#include "spinglass_bits.hpp"

#include <sys/stat.h> // for mkdir

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

void make_dir(const string& path) {
    mkdir(path.c_str(), 0755);
}


// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────
int main() {


    struct timespec t_init_1, t_final_1, t_init_2, t_final_2, t_start, t_end ,t0, t1;

    // ── Parameters ────────────────────────────
    int       N_sweeps = 1 << 16;

    // ── Lattice sizes ──────────────────────────────
    vector<int> N_vals = {60*60};//10 *10, 15*15, 20*20, 25*25, 30*30, 40*40, 50*50, 60*60, 70*70, 80*80};
    
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

    /*

    // ── Temperature Range ───────────────────
    vector<double> temperatures1;
    const double T_min1 = 9;
    const double T_max1 = 9;
    const double step1  = 1;
    int    n_T1   = (int)((T_max1 - T_min1) / step1) + 1;

    for (int i = 0; i < n_T1; ++i)
        temperatures1.push_back(round((T_min1 + i * step1) * 1000.0) / 1000.0);

    for (double T : temperatures1) cout <<T <<"  ";
        cout <<" \n";

    // ── Output file ────────────────────────────
    ofstream out_ising_csv("../results/Ising/speedup.csv");
    out_ising_csv << "N,T,t_bits,t_nobits,ratio\n";
    out_ising_csv << fixed << setprecision(6);

    clock_gettime(CLOCK_MONOTONIC, &t_init_1);

    for (int N_spins : N_vals) {

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        cout << "\n##############################################\n";
        cout << "N = " << N_spins << "\n";
        cout << "##############################################\n";

        IsingBits   bits  (N_spins, 1, N_sweeps);
        IsingNoBits nobits(N_spins, 1, N_sweeps);

        for (int i = 0; i < (int)temperatures1.size(); ++i) {
            double T     = temperatures1[i];
            double betaJ = 1.0 / T;

            bits.initNetwork2D_PBC();
            nobits.initNetwork2D_PBC();

            gsl_rng_set(ran, 123);
            bits.initSpins(ran);
            vector<int> initial_config = bits.getSpinsConfig();
            nobits.initSpinsFromConfig(initial_config);

            bits.setBeta(betaJ);
            nobits.setBeta(betaJ);

            cout << "N=" << N_spins << "  T=" << T
                 << "  Step " << i+1 << "/" << temperatures1.size() << endl;

            // ── Bits timing ──────────────────
            bits.fromCanonical(); 

            clock_gettime(CLOCK_MONOTONIC, &t0);
            bits.runSweeps(ran, false, 0, 0);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double t_bits = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            cout << "  bits:   " << t_bits << " s" << endl;

            // ── NoBits timing ─────────────────
            clock_gettime(CLOCK_MONOTONIC, &t0);
            nobits.runSweepsIndependentRNG(ran, false, 0);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double t_nobits = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            cout << "  nobits: " << t_nobits << " s" << endl;

            double ratio = (t_bits > 0) ? t_nobits / t_bits : 0.0;
            cout << "  ratio:  " << ratio << endl;

            out_ising_csv << N_spins << "," << T << ","
                    << t_bits  << "," << t_nobits << ","
                    << ratio   << "\n";
            out_ising_csv.flush();
        }
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        double elapsed_N = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
        cout << "\nN = " << N_spins << " done, " << elapsed_N << " s\n";
    }

    clock_gettime(CLOCK_MONOTONIC, &t_final_1);
    double elapsed1 = (t_final_1.tv_sec  - t_init_1.tv_sec) + (t_final_1.tv_nsec - t_init_1.tv_nsec) * 1e-9;

    cout << "total time: " << elapsed1 <<endl;

    out_ising_csv.close();

    */

    N_sweeps = 1 << 14;

    // ── Temperature Range ───────────────────
    vector<double> temperatures2;
    const double T_min2 = 3.5;
    const double T_max2 = 3.5;
    const double step2 = 1;
    int    n_T2   = (int)((T_max2 - T_min2) / step2) + 1;

    for (int i = 0; i < n_T2; ++i)
        temperatures2.push_back(round((T_min2 + i * step2) * 1000.0) / 1000.0);


    for (double T : temperatures2) cout <<T <<"  ";
        cout <<" \n";

    // ── Output file ────────────────────────────
    ofstream out_SG_csv("../results/SpinGlass/speedup.csv");
    out_SG_csv << "N,T,t_bits,t_nobits,ratio\n";
    out_SG_csv << fixed << setprecision(6);


    clock_gettime(CLOCK_MONOTONIC, &t_init_2);


    for (int N_spins : N_vals) {

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        cout << "\n##############################################\n";
        cout << "N = " << N_spins << "\n";
        cout << "##############################################\n";

        SpinGlassBits   bits  (N_spins, 1, N_sweeps);
        SpinGlassNoBits nobits(N_spins, 1, N_sweeps);

        for (int i = 0; i < (int)temperatures2.size(); ++i) {
            double T     = temperatures2[i];
            double betaJ = 1.0 / T;

            bits.initNetwork2D_PBC();
            nobits.initNetwork2D_PBC();

            gsl_rng_set(ran, 123);
            bits.initSpins(ran);
            vector<int> initial_config = bits.getSpinsConfig();
            nobits.initSpinsFromConfig(initial_config);

            bits.initCouplings(ran);
            vector<vector<vector<int>>> couplings = bits.getCouplingsConfig();
            nobits.initCouplingsFromConfig(couplings);

            bits.setBeta(betaJ);
            nobits.setBeta(betaJ);

            cout << "N=" << N_spins << "  T=" << T
                 << "  Step " << i+1 << "/" << temperatures2.size() << endl;

            // ── Bits timing ──────────────────
            bits.fromCanonical();

            clock_gettime(CLOCK_MONOTONIC, &t0);
            bits.runSweeps(ran, false, 0);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double t_bits = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            cout << "  bits:   " << t_bits << " s" << endl;

            // ── NoBits timing ─────────────────
            clock_gettime(CLOCK_MONOTONIC, &t0);
            nobits.runSweepsIndependentRNG(ran, false, 0);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double t_nobits = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            cout << "  nobits: " << t_nobits << " s" << endl;

            double ratio = (t_bits > 0) ? t_nobits / t_bits : 0.0;
            cout << "  ratio:  " << ratio << endl;

            out_SG_csv << N_spins << "," << T << ","
                    << t_bits  << "," << t_nobits << ","
                    << ratio   << "\n";
            out_SG_csv.flush();
        }
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        double elapsed_N = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
        cout << "\nN = " << N_spins << " done, " << elapsed_N << " s\n";
    }

    clock_gettime(CLOCK_MONOTONIC, &t_final_2);
    double elapsed2 = (t_final_2.tv_sec  - t_init_2.tv_sec) + (t_final_2.tv_nsec - t_init_2.tv_nsec) * 1e-9;
    cout << "total time: " << elapsed2 <<endl;

    out_SG_csv.close();
    gsl_rng_free(ran);

    return 0;
}