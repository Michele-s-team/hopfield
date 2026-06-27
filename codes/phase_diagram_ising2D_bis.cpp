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

    const int N_spins = 150*150;

    int N_sweeps = 1 << 20;

    // ── Temperature Range ───────────────────
    vector<double> temperatures1 = {2.188,2.218,2.24,2.255,2.265,2.269,2.273,2.286,2.299,2.316,2.338};
    vector<double> temperatures2 = {1.00, 1.169, 1.324, 1.465, 1.592, 1.707, 1.809, 1.899, 1.977, 2.045, 2.102, 2.149, };

    for (double T : temperatures2) cout << T << "  ";
    for (double T : temperatures1) cout << T << "  ";
        
    cout << "\n";
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);

    // ── Simulation ─────────────────────────
    clock_t start = clock();

    cout << "\n##############################################\n";
    cout << "N = " << N_spins << "\n";
    cout << "##############################################\n";

    IsingBits bits(N_spins, 1, N_sweeps);
    bits.initNetwork2D_PBC();
    cout << "Network initialized\n" << endl;

    bits.initSpinsBits(ran);
    cout << "Bitwise initialization done" << endl;

    // ── Préchauffage : cooling depuis T=3 jusqu'à T=2.4 ──
    vector<double> warmup_temps = {3.0, 2.75, 2.4};
    int N_warmup = 1 << 18;  // suffisant, pas besoin de sauvegarder
    cout << "Warming up.." << endl;
    bits.setNSweeps(N_warmup);
    for (double T_w : warmup_temps) {
        bits.setBeta(1.0 / T_w);
        bits.runSweeps(ran, /*save=*/false, 0, /*shift=*/0);
        cout << "Warmup T=" << T_w << " done" << endl;
    }
    cout << "Warmup complete\n" << endl;

    // =====================================================
    // First set: always 2^20 sweeps
    // =====================================================

    const int N_sweeps_fixed = 1 << 20;

    for (int i = 0; i < (int)temperatures1.size(); ++i) {

        double T = temperatures1[temperatures1.size() - i - 1];
        double betaJ = 1.0 / T;

        bits.setBeta(betaJ);
        bits.setNSweeps(N_sweeps_fixed);

        cout << "N=" << N_spins
            << "  T=" << fixed << setprecision(4) << T
            << "  N_sweeps=2^20"
            << "  Step " << i + 1 << "/" << temperatures1.size()
            << endl;

        ostringstream folder_name;
        folder_name << "../results/Ising/N" << N_spins
                    << "/T_" << fixed << setprecision(3) << T;

        string base_folder = folder_name.str();
        make_dir("../results/Ising/N" + to_string(N_spins));
        make_dir(base_folder);

        bits.SetBaseFolder(base_folder.c_str());

        bits.runSweeps(ran, false, 0, 0);

        string path = "/spins_N" + to_string(N_spins)
                    + "_beta" + SimulationBase::format_beta(betaJ);

        bits.OpenSpinFiles(path.c_str());
        bits.runSweeps(ran, true, 0.005, N_sweeps_fixed);
        bits.SaveSpinConfigurations(N_sweeps_fixed);
        bits.CloseSpinFiles();
    }


    // =====================================================
    // Second set: adaptive number of sweeps
    // =====================================================

    const double T_min = 1.0;
    const double T_max = 2.149;
    const int n_T = temperatures2.size();

    make_dir("../results/Ising/N" + to_string(N_spins));

    for (int i = 0; i < (int)temperatures2.size(); ++i) {

        double T = temperatures2[temperatures2.size() - i - 1];
        double betaJ = 1.0 / T;

        // ----- compute local dT -----
        double dT;
        int j = temperatures2.size() - 1 - i;

        if (j == 0)
            dT = temperatures2[1] - temperatures2[0];
        else if (j == temperatures2.size()-1)
            dT = temperatures2[j] - temperatures2[j-1];
        else
            dT = (temperatures2[j+1] - temperatures2[j-1]) / 2.0;

        dT = std::abs(dT);

        const double dT_ref = (T_max - T_min) / (n_T - 1);

        const int exp_base = 15;
        const int exp_max  = 19;

        int exp_sweeps = exp_base
                    + (int)std::round(std::log2(dT_ref / dT));

        exp_sweeps = std::max(exp_base,
                    std::min(exp_max, exp_sweeps));

        int N_sweeps = 1 << exp_sweeps;

        bits.setBeta(betaJ);
        bits.setNSweeps(N_sweeps);

        cout << "N=" << N_spins
            << "  T=" << fixed << setprecision(4) << T
            << "  dT=" << dT
            << "  N_sweeps=2^" << exp_sweeps
            << "  Step " << i + 1 << "/" << temperatures2.size()
            << endl;

        ostringstream folder_name;
        folder_name << "../results/Ising/N" << N_spins
                    << "/T_" << fixed << setprecision(3) << T;

        string base_folder = folder_name.str();
        make_dir(base_folder);

        bits.SetBaseFolder(base_folder.c_str());

        bits.runSweeps(ran, false, 0, 0);

        string path = "/spins_N" + to_string(N_spins)
                    + "_beta" + SimulationBase::format_beta(betaJ);

        bits.OpenSpinFiles(path.c_str());
        bits.runSweeps(ran, true, 0.005, N_sweeps);
        bits.SaveSpinConfigurations(N_sweeps);
        bits.CloseSpinFiles();
    }

    gsl_rng_free(ran);
    return 0;
}