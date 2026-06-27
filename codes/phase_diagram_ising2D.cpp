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
int main(int argc, char* argv[]) {

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <N_spins>\n";
        cerr << "  e.g. " << argv[0] << " 6400\n";
        return 1;
    }
    const int N_spins = atoi(argv[1]);

    int N_sweeps = 1 << 16;

    // ── Temperature Range ───────────────────
    vector<double> temperatures;

    const double Tc    = 2.0 / log(1.0 + sqrt(2.0));
    const double T_min = 1;
    const double T_max = 3;
    const int    n_T   = 37;
    const double alpha = 2.5;

    for (int i = 0; i < n_T; ++i) {
        double u = (double)i / (n_T - 1);
        double s = 2*u - 1;
        double T;
        if (s >= 0)
            T = Tc + (T_max - Tc) * std::pow(s, alpha);
        else
            T = Tc + (T_min - Tc) * std::pow(-s, alpha);
        temperatures.push_back(T);
    }

    for (double T : temperatures) cout << T << "  ";
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

    // ── Temperature sweep ──────────────────
    for (int i = 0; i < (int)temperatures.size(); ++i) {
        double T     = temperatures[temperatures.size() - i - 1];
        double betaJ = 1.0 / T;

        bits.setBeta(betaJ);

        double dT;
        if (i == 0)
            dT = std::abs(temperatures[temperatures.size()-1] - temperatures[temperatures.size()-2]);
        else if (i == (int)temperatures.size()-1)
            dT = std::abs(temperatures[1] - temperatures[0]);
        else
            dT = std::abs(temperatures[temperatures.size()-i] - temperatures[temperatures.size()-i-2]) / 2.0;

        const double dT_ref   = (T_max - T_min) / (n_T - 1);
        const int    exp_base = 15;
        const int    exp_max  = 19;

        int exp_sweeps = exp_base + (int)std::round(std::log2(dT_ref / dT));
        exp_sweeps     = std::max(exp_base, std::min(exp_max, exp_sweeps));
        N_sweeps       = 1 << exp_sweeps;

        bits.setNSweeps(N_sweeps);

        cout << "N=" << N_spins << "  T=" << fixed << setprecision(4) << T
             << "  dT=" << setprecision(4) << dT
             << "  N_sweeps=2^" << exp_sweeps
             << "  Step " << i+1 << "/" << temperatures.size() << endl;

        ostringstream folder_name;
        folder_name << "../results/Ising/N" << N_spins
                    << "/T_" << fixed << setprecision(3) << T;
        string base_folder = folder_name.str();
        make_dir("../results/Ising/N" + to_string(N_spins));
        make_dir(base_folder);
        bits.SetBaseFolder(base_folder.c_str());

        bits.runSweeps(ran, /*save=*/false, 0, /*shift=*/0);
        cout << " first half done" << endl;
        bits.setNSweeps(N_sweeps);

        string path = "/spins_N" + to_string(N_spins)
                    + "_beta" + SimulationBase::format_beta(betaJ);
        bits.OpenSpinFiles(path.c_str());
        bits.runSweeps(ran, /*save=*/true, 0.005, /*shift=*/N_sweeps);
        cout << " second half done" << endl;
        bits.SaveSpinConfigurations(N_sweeps);
        bits.CloseSpinFiles();
    }

    clock_t end = clock();
    cout << "\nN=" << N_spins << " done in "
         << double(end-start)/CLOCKS_PER_SEC << " s\n";

    gsl_rng_free(ran);
    return 0;
}