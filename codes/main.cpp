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

//  compile on mac, without optimization:
//  clear; clear;  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O0 -Wno-deprecated -I/Users/michelecastellana/Documents/office_stuff/work/stages/stage_bastien_dumont_2026/hopfield/codes/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//  compile on mac, with optimization:
//  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated  -I/Users/michelecastellana/Documents/gillespie/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//  compile on calcsub:
//  clear; clear;  g++ main.cpp src/*.cpp  -llapack -lgsl -lgslcblas -lm -O3 -Wno-deprecated -I /usr/include/gsl/ -I./include/ -o main.o -Wall -DHAVE_INLINE

//  compile on abacus:
//  g++ main.cpp src/*.cpp -I ./include/ -I /mnt/beegfs/home/mcastel1/gsl/include/gsl  -I/mnt/beegfs/home/mcastel1/gsl/include/ -L/mnt/beegfs/home/mcastel1/gsl/lib/ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o main.o -DHAVE_INLINE

BitSet BitSet_one;
Bits Bits_one, Bits_zero;

void InitGlobals() {  // question to Michele : it is not so easy to define them in the Bits class 
    for(int i = 0; i<n_bits; i++){
        Bits_one.Set(i, true); 
        Bits_zero.Set(i, false);
    }
}
// ──────────────────────────────────────────────
// Comparaison classic vs bits
// ──────────────────────────────────────────────
void print_neurons(const vector<vector<int>>& neurons_set_before,
                   const vector<vector<int>>& neurons_set_classic,
                   const vector<vector<int>>& neurons_set_bits,
                   int N_neurons, int prefix_width, int col_width) {

    bool all_equal = true;

    for (size_t r = 0; r < neurons_set_classic.size(); r++) {

        ostringstream oss_before;
        oss_before << "Realization" << right << setw(3) << r+1 << "\n       before: ";
        cout << left << setw(prefix_width) << oss_before.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_before[r][i] << " ";
        cout << "\n";

        ostringstream oss_after;
        oss_after << "after classic: ";
        cout << left << setw(prefix_width) << oss_after.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_classic[r][i] << " ";
        cout << "\n";

        ostringstream oss_after_bits;
        oss_after_bits << "   after bits: ";
        cout << left << setw(prefix_width) << oss_after_bits.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_bits[r][i] << " ";
        cout << "\n\n";

        for (int i = 0; i < N_neurons; i++) {
            if (neurons_set_classic[r][i] != neurons_set_bits[r][i]) {
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

    const int L        = 50;
    const int N_sweeps = 20000;// pow(2,18);
    double BJ_dummy=0.01;

    InitGlobals();

    cout << "[main] Parameters: N_neurons=" << L*L
         << " n_bits=" << n_bits
         << " N_sweeps=" << N_sweeps << endl;

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);

    // ── Configuration initiale des spins ──────
    IsingBits bits(L, BJ_dummy, N_sweeps);
    IsingNoBits nobits(L, BJ_dummy, N_sweeps);

    bits.initConnections2D();
    nobits.initConnections2D();

    gsl_rng_set(ran, 123);
    bits.initSpins(ran);
    vector<vector<int>> initial_config = bits.getSpinsConfig();
    nobits.initSpinsFromConfig(initial_config);
    
    //gsl_rng_set(ran, 456);
    /* // ── Génération unique des exponentielles ──
    vector<vector<double>> exp_base(N_sweeps, vector<double>(L*L));
    
    for (int step = 0; step < N_sweeps; step++)
        for (int i = 0; i < L*L; i++)
            exp_base[step][i] = gsl_ran_exponential(ran, 1.0);
    */

    vector<double> magnetizations_bits(n_bits);
    vector<double> magnetizations_nobits(n_bits);

    bool equal;
    const double T_min  = 1.0;
    const double T_max  = 5.0;
    const double step_T = 0.05;

    const int totalSteps = (int)((T_max - T_min) / step_T) + 1;
    int step = 1;

    for (int i = 0; i < totalSteps; i++, step++) {
        double BJ_loop = 1.0 / (T_max - i * step_T);
        cout << "BJ=" << BJ_loop << " Step " << step << "/" << totalSteps << endl;

        // --- Test approche BITS ---
        bits.setBJ(BJ_loop);
        bits.initSpinsFromConfig(initial_config);
        gsl_rng_set(ran, 123);
        auto start_bits = std::chrono::high_resolution_clock::now();
        bits.initRandomNumbers(ran);
        bits.evolve_monolithic();
        auto end_bits = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration_bits = end_bits - start_bits;

        // --- Test approche NOBITS ---
        nobits.setBJ(BJ_loop);
        nobits.initSpinsFromConfig(initial_config);
        gsl_rng_set(ran, 123);

        auto start_nobits = std::chrono::high_resolution_clock::now();
        nobits.initRandomNumbers(ran);
        nobits.evolve_monolithic();
        auto end_nobits = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration_nobits = end_nobits - start_nobits;


        // --- Affichage des résultats ---
        cout << "  > Temps Bits   : " << duration_bits.count()/1000 << " s" << endl;
        cout << "  > Temps NoBits : " << duration_nobits.count()/1000 << " s" << endl;
        cout << "  > Speedup      : " << duration_nobits.count() / duration_bits.count() << "x" << endl;
        cout << "------------------------------------------" << endl;

        bits.GetMagnetizations(magnetizations_bits);
        nobits.GetMagnetizations(magnetizations_nobits);

        for (int r = 0; r < n_bits; r++) {
            if (magnetizations_bits[r] != magnetizations_nobits[r]) {
                equal = false;

                cout << "Difference at r = " << r
                        << " : bits = " << magnetizations_bits[r]
                        << ", nobits = " << magnetizations_nobits[r]
                        << endl;
            }
        }
        if (equal) {cout << "OK: magnetizations vectors are identical." << endl;}
        
        bits.SaveMagnetizations("../results/magnetizations_bits.csv");
        nobits.SaveMagnetizations("../results/magnetizations_nobits.csv");
    }

    gsl_rng_free(ran);
    return 0;

    /*
    InitGlobals();

    const int    L          = 40;
    const double BJ         = 0.01;
    const int    N_sweeps   = 1000;
    const int    col_width  = 3;
    const int    prefix_width = 12;

    cout << "[main] Parameters: N_neurons=" << L*L
         << " n_bits=" << n_bits
         << " N_sweeps=" << N_sweeps
         << " Beta*J=" << BJ << endl;

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);

    // ── Bitwise ───────────────────────────────
    // Réinitialise avec la même graine pour avoir le même état initial
    IsingBits bits(L, BJ, N_sweeps);
    bits.init(ran);

    clock_t start_bits = clock();
    bits.evolve();
    clock_t end_bits = clock();
    double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
    cout << "Bits done.    Time: " << clock_bits << " s\n";

    // ── Classic ───────────────────────────────
    gsl_rng_set(ran, 123);
    IsingNoBits classic(L, BJ, N_sweeps);
    classic.init(ran);
    vector<vector<int>> neurons_before = classic.getState();

    clock_t start_ref = clock();
    classic.evolve();
    clock_t end_ref = clock();
    double clock_ref = double(end_ref - start_ref) / CLOCKS_PER_SEC;
    cout << "Classic done. Time: " << clock_ref << " s\n";



    // ── Comparaison ───────────────────────────
    print_neurons(neurons_before,
                  classic.getState(),
                  bits.getState(),
                  L*L, prefix_width, col_width);

    cout << "Total clock_ref:    " << clock_ref  << " s\n";
    cout << "Total clock_bits:   " << clock_bits << " s\n";
    cout << "Acceleration factor = " << clock_ref / clock_bits << "\n";

    gsl_rng_free(ran);
    return 0;

    */
}