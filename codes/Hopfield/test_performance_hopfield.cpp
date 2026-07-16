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


#include "hopfield_model.hpp"
#include "hopfield_nobits.hpp"
#include "hopfield_bits.hpp"

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
//   - sets the inverse temperature beta = 1/T
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

void make_dir(const string& path) {
    mkdir(path.c_str(), 0755);
}


// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────

int main(){

    SimulationIO IO;
    struct timespec t_init, t_final, t_start, t_end, t0, t1;

    const int N_sweeps = 1 << 5;

    vector<int> N_vals = {
        //10 * 10, 20 * 20, 
        32 * 32
        //80 * 80
    };

    vector<double> alpha_vals = {//
        0.15
        //,     0.10,  0.15
         //, 0.20
         };

    vector<double> temperatures = {1.0,2.0, 3.0, 4.0, 5.0, 6.0,  7.0, 8.0, 9.0, 10.0};

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng* ran_bits = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng* ran_nobits = gsl_rng_alloc(gsl_rng_gfsr4);


    ofstream out("../results/Hopfield/test_speedup/speedup.csv");

    out << "N,P,alpha,T,t_bits,t_nobits,ratio\n";
    out << fixed << setprecision(6);

    clock_gettime(CLOCK_MONOTONIC, &t_init);

    // ============================================================
    // GLOBAL COUNTER (AJOUT)
    // ============================================================
    double total_time_global = 0.0;

    for (int N_spins : N_vals){
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        // ============================================================
        // N COUNTER (AJOUT)
        // ============================================================
        double total_time_N = 0.0;

        cout << "\n==================================================\n";
        cout << "Lattice size : N = " << N_spins << '\n';
        cout << "==================================================\n";

        for (double alpha : alpha_vals)  {
            int P = max(1, static_cast<int>(round(alpha * N_spins)));

            // ============================================================
            // ALPHA COUNTER (AJOUT)
            // ============================================================
            double total_time_alpha = 0.0;

            cout << "\n------------------------------------------\n";
            cout << "alpha = " << alpha
                      << "   P = " << P << '\n';
            cout << "------------------------------------------\n";

            HopfieldBits bits(N_spins, 1, N_sweeps, P);
            HopfieldNoBits nobits(N_spins, 1, N_sweeps, P);

            for (size_t i = 0; i < temperatures.size(); ++i) {
                double T = temperatures[i];
                double beta = 1.0 / T;

                bits.initNetworkFullyConnected();
                nobits.initNetworkFullyConnected();

                cout << "Network initialized" <<endl;

                gsl_rng_set(ran, 123);

                bits.initSpinsBits(ran);
                bits.toCanonical();
                auto initial_config_bits = bits.getSpinsConfig();
                nobits.initSpinsFromConfig(initial_config_bits);
                auto initial_config_nobits= nobits.getSpinsConfig();

                IO.check_equality_configs(initial_config_bits, initial_config_nobits, initial_config_bits, N_spins);

        

                cout << "Initial Configuration initialized" <<endl;

                bits.initPatternsBits(ran);
                cout << "Patterns initialized (bitwise)" <<endl;
                auto patterns = bits.getPatternsBitsToCanonical();
                nobits.initPatternsFromConfig(patterns);

                cout << "Patterns initialized" <<endl;

                bits.setBeta(beta);

                nobits.setBeta(beta);

                cout
                    << "N = " << N_spins
                    << " | alpha = " << alpha
                    << " | T = " << T
                    << " | " << i + 1 << "/" << temperatures.size()
                    << endl;

                // ----------------------------------------------
                // Bitwise implementation
                // ----------------------------------------------

                cout << "bitwise evolution ready"<< endl;
                gsl_rng_set(ran_bits, 42);    
                clock_gettime(CLOCK_MONOTONIC, &t0);
                bits.runSweeps(ran_bits, false, 0, 0);
                clock_gettime(CLOCK_MONOTONIC, &t1);
                bits.toCanonical();
                auto final_config_bits = bits.getSpinsConfig();

                double t_bits =
                    (t1.tv_sec - t0.tv_sec) +
                    (t1.tv_nsec - t0.tv_nsec) * 1e-9;

                cout << "   bitwise   : " << t_bits << " s\n";

                // ----------------------------------------------
                // Classical implementation
                // ----------------------------------------------
                gsl_rng_set(ran_bits, 42);
                clock_gettime(CLOCK_MONOTONIC, &t0);
                nobits.runSweepsSharedRNG(ran_bits, false, 0);
                clock_gettime(CLOCK_MONOTONIC, &t1);
                auto final_config_nobits = nobits.getSpinsConfig();

                double t_nobits =
                    (t1.tv_sec - t0.tv_sec) +
                    (t1.tv_nsec - t0.tv_nsec) * 1e-9;

                // ----------------------------------------------
                // Check equality
                // ----------------------------------------------

                IO.check_equality_configs(initial_config_bits, final_config_nobits, final_config_bits, N_spins);

                // ----------------------------------------------
                // Compute speedup
                // ----------------------------------------------

                double ratio =
                    (t_bits > 0.0) ? t_nobits / t_bits : 0.0;

                cout
                    << "   classical : " << t_nobits << " s\n"
                    << "   speedup   : " << ratio << '\n';

                out << N_spins << ","
                    << P << ","
                    << alpha << ","
                    << T << ","
                    << t_bits << ","
                    << t_nobits << ","
                    << ratio << "\n";

                out.flush();

                // ============================================================
                // TIME ACCUMULATION (AJOUT)
                // ============================================================
                double t_total = t_bits + t_nobits;
                total_time_alpha += t_total;
                total_time_N += t_total;
                total_time_global += t_total;
            }

            // ============================================================
            // PRINT ALPHA TIME (AJOUT)
            // ============================================================
            cout << "\n[alpha timing] alpha = "
                 << alpha
                 << " | total time = "
                 << total_time_alpha
                 << " s\n";
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);

        double elapsed_N =
            (t_end.tv_sec - t_start.tv_sec) +
            (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

        cout << "\n[N timing] N = "
             << N_spins
             << " | accumulated = "
             << total_time_N
             << " s\n";

        cout << "Finished N = "
             << N_spins
             << " in "
             << elapsed_N
             << " s\n";
    }

    clock_gettime(CLOCK_MONOTONIC, &t_final);

    double elapsed =
        (t_final.tv_sec - t_init.tv_sec) +
        (t_final.tv_nsec - t_init.tv_nsec) * 1e-9;

    cout << "\n=========================================\n";
    cout << "Total execution time : "
              << elapsed
              << " s\n";

    cout << "Total accumulated compute time : "
         << total_time_global
         << " s\n";

    cout << "=========================================\n";

    out.close();
    gsl_rng_free(ran);

    return 0;
}