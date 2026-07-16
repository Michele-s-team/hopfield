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
//  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O3 -ftlo -Wno-deprecated  -I/Users/michelecastellana/Documents/gillespie/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//  compile on calcsub:
//  clear; clear;  g++ main.cpp src/*.cpp  -llapack -lgsl -lgslcblas -lm -O3 -Wno-deprecated -I /usr/include/gsl/ -I./include/ -o main.o -Wall -DHAVE_INLINE

//  compile on abacus:
//  g++ main.cpp src/*.cpp -I ./include/ -I /mnt/beegfs/home/mcastel1/gsl/include/gsl  -I/mnt/beegfs/home/mcastel1/gsl/include/ -L/mnt/beegfs/home/mcastel1/gsl/lib/ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o main.o -DHAVE_INLINE

// ──────────────────────────────────────────────
// Global bit constants (initialized once at startup)
// ──────────────────────────────────────────────

// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────
int main() {

    // ── Parameters ────────────────────────────
    const double T        = 2.2;
    const int    N_sweeps = pow(2, 20);
    vector<int> sizes = {30, 50, 75, 100, 150, 200};


    // ── Model initialization ───────────────────

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 42);


    for (int i = 0; i < (int)sizes.size(); ++i) {
        int L = sizes[i];
        cout << "L=" << L << "  Step " << i+1 << "/" << sizes.size() << endl;
        IsingBits bits(L, 1.0 / T, N_sweeps);
        bits.initNetwork2D_PBC();
        bits.initSpins(ran);
        auto t_start_bits = chrono::high_resolution_clock::now();
        bits.evolve_save(ran,0.1, "../results/magnetizations/");  
        auto t_end_bits = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> dt_bits = t_end_bits - t_start_bits;

        cout << "  > Bits: " << dt_bits.count() / 1000.0 << " s" << endl;
    }
    
    gsl_rng_free(ran);
    return 0;
}