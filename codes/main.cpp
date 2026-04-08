#include <iostream>
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
//this has to be uncommented to run on mesopsl
#include <stdint.h>
#include <chrono>


#include "gsl_rng.h"
#include "gsl_math.h"
#include "gsl_sf_log.h"
#include "gsl_randist.h"

#include "main.hpp"
#include "int.hpp"
#include "system_bits.hpp"
#include "system_nobits.hpp"


/*
 compile on mac
 compile without optimization
*/
// clear; clear;  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O0 -Wno-deprecated -I/Users/michelecastellana/Documents/office_stuff/work/stages/stage_bastien_dumont_2026/hopfield/codes/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE
 

/*
 compile with optimization
 */
// g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated  -I/Users/michelecastellana/Documents/gillespie/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//compile on calcsub
//clear; clear;  g++ main.cpp src/*.cpp  -llapack -lgsl -lgslcblas -lm -O3 -Wno-deprecated -I /usr/include/gsl/ -I./include/ -o main.o -Wall -DHAVE_INLINE
//compile on abacus
//g++ main.cpp src/*.cpp -I ./include/ -I /mnt/beegfs/home/mcastel1/gsl/include/gsl  -I/mnt/beegfs/home/mcastel1/gsl/include/ -L/mnt/beegfs/home/mcastel1/gsl/lib/ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o main.o -DHAVE_INLINE

//run with
//./main.o -N 128 -T 1 -S 5 -s 0 -o /Users/michelecastellana/Desktop
/*
 N is the total perticle number in the Frank model
 S is log_10(total number of iterations)
 s is the seed of the random-number generator
 o is the path where to store the results
 */

/*
 void UnsignedInt::SetRandom(unsigned long long int seed){
 
 int i;
 gsl_rng* ran;
 
 ran = gsl_rng_alloc(gsl_rng_gfsr4);
 gsl_rng_set(ran, seed);
 
 
 for(i=0; i<b.size(); i++){
 
 
 }
 
 }
 */

/*
 BitSet BitSet_one;
Bits Bits_one, Bits_zero;

// Constants
const int MAX_VALUE = 127;   // maximum random value
const int N_SETS = 6;        // number of sets of integers

int main() {
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 12345);

    for (int game = 0; game < N_SETS; game++) {

        vector<unsigned long long> A(n_bits);

        // Generate random integers
    for (int s = 0; s < n_bits; s++) {
        A[s] = gsl_rng_uniform_int(ran, MAX_VALUE + 1);
    }
        UnsignedInt A_bit(MAX_VALUE), B_bit(MAX_VALUE);
        A_bit.SetFromVector(&A);
        string print = "A in the BitSet format";
        A_bit.PrintBase10(cout);

        // Reconstruct vector from Int
        vector<unsigned long long> C(n_bits); // resize before filling
        A_bit.GetBase10(C);

        // Print results and compare
        std::cout << "\n===== Game " << game << " =====\n";
        for (int s = 0; s < n_bits; s++) {
            cout << "s=" << s
                      << " | Input =" << A[s]
                      << " | Output =" << C[s];
            if (A[s] != C[s]) {
                cout << "  <-- ERROR";
            }
            else{cout << "   OK"; }
            cout << std::endl;
        }
    }


    return 0;
}
*/



//all entries of BitSet_one are equal to 1
BitSet BitSet_one;
Bits Bits_one, Bits_zero;


const int N_sets = 50;
const long long MAX_VALUE = 100000;

int main() {

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 12345);

    // =========================================
    // INITIALISATION (NON MESURÉE)
    // =========================================

    // --- Bitsets ---
    vector<UnsignedInt> A_bit;
    vector<UnsignedInt> B_bit;

    A_bit.reserve(N_sets);
    B_bit.reserve(N_sets);

    for (int i = 0; i < N_sets; i++) {

        vector<unsigned long long> A(n_bits), B(n_bits);

        for (int s = 0; s < n_bits; s++) {
            A[s] = gsl_rng_uniform_int(ran, MAX_VALUE + 1);
            B[s] = gsl_rng_uniform_int(ran, MAX_VALUE + 1);
        }

        UnsignedInt A_tmp(MAX_VALUE), B_tmp(MAX_VALUE);
        A_tmp.SetFromVector(&A);
        B_tmp.SetFromVector(&B);

        A_bit.push_back(A_tmp);
        B_bit.push_back(B_tmp);
    }

    // --- Entiers scalaires ---
    vector<long long> A_ref(N_sets * n_bits);
    vector<long long> B_ref(N_sets * n_bits);

    for (int i = 0; i < N_sets * n_bits; i++) {
        A_ref[i] = gsl_rng_uniform_int(ran, MAX_VALUE + 1);
        B_ref[i] = gsl_rng_uniform_int(ran, MAX_VALUE + 1);
    }

    // =========================================
    // BENCHMARK BITSET
    // =========================================

    double clock_bitset = 0.0;

    auto start_bit = chrono::high_resolution_clock::now();

    for (int i = 0; i < N_sets; i++) {
        A_bit[i] += &B_bit[i];
    }

    auto end_bit = chrono::high_resolution_clock::now();
    chrono::duration<double> dt_bit = end_bit - start_bit;
    clock_bitset = dt_bit.count();

    // =========================================
    // BENCHMARK SCALAIRE
    // =========================================

    double clock_ref = 0.0;

    auto start_ref = chrono::high_resolution_clock::now();

    for (int i = 0; i < N_sets * n_bits; i++) {
        A_ref[i] += B_ref[i];
    }

    auto end_ref = chrono::high_resolution_clock::now();
    chrono::duration<double> dt_ref = end_ref - start_ref;
    clock_ref = dt_ref.count();

    // =========================================
    // RESULTATS
    // =========================================

    cout << "\nTotal clock_bitset: " << clock_bitset << " s\n";
    cout << "Total clock_ref:    " << clock_ref << " s\n";

    gsl_rng_free(ran);
    return 0;
}