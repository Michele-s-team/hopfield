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
#include "system_bits.hpp"
#include "system_nobits.hpp"

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

const int col_width = 3;
const int prefix_width = 12;

const int N_neurons = 100;
// CLASSIC IMPLEMENTATION
// connections: Matrix N_neurons x N_neurons, 1 if there is a connection, 0 otherwise, random.
// neighbor_count: number of neighbors of each neuron (identical across all realizations).
// neurons_set: Matrix n_bits x N_neurons, initial spin value (+1/-1) per realization per neuron.

// BITWISE IMPLEMENTATION
// Same connections as classic implementation.
// Neurons_Set: vector of N_neurons UnsignedInts. Neurons_Set[i] encodes the state of neuron iand neighbor_counts
// Neighbor_Count: vector of N_neurons UnsignedInts of same as neighbor_count
// 
// across all n_bits realizations simultaneously: bit r = 1 if neuron i is +1 in realization r.


void init_neurons_set(vector<vector<int>>& neurons_set, gsl_rng* ran) {
    for (size_t r = 0; r < neurons_set.size(); r++)
        for (int i = 0; i < N_neurons; i++){
            neurons_set[r][i] = 2 * gsl_rng_uniform_int(ran, 2) - 1;
        }
}

// Bitwise implementation of the Hopfield network evolution.
// Evolves N_neurons neurons over N_steps time steps, processing all n_bits realizations
// simultaneously using bitwise arithmetic.
//
// Two branches avoid unnecessary computation:
//   BRANCH 1: rho >= neighbor_count[i]  =>  flip neuron i in all realizations unconditionally
//   BRANCH 2: general case, compute the neighbor sum and apply the mask
void test_spin_multiplication(vector<vector<int>>& neurons_set, vector<int>& sum,  vector<int>& sum_bitwise){

    vector<Bits> Neurons_Set;
    Neurons_Set.reserve(N_neurons);

    // Build bit representation
    for (int i = 0; i < N_neurons; i++) {
        Bits Neuron_tmp(n_bits); // FIXED SIZE
        for (int r = 0; r < n_bits; r++) {
            int bit = (neurons_set[r][i] + 1) / 2;
            Neuron_tmp.Set(r, bit);
        }
        Neurons_Set.push_back(Neuron_tmp);
    }

    Bits xnor;
    UnsignedInt SUM(n_bits);
    SUM.SetAll(0);

    // Compute sums
    for (int j = 1; j < N_neurons; j++) {

        xnor = (Neurons_Set[0] == Neurons_Set[j]);
        SUM += &xnor;

        for (int r = 0; r < n_bits; r++){
            sum[r] += neurons_set[r][j];
        }
    }
    SUM.MultiplyByTwoTo();

    // Multiply once at the end
    for (int r = 0; r < n_bits; r++){
        sum[r] = neurons_set[r][0] * sum[r];
        sum_bitwise[r]=SUM.Get(r)-N_neurons+1;
        cout << "classic sum = " << sum[r]
        << "   bitwise sum = " << sum_bitwise[r] << endl;
    }
}


int main() {
    // Init RNG
    gsl_rng_env_setup();
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_default);

    InitGlobals();


    vector<vector<int>> neurons_set(n_bits, vector<int>(N_neurons));
    vector<int> sums_set(n_bits, 0);
    vector<int> sums_set_bitwise(n_bits, 0);

    init_neurons_set(neurons_set, ran);

    test_spin_multiplication(neurons_set, sums_set, sums_set_bitwise);

    gsl_rng_free(ran);
    return 0;
}