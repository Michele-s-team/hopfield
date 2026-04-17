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
//const int J = 1;
//const double k_B = 1.38 * pow(10, -23);  // has to be changed!!!!!!!!
double Beta_times_J = 0.1;
const int N_steps = 100;

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

void init_connections_set(vector<vector<int>>& connections,
                          vector<int>& neighbor_counts,
                          gsl_rng* ran) {
    fill(neighbor_counts.begin(), neighbor_counts.end(), 0);
    for (int i = 0; i < N_neurons; i++) {
        for (int j = 0; j < N_neurons; j++) {
            if (i != j) {
                int connection = gsl_rng_uniform_int(ran, 2);
                connections[i][j] = connection;
                neighbor_counts[i] += connection;
            } else {
                connections[i][j] = 0;  // no self-connections
            }
        }
    }
}

// Returns sigma_i * sum_{j neighbor of i} sigma_j.
// Positive means flipping neuron i would cost energy (unfavorable).
double DeltaE(int neuron, const vector<int>& connection, const vector<int>& neurons) {
    int sum = 0;
    for (size_t j = 0; j < neurons.size(); j++)
        if (connection[j] == 1) sum += neurons[j];
    return neurons[neuron] * sum;
}

// Classic implementation of the Hopfield network evolution.
// At each step, the same random threshold rho is applied to all n_bits realizations.
// Neuron i flips in realization r if rho >= DeltaE(i, r).
void evolve_systems(vector<vector<int>>& neurons_set,
                    const vector<vector<int>>& connections,
                    const vector<int>& neighbor_count,
                    const vector<vector<int>>& random_numbers,
                    const int N_steps) {
    double dE;
    for (int step = 0; step < N_steps; step++) {
        cout << "step "<< step <<endl;
        for (int i = 0; i < N_neurons; i++) {
            double rho = random_numbers[step][i];
            for (size_t r = 0; r < n_bits; r++) {       
                dE = DeltaE(i, connections[i], neurons_set[r]);  
                if (rho >= dE) neurons_set[r][i] = -neurons_set[r][i];
            }
        }
    }
}

// Bitwise implementation of the Hopfield network evolution.
// Evolves N_neurons neurons over N_steps time steps, processing all n_bits realizations
// simultaneously using bitwise arithmetic.
//
// Two branches avoid unnecessary computation:
//   BRANCH 1: rho >= neighbor_count[i]  =>  flip neuron i in all realizations unconditionally
//   BRANCH 2: general case, compute the neighbor sum and apply the mask
void evolve_systems_bits(vector<vector<int>>& neurons_set,
                         const vector<vector<int>>& connections,
                         const vector<int>& neighbor_count,
                         const vector<vector<int>>& random_numbers,
                         const int N_steps) {

    // Build the bitwise neuron representation from the classic neurons_set.
    // Neurons_Set[i] is a Bits encoding neuron i across all n_bits realizations.
    // Bit r of Neurons_Set[i] = 1 if neuron i is +1 in realization r, 0 if -1.
    vector<Bits> Neurons_Set;
    Neurons_Set.reserve(N_neurons);
    vector<UnsignedInt> Neighbor_Count;
    Neighbor_Count.reserve(N_neurons);

    for (int i = 0; i < N_neurons; i++) {
        Bits Neuron_tmp(1);
        for (int r = 0; r < n_bits; r++) {
            int bit = (neurons_set[r][i] + 1) / 2;
            Neuron_tmp.Set(r, bit);
        }
        Neurons_Set.push_back(Neuron_tmp);

        UnsignedInt Neighbor_tmp(neighbor_count[i]);
        Neighbor_tmp.SetAll(neighbor_count[i]);
        Neighbor_Count.push_back(Neighbor_tmp);
    }

    vector<vector<UnsignedInt>> Random_Numbers;
    Random_Numbers.resize(N_steps, vector<UnsignedInt>(N_neurons, UnsignedInt(N_neurons)));
    Bits xnor_ij;  // xnor_ij: XNOR of neuron_i bits and neuron_j bits.
    Bits mask;   // mask[r]=1 if sum[r] < Random_Numbers[step]: neuron i flips in realization r

    for (int step = 0; step < N_steps; step++) {
        for (int i = 0; i < N_neurons; i++) {
            unsigned long long int v=(unsigned long long int) random_numbers[step][i];
            Random_Numbers[step][i].SetAll(v);
        }
    }
    for (int step = 0; step < N_steps; step++) {
        cout << "step "<< step <<endl;
        for (int i = 0; i < N_neurons; i++) {

            // BRANCH 1 IS NOT STRICTLY NECESSARY, COMMENTED FOR DEBUG
            // BRANCH 1: rho >= neighbor_count[i], the maximum possible sum for neuron i.
            // The flip is guaranteed for ALL realizations.
            //if (random_numbers[i][step] >= neighbor_count[i]) {Neurons_Set[i].ComplementTo();} 
           // else {
                // BRANCH 2: compute the bitwise neighbor sum, then compare to threshold.
                UnsignedInt sum(1);
                sum.SetAll(0);
                
                // Bit r is 1 only if both neuron i and neuron j are +1 in realization r.
                for (int j = 0; j < N_neurons; j++) {                    
                    if (i != j && connections[i][j]) {
                        xnor_ij = Neurons_Set[i] == Neurons_Set[j];  //c_ij= b_i == b_j
                        sum += &xnor_ij;
                    }
                }
                
                BitSet temp = Random_Numbers[step][i] + &Neighbor_Count[i];
                
                Random_Numbers[step][i].Print("Random_Numbers[step][i]");
                Neighbor_Count[i].Print("Neighbor_Count[i]");
                temp.Print("Random_Numbers[step][i] + &Neighbor_Count[i]");

                sum.Print("sum");
                sum.MultiplyByTwoTo();   
                sum.Print("2*sum");
                mask = sum <= temp;
                Neurons_Set[i].Print("Neurons_Set[i] before flip");
                mask.Print("mask");
                Neurons_Set[i] ^= &mask;
                Neurons_Set[i].Print("Neurons_Set[i] after flip");
                cout<<"\n\n\n\n\n";
           // }
        }      
    }
    // Write the bitwise result back to neurons_set for comparison with the classic result.
    // Convert bit {0,1} back to spin {-1,+1}: spin = -1 + 2*bit
    for (int r = 0; r < n_bits; r++)
        for (int i = 0; i < N_neurons; i++)
            neurons_set[r][i] = -1 + 2 * Neurons_Set[i].Get(r);
}


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

        // Comparison between classic and bits
        for (int i = 0; i < N_neurons; i++) {
            if (neurons_set_classic[r][i] != neurons_set_bits[r][i]) {
                all_equal = false;
                cout << "Mismatch at r=" << r+1
                     << " i=" << i+1 << endl;
            }
        }
    }

    if (all_equal) {
        cout << "OK: classic and bitwise results are identical\n";
    } else {
        cout << "WARNING: differences detected between classic and bitwise results\n";
    }
}

int main() {
    InitGlobals();  // initializes Bits_one and Bits_zero


    cout << "[main] Parameters: N_neurons=" << N_neurons
         << " n_bits=" << n_bits
         << " N_steps=" << N_steps
         << " Beta*J=" << Beta_times_J<< endl;


    clock_t start_bits, end_bits;
    clock_t start_ref, end_ref;
    double clock_bitset, clock_ref;

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);

    // Initialize neuron states: neurons_set[r][i] is the spin of neuron i in realization r
    vector<vector<int>> neurons_set(n_bits, vector<int>(N_neurons));
    init_neurons_set(neurons_set, ran);

    // Initialize connectivity
    vector<vector<int>> connections(N_neurons, vector<int>(N_neurons, 0));
    vector<int> neighbor_count(N_neurons, 0);
    init_connections_set(connections, neighbor_count, ran);

    // Generate random thresholds, capped at N_neurons (the maximum possible neighbor sum).
    // NOTE: the exponential is artificially clamped to >= 1.0 — has to be checked.
    vector<vector<int>> random_numbers(N_steps, vector<int>(N_neurons, 0));
    for (int step = 0; step < N_steps; step++){
        for (int i = 0; i < N_neurons; i++){  
        random_numbers[step][i] = (int)min((double)N_neurons, 1.0 / (2.0 * Beta_times_J)
                                       * gsl_ran_exponential(ran, 1.0));
        cout <<"random number "<< random_numbers[step][i];
        }
    }

    vector<vector<int>> neurons_set_before = neurons_set;

    // Bitwise evolution: takes the same inputs, writes result back into neurons_set_bits
    vector<vector<int>> neurons_set_bits = neurons_set;
    cout<<"start Bitwise evolution"<<endl;
    start_bits = clock();
    evolve_systems_bits(neurons_set_bits, connections, neighbor_count, random_numbers, N_steps);
    end_bits = clock();
    clock_bitset = double(end_bits - start_bits) / CLOCKS_PER_SEC;
    cout<<"Bitwise evolution done"<<endl;

    // Classic evolution: modifies neurons_set in place
    vector<vector<int>> neurons_set_classic = neurons_set;
    cout<<"start classical evolution"<<endl;
    start_ref = clock();
    evolve_systems(neurons_set_classic, connections, neighbor_count, random_numbers, N_steps);
    end_ref = clock();
    clock_ref = double(end_ref - start_ref) / CLOCKS_PER_SEC;
    cout<<"Classical evolution done"<<endl;

    print_neurons(neurons_set_before, neurons_set_classic, neurons_set_bits, N_neurons, prefix_width, col_width);

    cout << "Total clock_ref:    " << clock_ref << " s\n";
    cout << "Total clock_bitset: " << clock_bitset << " s\n";
    cout << "Acceleration factor = " << clock_ref/clock_bitset << "\n";

    gsl_rng_free(ran);
    return 0;
}