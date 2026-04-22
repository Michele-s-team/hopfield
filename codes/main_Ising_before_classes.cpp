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

//THE CODE IMPLEMENTS THE THERMALIZATION OF A ISING NETWORK USING CLASSIC OR BITIWSE IMPLEMENTATION
// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL PARAMETERS
// ─────────────────────────────────────────────────────────────────────────────

const int L            = 5;     // number of spins on a side of the lattice    
const int N_neurons    = L * L; // total number of spins
double betaJ = 0.8;             // \beta*J
const int N_sweeps = 20;         // number of sweeps
const int col_width    = 3;     // printing parameter
const int prefix_width = 12;    // printing parameter


// ─────────────────────────────────────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────────────────────────────────────

// Initialize neuron states: neurons_set[r][i] is the spin of neuron i in realization r
void init_neurons_set(vector<vector<int>>& neurons_set, gsl_rng* ran) {
    for (size_t r = 0; r < neurons_set.size(); r++)
        for (int i = 0; i < N_neurons; i++)
            neurons_set[r][i] = 2 * gsl_rng_uniform_int(ran, 2) - 1;
}

// Initialize a random connectivity matrix with no self-connections
void init_connections_random(vector<vector<int>>& connections,
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

// Initialize a 2D square lattice with periodic boundary conditions
void init_connections_2D(vector<vector<int>>& connections,
                         vector<int>& neighbor_counts,
                         int L) {
    int N = L * L;
    // Reset
    fill(neighbor_counts.begin(), neighbor_counts.end(), 0);
    for (int i = 0; i < N; i++)
        fill(connections[i].begin(), connections[i].end(), 0);
    // Loop over lattice
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            int i = x + L * y;
            // Periodic boundary conditions
            int x_right = (x + 1) % L;
            int x_left  = (x - 1 + L) % L;
            int y_up    = (y + 1) % L;
            int y_down  = (y - 1 + L) % L;
            // Neighbors
            int neighbors[4] = {
                x_right + L * y,
                x_left  + L * y,
                x + L * y_up,
                x + L * y_down
            };
            for (int k = 0; k < 4; k++) {
                int j = neighbors[k];
                connections[i][j] = 1;
                neighbor_counts[i]+=1;
            }
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// ENERGY
// ─────────────────────────────────────────────────────────────────────────────

// Returns sigma_i * sum_{j neighbor of i} sigma_j.
double DeltaE(int neuron, const vector<int>& connection, const vector<int>& neurons) {
    int sum = 0;
    for (size_t j = 0; j < neurons.size(); j++)
        if (connection[j] == 1) sum += neurons[j];
    return neurons[neuron] * sum;
}


// ─────────────────────────────────────────────────────────────────────────────
// CLASSIC IMPLEMENTATION
// connections: Matrix N_neurons x N_neurons, 1 if there is a connection, 0 otherwise.
// neighbor_count: number of neighbors of each neuron (identical across all realizations).
// neurons_set: Matrix n_bits x N_neurons, initial spin value (+1/-1) per realization per neuron.
// ─────────────────────────────────────────────────────────────────────────────

// Classic implementation: one sweep over all neurons.
// At each sweep, the same random threshold rho is applied to all n_bits realizations.
// Neuron i flips in realization r if rho >= DeltaE(i, r).
void evolve_classic_one_sweep(vector<vector<int>>& neurons_set,
                              const vector<vector<int>>& connections,
                              const vector<int>& neighbor_count,
                              const vector<vector<int>>& random_numbers,
                              const int sweep) {
    double dE;
    for (int i = 0; i < N_neurons; i++) {
        double rho = random_numbers[sweep][i];
        for (size_t r = 0; r < n_bits; r++) {
            dE = DeltaE(i, connections[i], neurons_set[r]);
            if (rho >= dE) neurons_set[r][i] = -neurons_set[r][i];
        }
    }
}

// Classic implementation: N_sweeps sweeps.
void evolve_systems(vector<vector<int>>& neurons_set,
                    const vector<vector<int>>& connections,
                    const vector<int>& neighbor_count,
                    const vector<vector<int>>& random_numbers,
                    const int N_sweeps) {
    for (int sweep = 0; sweep < N_sweeps; sweep++) {
        if ((sweep+1) % (N_sweeps / 10) == 0)
            cout << "\rsweep: " << sweep+1 << " (" << ((sweep+1) * 100 / N_sweeps) << "%)    " << flush;
        evolve_classic_one_sweep(neurons_set, connections, neighbor_count, random_numbers, sweep);
    }
    cout << "\n";
}

// Classic implementation: N_sweeps sweeps (monolithic version, kept for reference).
void evolve_systems_old(vector<vector<int>>& neurons_set,
                        const vector<vector<int>>& connections,
                        const vector<int>& neighbor_count,
                        const vector<vector<int>>& random_numbers,
                        const int N_sweeps) {
    double dE;
    for (int sweep = 0; sweep < N_sweeps; sweep++) {
        if (sweep % (N_sweeps / 10) == 0)
            cout << "\rsweep: " << sweep+1 << " (" << ((sweep+1) * 100 / N_sweeps) << "%)    " << flush;
        for (int i = 0; i < N_neurons; i++) {
            double rho = random_numbers[sweep][i];
            for (size_t r = 0; r < n_bits; r++) {
                dE = DeltaE(i, connections[i], neurons_set[r]);
                if (rho >= dE) neurons_set[r][i] = -neurons_set[r][i];
            }
        }
    }
    cout << "\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// BITWISE IMPLEMENTATION
// Neurons_Set: vector of N_neurons Bits. Neurons_Set[i] encodes neuron i across all n_bits realizations.
// Neighbor_Count: vector of N_neurons UnsignedInts, same value as neighbor_count.
// ─────────────────────────────────────────────────────────────────────────────

// Build Neurons_Set, Neighbor_Count, and Random_Numbers from their classic counterparts.
void initEvolveContext(
    const vector<vector<int>>& neurons_set_bits,
    const vector<int>& neighbor_count,
    const vector<vector<int>>& random_numbers,
    const int N_sweeps,
    vector<Bits>& Neurons_Set_bits,
    vector<UnsignedInt>& Neighbor_Count,
    vector<vector<UnsignedInt>>& Random_Numbers)
{
    // Build Neurons_Set and Neighbor_Count
    Neurons_Set_bits.reserve(N_neurons);
    Neighbor_Count.reserve(N_neurons);
    for (int i = 0; i < N_neurons; i++) {
        Bits Neuron_tmp(1);
        for (int r = 0; r < n_bits; r++) {
            int bit = (neurons_set_bits[r][i] + 1) / 2;
            Neuron_tmp.Set(r, bit);
        }
        Neurons_Set_bits.push_back(Neuron_tmp);
        UnsignedInt Neighbor_tmp(neighbor_count[i]);
        Neighbor_tmp.SetAll(neighbor_count[i]);
        Neighbor_Count.push_back(Neighbor_tmp);
    }
    // Pre-compute Random_Numbers as UnsignedInt
    Random_Numbers.resize(N_sweeps, vector<UnsignedInt>(N_neurons, UnsignedInt(N_neurons)));
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N_neurons; i++) {
            unsigned long long int v = (unsigned long long int) random_numbers[sweep][i];
            Random_Numbers[sweep][i].SetAll(v);
        }
}

// Bitwise implementation: one sweep over all neurons.
// xnor_ij and mask are passed by reference to avoid reallocation at each call.
void evolve_bits_one_sweep(
    vector<Bits>& Neurons_Set,
    const vector<vector<int>>& connections,
    vector<UnsignedInt>& Neighbor_Count,
    const vector<int>& neighbor_count,
    vector<UnsignedInt>& Random_Numbers_sweep,
    const vector<int>& random_numbers_sweep,
    BitSet& threshold,
    Bits& xnor_ij,
    Bits& mask)
{
    for (int i = 0; i < N_neurons; i++) {
        // BRANCH 1: rho >= neighbor_count[i], the maximum possible sum for neuron i.
        // The flip is guaranteed for ALL realizations.
        if (random_numbers_sweep[i] >= neighbor_count[i]) {
            Neurons_Set[i].ComplementTo();
        }
        else {
            // BRANCH 2: compute the bitwise neighbor sum, then compare to threshold.
            UnsignedInt sum(1);
            sum.SetAll(0);
            for (int j = 0; j < N_neurons; j++) {
                if (i != j && connections[i][j]) {
                    xnor_ij = Neurons_Set[i] == Neurons_Set[j];  // c_ij = b_i == b_j
                    sum += &xnor_ij;                              // Sum S_iS_j = 2*Sum c_ij - degree(i)
                }
            }
            sum.MultiplyByTwoTo();
            threshold = Random_Numbers_sweep[i] + &Neighbor_Count[i];
            mask = sum <= threshold;
            Neurons_Set[i] ^= &mask;
        }
    }
}

// Bitwise implementation: N_sweeps sweeps.
void evolve_systems_bits(
    vector<vector<int>>& neurons_set,
    const vector<vector<int>>& connections,
    const vector<int>& neighbor_count,
    const vector<vector<int>>& random_numbers,
    const int N_sweeps,
    int prefix_width, int col_width)
{
    vector<Bits> Neurons_Set_bits;
    vector<UnsignedInt> Neighbor_Count;
    vector<vector<UnsignedInt>> Random_Numbers;
    Bits xnor_ij;
    Bits mask;
    BitSet threshold; // threshold condition to flip spin i (different among realizations)
    initEvolveContext(neurons_set, neighbor_count, random_numbers, N_sweeps,
                      Neurons_Set_bits, Neighbor_Count, Random_Numbers);

    for (int sweep = 0; sweep < N_sweeps; sweep++) {
        if (sweep % (N_sweeps / 10) == 0)
            cout << "\rsweep: " << sweep << " (" << (sweep * 100 / N_sweeps) << "%)    " << flush;
        evolve_bits_one_sweep(
            Neurons_Set_bits, connections, Neighbor_Count, neighbor_count,
            Random_Numbers[sweep], random_numbers[sweep], threshold, xnor_ij, mask);
    }
    cout << "\n";
}

// Bitwise implementation: N_sweeps sweeps (monolithic version, kept for reference).
void evolve_systems_bits_old(
    vector<vector<int>>& neurons_set,
    const vector<vector<int>>& connections,
    const vector<int>& neighbor_count,
    const vector<vector<int>>& random_numbers,
    const int N_sweeps)
{
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
    Random_Numbers.resize(N_sweeps, vector<UnsignedInt>(N_neurons, UnsignedInt(N_neurons)));
    Bits xnor_ij;  // xnor_ij: XNOR of neuron_i bits and neuron_j bits.
    Bits mask;     // mask[r]=1 if sum[r] < Random_Numbers[sweep]: neuron i flips in realization r
    BitSet threshold; // threshold condition to flip spin i (different among realizations)
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N_neurons; i++) {
            unsigned long long int v = (unsigned long long int) random_numbers[sweep][i];
            Random_Numbers[sweep][i].SetAll(v);
        }
    for (int sweep = 0; sweep < N_sweeps; sweep++) {
        if (sweep % (N_sweeps / 10) == 0)
            cout << "\rsweep: " << sweep << " (" << (sweep * 100 / N_sweeps) << "%)    " << flush;
        for (int i = 0; i < N_neurons; i++) {
            // BRANCH 1: rho >= neighbor_count[i], the maximum possible sum for neuron i.
            // The flip is guaranteed for ALL realizations.
            if (random_numbers[sweep][i] >= neighbor_count[i]) {
                Neurons_Set[i].ComplementTo();
            }
            else {
                // BRANCH 2: compute the bitwise neighbor sum, then compare to threshold.
                UnsignedInt sum(1);
                sum.SetAll(0);
                // Bit r is 1 only if both neuron i and neuron j are +1 in realization r.
                for (int j = 0; j < N_neurons; j++) {
                    if (i != j && connections[i][j]) {
                        xnor_ij = Neurons_Set[i] == Neurons_Set[j];  // c_ij = b_i == b_j
                        sum += &xnor_ij;                              // Sum S_iS_j = 2*Sum c_ij - degree(i)
                    }
                }
                sum.MultiplyByTwoTo();
                threshold = Random_Numbers[sweep][i] + &Neighbor_Count[i];
                mask = sum <= threshold;
                Neurons_Set[i] ^= &mask;
            }
        }
    }
    // Convert bit {0,1} back to spin {-1,+1}: spin = -1 + 2*bit
    for (int r = 0; r < n_bits; r++)
        for (int i = 0; i < N_neurons; i++)
            neurons_set[r][i] = -1 + 2 * Neurons_Set[i].Get(r);
    cout << "\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// COMBINED EVOLUTION (classic + bitwise in locksweep for comparison)
// ─────────────────────────────────────────────────────────────────────────────


// --- PROTOTYPES ---
void compare_neurons(
    const vector<vector<int>>& neurons_set_classic,
    const vector<vector<int>>& neurons_set_bits,
    int N_neurons, int prefix_width, int col_width);
    

// Run one sweep of both classic and bitwise implementations, then compare results for all N_st.
void evolve(
    vector<vector<int>>& neurons_set_classic,
    vector<vector<int>>& neurons_set_bits,
    const vector<vector<int>>& connections,
    const vector<int>& neighbor_count,
    const vector<vector<int>>& random_numbers,
    const int N_sweeps)
{
    vector<Bits> Neurons_Set_bits;
    vector<UnsignedInt> Neighbor_Count;
    vector<vector<UnsignedInt>> Random_Numbers;
    Bits xnor_ij;
    Bits mask;
    BitSet threshold; // threshold condition to flip spin i (different among realizations)
    initEvolveContext(neurons_set_bits, neighbor_count, random_numbers, N_sweeps,
                      Neurons_Set_bits, Neighbor_Count, Random_Numbers);
    cout <<"INITIAL SETS:  \n";
    compare_neurons(neurons_set_classic, neurons_set_bits, N_neurons, prefix_width, col_width);
    for (int sweep = 0; sweep < N_sweeps; sweep++) {
       // if (sweep % (N_sweeps / 10) == 0)
       //    cout << "\rsweep: " << sweep << " (" << (sweep * 100 / N_sweeps) << "%)    " << flush;
        cout << "\rsweep: " << sweep << " (" << (sweep * 100 / N_sweeps) << "%)    \n";
        evolve_classic_one_sweep(neurons_set_classic, connections, neighbor_count, random_numbers, sweep);
        evolve_bits_one_sweep(Neurons_Set_bits, connections, Neighbor_Count, neighbor_count,
                              Random_Numbers[sweep], random_numbers[sweep], threshold, xnor_ij, mask);
        for (int r = 0; r < n_bits; r++)
            for (int i = 0; i < N_neurons; i++)
                neurons_set_bits[r][i] = -1 + 2 * Neurons_Set_bits[i].Get(r);
        compare_neurons(neurons_set_classic, neurons_set_bits, N_neurons, prefix_width, col_width);
    }
    cout << "\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// DISPLAY AND COMPARISON
// ─────────────────────────────────────────────────────────────────────────────

// Compare classic and bitwise neuron states, print both and report mismatches.
void compare_neurons(
    const vector<vector<int>>& neurons_set_classic,
    const vector<vector<int>>& neurons_set_bits,
    int N_neurons, int prefix_width, int col_width)
{
    bool all_equal = true;
    for (size_t r = 0; r < neurons_set_classic.size(); r++) {
        cout << left << setw(prefix_width) << "after classic: ";
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_classic[r][i] << " ";
        cout << "\n";
        cout << left << setw(prefix_width) << "   after bits: ";
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

// Print before/after states for all realizations, then delegate comparison to compare_neurons.
void print_neurons(
    const vector<vector<int>>& neurons_set_before,
    const vector<vector<int>>& neurons_set_classic,
    const vector<vector<int>>& neurons_set_bits,
    int N_neurons, int prefix_width, int col_width)
{
    for (size_t r = 0; r < neurons_set_classic.size(); r++) {
        ostringstream oss_before;
        oss_before << "Realization" << right << setw(3) << r+1 << "\n       before: ";
        cout << left << setw(prefix_width) << oss_before.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_before[r][i] << " ";
        cout << "\n";
    }
    // Delegate to compare_neurons for the rest
    compare_neurons(neurons_set_classic, neurons_set_bits, N_neurons, prefix_width, col_width);
}


// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    InitGlobals();  // initializes Bits_one and Bits_zero
    cout << "[main] Parameters: N_neurons=" << N_neurons
         << " n_bits=" << n_bits
         << " N_sweeps=" << N_sweeps
         << " Beta*J=" << betaJ << endl;

    clock_t start_bits, end_bits;
    clock_t start_ref, end_ref;
    double clock_bitset, clock_ref;

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 42);

    // Initialize neuron states: neurons_set[r][i] is the spin of neuron i in realization r
    vector<vector<int>> neurons_set(n_bits, vector<int>(N_neurons));
    init_neurons_set(neurons_set, ran);

    // Initialize connectivity
    vector<vector<int>> connections(N_neurons, vector<int>(N_neurons, 0));
    vector<int> neighbor_count(N_neurons, 0);
    init_connections_2D(connections, neighbor_count, L);

    // Generate random thresholds, capped at N_neurons (the maximum possible neighbor sum).
    // NOTE: the exponential is artificially clamped to >= 1.0 — has to be checked.
    vector<vector<int>> random_numbers(N_sweeps, vector<int>(N_neurons, 0));
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N_neurons; i++)
            random_numbers[sweep][i] = (int) min((double) N_neurons,
                1.0 / (2.0 * betaJ) * gsl_ran_exponential(ran, 1.0));

    vector<vector<int>> neurons_set_before = neurons_set;
    evolve(neurons_set, neurons_set, connections, neighbor_count, random_numbers, N_sweeps);

    /*
    // Bitwise evolution: takes the same inputs, writes result back into neurons_set_bits
    vector<vector<int>> neurons_set_bits = neurons_set;
    cout<<"start Bitwise evolution"<<endl;
    start_bits = clock();
    evolve_systems_bits(neurons_set_bits, connections, neighbor_count, random_numbers, N_sweeps);
    end_bits = clock();
    clock_bitset = double(end_bits - start_bits) / CLOCKS_PER_SEC;
    cout<<"Bitwise evolution done"<<endl;
    cout << "Total clock_bitset: " << clock_bitset << " s\n";

    // Classic evolution: modifies neurons_set in place
    vector<vector<int>> neurons_set_classic = neurons_set;
    cout<<"start classical evolution"<<endl;
    start_ref = clock();
    evolve_systems(neurons_set_classic, connections, neighbor_count, random_numbers, N_sweeps);
    end_ref = clock();
    clock_ref = double(end_ref - start_ref) / CLOCKS_PER_SEC;
    cout<<"Classical evolution done"<<endl;
    cout << "Total clock_ref:    " << clock_ref << " s\n";

    print_neurons(neurons_set_before, neurons_set_classic, neurons_set_bits, N_neurons, prefix_width, col_width);
    cout << "Total clock_bitset: " << clock_bitset << " s\n";
    cout << "Total clock_ref:    " << clock_ref << " s\n";
    cout << "Acceleration factor = " << clock_ref/clock_bitset << "\n";
    */

    gsl_rng_free(ran);
    return 0;
}