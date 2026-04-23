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
const int N_sweeps = pow(2,18);         // number of sweeps
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
void init_connections_2D(vector<vector<int>>& neighbors_list, vector<int>& neighbor_counts, int L_side) {
    neighbors_list.assign(N_neurons, vector<int>());
    fill(neighbor_counts.begin(), neighbor_counts.end(), 4);
    for (int y = 0; y < L_side; y++) {
        for (int x = 0; x < L_side; x++) {
            int i = x + L_side * y;
            int r = (x + 1) % L_side + L_side * y;
            int l = (x - 1 + L_side) % L_side + L_side * y;
            int u = x + L_side * ((y + 1) % L_side);
            int d = x + L_side * ((y - 1 + L_side) % L_side);
            neighbors_list[i] = {r, l, u, d};
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ENERGY
// ─────────────────────────────────────────────────────────────────────────────

// Returns sigma_i * sum_{j neighbor of i} sigma_j.
double DeltaE(int neuron, const vector<int>& neighbors_of_i, const vector<int>& neurons) {
    int sum = 0;
    for (int j : neighbors_of_i) sum += neurons[j];
    return (double)(neurons[neuron] * sum);
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
                              const vector<vector<int>>& neighbors_list,
                              const vector<vector<int>>& random_numbers,
                              const int sweep) {
    for (int i = 0; i < N_neurons; i++) {
        double rho = random_numbers[sweep][i];
        for (size_t r = 0; r < n_bits; r++) {
            double dE = DeltaE(i, neighbors_list[i], neurons_set[r]);
            if (rho >= dE) neurons_set[r][i] = -neurons_set[r][i];
        }
    }
}

// Classic implementation: N_sweeps sweeps.
void evolve_systems_modular(vector<vector<int>>& neurons_set,
                            const vector<vector<int>>& neighbors_list,
                            const vector<vector<int>>& random_numbers,
                            const int N_sweeps_count) {
    for (int sweep = 0; sweep < N_sweeps_count; sweep++) {
        evolve_classic_one_sweep(neurons_set, neighbors_list, random_numbers, sweep);
    }
}
// Classic implementation: N_sweeps sweeps (monolithic version, kept for reference).
void evolve_systems_monolithic(vector<vector<int>>& neurons_set,
                               const vector<vector<int>>& neighbors_list,
                               const vector<int>& neighbor_count,
                               const vector<vector<int>>& random_numbers,
                               const int N_sweeps_count) {
    for (int sweep = 0; sweep < N_sweeps_count; sweep++) {
        if (sweep % max(1, N_sweeps_count / 10) == 0)
            cout << "\rsweep classic mono: " << sweep+1 << " (" << ((sweep+1) * 100 / N_sweeps_count) << "%)    " << flush;
        for (int i = 0; i < N_neurons; i++) {
            double rho = random_numbers[sweep][i];
            for (size_t r = 0; r < n_bits; r++) {
                // Utilisation directe de la liste de voisins i
                int sum = 0;
                for (int j : neighbors_list[i]) sum += neurons_set[r][j];
                double dE = neurons_set[r][i] * sum;
                
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
    const int N_sweeps_count,
    vector<Bits>& Neurons_Set_bits,
    vector<UnsignedInt>& Neighbor_Count_obj,
    vector<vector<UnsignedInt>>& Random_Numbers_obj)
{
    Neurons_Set_bits.clear(); Neurons_Set_bits.reserve(N_neurons);
    Neighbor_Count_obj.clear(); Neighbor_Count_obj.reserve(N_neurons);
    for (int i = 0; i < N_neurons; i++) {
        Bits Neuron_tmp(1);
        for (int r = 0; r < n_bits; r++) Neuron_tmp.Set(r, (neurons_set_bits[r][i] + 1) / 2);
        Neurons_Set_bits.push_back(Neuron_tmp);
        UnsignedInt Neighbor_tmp(neighbor_count[i]);
        Neighbor_tmp.SetAll(neighbor_count[i]);
        Neighbor_Count_obj.push_back(Neighbor_tmp);
    }
    Random_Numbers_obj.assign(N_sweeps_count, vector<UnsignedInt>(N_neurons, UnsignedInt(N_neurons)));
    for (int sweep = 0; sweep < N_sweeps_count; sweep++)
        for (int i = 0; i < N_neurons; i++) Random_Numbers_obj[sweep][i].SetAll((unsigned long long int)random_numbers[sweep][i]);
}

// Bitwise implementation: one sweep over all neurons.
// xnor_ij and mask are passed by reference to avoid reallocation at each call.
void evolve_bits_one_sweep(
    vector<Bits>& Neurons_Set,
    const vector<vector<int>>& neighbors_list,
    vector<UnsignedInt>& Neighbor_Count_obj,
    const vector<int>& neighbor_count_raw,
    vector<UnsignedInt>& Random_Numbers_sweep,
    const vector<int>& random_numbers_sweep_raw,
    BitSet& threshold, Bits& xnor_ij, Bits& mask, UnsignedInt& sum_obj) 
{
    for (int i = 0; i < N_neurons; i++) {
        if (random_numbers_sweep_raw[i] >= neighbor_count_raw[i]) {
            Neurons_Set[i].ComplementTo();
        } else {
            sum_obj.SetAll(0);
            for (int j : neighbors_list[i]) {
                xnor_ij = (Neurons_Set[i] == Neurons_Set[j]);
                sum_obj += &xnor_ij;
            }
            sum_obj.MultiplyByTwoTo();
            threshold = Random_Numbers_sweep[i] + &Neighbor_Count_obj[i];
            mask = (sum_obj <= threshold);
            Neurons_Set[i] ^= &mask;
        }
    }
}

// Bitwise implementation: N_sweeps sweeps.
void evolve_systems_bits_modular(
    vector<vector<int>>& neurons_set,
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
    UnsignedInt sum_obj;//
    initEvolveContext(neurons_set, neighbor_count, random_numbers, N_sweeps,
                      Neurons_Set_bits, Neighbor_Count, Random_Numbers);

    for (int sweep = 0; sweep < N_sweeps; sweep++) {
        if (sweep % (N_sweeps / 10) == 0)
            cout << "\rsweep: " << sweep << " (" << (sweep * 100 / N_sweeps) << "%)    " << flush;
        evolve_bits_one_sweep(
            Neurons_Set_bits, connections, Neighbor_Count, neighbor_count,
            Random_Numbers[sweep], random_numbers[sweep], threshold, xnor_ij, mask, sum_obj);
    }
    // Convert bit {0,1} back to spin {-1,+1}: spin = -1 + 2*bit
    for (int r = 0; r < n_bits; r++)
        for (int i = 0; i < N_neurons; i++)
            neurons_set[r][i] = -1 + 2 * Neurons_Set_bits[i].Get(r);
    cout << "\n";
}

// Bitwise implementation: N_sweeps sweeps (monolithic version, kept for reference).
void evolve_systems_bits_monolithic(
    vector<vector<int>>& neurons_set,
    const vector<vector<int>>& neighbors_list,
    const vector<int>& neighbor_count,
    const vector<vector<int>>& random_numbers,
    const int N_sweeps_count)
{
    vector<Bits> Neurons_Set_bits;
    vector<UnsignedInt> Neighbor_Count_obj;
    vector<vector<UnsignedInt>> Random_Numbers_obj;
    Bits xnor_ij; Bits mask; BitSet threshold; UnsignedInt sum_obj(1); 

    initEvolveContext(neurons_set, neighbor_count, random_numbers, N_sweeps_count,
                      Neurons_Set_bits, Neighbor_Count_obj, Random_Numbers_obj);

    for (int sweep = 0; sweep < N_sweeps_count; sweep++) {
        if (sweep % max(1, N_sweeps_count / 10) == 0)
            cout << "\rsweep bits mono: " << sweep << " (" << (sweep * 100 / N_sweeps_count) << "%)    " << flush;
        for (int i = 0; i < N_neurons; i++) {
            if (random_numbers[sweep][i] >= neighbor_count[i]) {
                Neurons_Set_bits[i].ComplementTo();
            } else {
                sum_obj.SetAll(0);
                for (int j : neighbors_list[i]) {
                    xnor_ij = (Neurons_Set_bits[i] == Neurons_Set_bits[j]);
                    sum_obj += &xnor_ij;
                }
                sum_obj.MultiplyByTwoTo();
                threshold = Random_Numbers_obj[sweep][i] + &Neighbor_Count_obj[i];
                mask = (sum_obj <= threshold);
                Neurons_Set_bits[i] ^= &mask;
            }
        }
    }
    for (int r = 0; r < n_bits; r++)
        for (int i = 0; i < N_neurons; i++) neurons_set[r][i] = -1 + 2 * Neurons_Set_bits[i].Get(r);
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
// Run one sweep of both classic and bitwise implementations, then compare results.
// This ensures that the parallel bitwise logic is mathematically identical to the classic one.
void evolve(
    vector<vector<int>>& neurons_set_classic,
    vector<vector<int>>& neurons_set_bits,
    const vector<vector<int>>& neighbors_list, // Updated from connections
    const vector<int>& neighbor_count,
    const vector<vector<int>>& random_numbers,
    const int N_sweeps_count,
    int prefix_width_val, int col_width_val)
{
    vector<Bits> Neurons_Set_bits_obj;
    vector<UnsignedInt> Neighbor_Count_obj;
    vector<vector<UnsignedInt>> Random_Numbers_obj;
    
    Bits xnor_ij;
    Bits mask;
    BitSet threshold; 
    UnsignedInt sum_obj(1); // Reused object to avoid overhead

    // Initialize the bitwise context from the current state
    initEvolveContext(neurons_set_bits, neighbor_count, random_numbers, N_sweeps_count,
                      Neurons_Set_bits_obj, Neighbor_Count_obj, Random_Numbers_obj);

    cout << "INITIAL SETS:  \n";
    compare_neurons(neurons_set_classic, neurons_set_bits, N_neurons, prefix_width_val, col_width_val);

    for (int sweep = 0; sweep < N_sweeps_count; sweep++) {
        cout << "\rsweep comparison: " << sweep << " (" << (sweep * 100 / N_sweeps_count) << "%)    " << endl;

        // 1. Perform one classic sweep (O(1) neighbor access)
        evolve_classic_one_sweep(neurons_set_classic, neighbors_list, random_numbers, sweep);

        // 2. Perform one bitwise sweep (Parallel processing of n_bits)
        evolve_bits_one_sweep(Neurons_Set_bits_obj, neighbors_list, Neighbor_Count_obj, neighbor_count,
                              Random_Numbers_obj[sweep], random_numbers[sweep], 
                              threshold, xnor_ij, mask, sum_obj);

        // 3. Convert bitwise representation back to classic spins for comparison
        for (int r = 0; r < n_bits; r++)
            for (int i = 0; i < N_neurons; i++)
                neurons_set_bits[r][i] = -1 + 2 * Neurons_Set_bits_obj[i].Get(r);

        // 4. Check if both methods produced the exact same configuration
        compare_neurons(neurons_set_classic, neurons_set_bits, N_neurons, prefix_width_val, col_width_val);
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

    double clock_bitset, clock_ref;

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 42);

    // Initialize neuron states: neurons_set[r][i] is the spin of neuron i in realization r
    vector<vector<int>> neurons_set_init(n_bits, vector<int>(N_neurons));
    init_neurons_set(neurons_set_init, ran);

    // Initialize connectivity using adjacency list (neighbors_list) instead of matrix
    // Updated to use the more efficient neighbor-based structure
    vector<vector<int>> neighbors_list(N_neurons);
    vector<int> neighbor_count(N_neurons, 0);
    init_connections_2D(neighbors_list, neighbor_count, L);

    // Generate random thresholds, capped at N_neurons (the maximum possible neighbor sum).
    // NOTE: the exponential is artificially clamped to >= 1.0 — has to be checked.
    vector<vector<int>> random_numbers(N_sweeps, vector<int>(N_neurons, 0));
    for (int sweep = 0; sweep < N_sweeps; sweep++) {
        for (int i = 0; i < N_neurons; i++) {
            random_numbers[sweep][i] = (int) min((double) N_neurons,
                1.0 / (2.0 * betaJ) * gsl_ran_exponential(ran, 1.0));
        }
    }

    vector<vector<int>> neurons_set_before = neurons_set_init;

    // Optional: Comparative evolution (locked sweep by sweep)
    // Useful for debugging to ensure bitwise results strictly match classic results

    // evolve(neurons_set_init, neurons_set_init, neighbors_list, neighbor_count, random_numbers, N_sweeps, prefix_width, col_width);
    
    // --- BITWISE EVOLUTION ───────────────────────────────────────────────────
    // Bitwise implementation: processes n_bits realizations in parallel
    vector<vector<int>> neurons_set_bits = neurons_set_init;
    cout << "start Bitwise evolution" << endl;
    
    auto start_bits = chrono::high_resolution_clock::now();
    
    // Monolithic version is used here for maximum performance
    evolve_systems_bits_monolithic(neurons_set_bits, neighbors_list, neighbor_count, random_numbers, N_sweeps);
    
    auto end_bits = chrono::high_resolution_clock::now();
    clock_bitset = chrono::duration<double>(end_bits - start_bits).count();
    
    cout << "Bitwise evolution done" << endl;
    cout << "Total clock_bitset: " << clock_bitset << " s\n";

    // --- CLASSIC EVOLUTION ───────────────────────────────────────────────────
    // Classic implementation: standard loop over realizations and neurons
    vector<vector<int>> neurons_set_classic = neurons_set_init;
    cout << "start classical evolution" << endl;
    
    auto start_ref = chrono::high_resolution_clock::now();
    
    // Optimized to O(1) neighbor access instead of O(N) matrix scan
    evolve_systems_monolithic(neurons_set_classic, neighbors_list, neighbor_count, random_numbers, N_sweeps);
    
    auto end_ref = chrono::high_resolution_clock::now();
    clock_ref = chrono::duration<double>(end_ref - start_ref).count();
    
    cout << "Classical evolution done" << endl;
    cout << "Total clock_ref:    " << clock_ref << " s\n";

    // --- RESULTS AND COMPARISON ──────────────────────────────────────────────
    // Final check and printing of the states
    print_neurons(neurons_set_before, neurons_set_classic, neurons_set_bits, N_neurons, prefix_width, col_width);
    
    cout << "Total clock_bitset: " << clock_bitset << " s\n";
    cout << "Total clock_ref:    " << clock_ref << " s\n";
    
    if (clock_bitset > 0) {
        cout << "Acceleration factor = " << clock_ref / clock_bitset << "\n";
    }
    
    gsl_rng_free(ran);
    return 0;
}