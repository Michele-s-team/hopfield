//
//  ising_bits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_bits.hpp"
#include "unsigned_int.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"

// =====================================================
// STATE CONVERSIONS
// =====================================================

// canonical spins {-1,+1} -> bit representation {0,1}
void IsingBits::fromCanonical()
{
    Neurons_Set.clear();
    Neighbor_Count.clear();

    Neurons_Set.reserve(N_neurons);
    Neighbor_Count.reserve(N_neurons);

    for (int i = 0; i < N_neurons; ++i) {

        Bits neuron_tmp(1);

        for (int r = 0; r < n_bits; ++r) {
            int bit = (neurons_set[r][i] + 1) / 2;
            neuron_tmp.Set(r, bit);
        }
        Neurons_Set.push_back(neuron_tmp);
        UnsignedInt neighbor_tmp(neighbor_count[i]);
        neighbor_tmp.SetAll(neighbor_count[i]);
        Neighbor_Count.push_back(neighbor_tmp);
    }
}


// bit representation {0,1} -> canonical spins {-1,+1}
void IsingBits::toCanonical()
{
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N_neurons; ++i)
            neurons_set[r][i] =
                -1 + 2 * Neurons_Set[i].Get(r);
}


// =====================================================
// RANDOM THRESHOLDS
// =====================================================

void IsingBits::convertRandomNumbers()
{
    Random_Numbers.clear();
    Random_Numbers.resize(
        N_sweeps,
        vector<UnsignedInt>(
            N_neurons,
            UnsignedInt(N_neurons)
        )
    );

    for (int step = 0; step < N_sweeps; ++step)
        for (int i = 0; i < N_neurons; ++i) {
            unsigned long long v =
                (unsigned long long) random_numbers[step][i];
            Random_Numbers[step][i].SetAll(v);
        }
}


void IsingBits::initRandomNumbers(gsl_rng* ran){
    IsingModel::initRandomNumbers(ran);
    convertRandomNumbers();
}


void IsingBits::initRandomNumbersFromExp(const vector<vector<double>>& exp_base){
    IsingModel::initRandomNumbersFromExp(exp_base);
    convertRandomNumbers();
}


// =====================================================
// EVOLUTION CONTEXT
// =====================================================

void IsingBits::initEvolveContext(){
    fromCanonical();
    convertRandomNumbers();
}



// =====================================================
// CORE UPDATE KERNEL
// =====================================================

void IsingBits::evolveOneSweep(int step, BitSet& threshold, Bits& xnor_ij, Bits& mask){
    for (int i = 0; i < N_neurons; ++i) {
        // --------------------------------
        // Branch 1: unconditional flip
        // --------------------------------
        if (random_numbers[step][i] >= neighbor_count[i]) {
            Neurons_Set[i].ComplementTo();
        }
        // --------------------------------
        // Branch 2: conditional flip
        // --------------------------------
        else {
            UnsignedInt sum(1);
            sum.SetAll(0);
            for (int j = 0; j < N_neurons; ++j) {
                if (i != j && connections[i][j]) {
                    xnor_ij = (Neurons_Set[i] == Neurons_Set[j]);
                    sum += &xnor_ij;
                }
            }
            sum.MultiplyByTwoTo();
            BitSet temp = Random_Numbers[step][i] + &Neighbor_Count[i];
            mask = (sum <= temp);
            Neurons_Set[i] ^= &mask;
        }
    }
}


// =====================================================
// EVOLUTION DRIVERS
// =====================================================

// Refactored version using reusable sweep kernel
void IsingBits::evolve_modular(){
    initEvolveContext();
    Bits xnor_ij;
    Bits mask;
    BitSet threshold;

    int progress_stride = max(1, N_sweeps / 10);

    for (int step = 0; step < N_sweeps; ++step) {

        evolveOneSweep(step, threshold, xnor_ij, mask);
        if ((step + 1) % progress_stride == 0) {
            cout << "\rStep: " << step + 1 << " (" << ((step + 1) * 100 / N_sweeps)<< "%)    " << flush;
        }
    }
    cout << "\n";
    toCanonical();
}


// Monolithic reference implementation
void IsingBits::evolve_monolithic(){
    fromCanonical();
    Bits xnor_ij;
    Bits mask;
    BitSet threshold;

    int progress_stride = max(1, N_sweeps / 10);

    for (int step = 0; step < N_sweeps; ++step) {
        for (int i = 0; i < N_neurons; ++i) {
            if (random_numbers[step][i] >= neighbor_count[i]) {
                Neurons_Set[i].ComplementTo();
            }
            else {
                UnsignedInt sum(1);
                sum.SetAll(0);
                for (int j = 0; j < N_neurons; ++j) {
                    if (i != j && connections[i][j]) {

                        xnor_ij = (Neurons_Set[i] == Neurons_Set[j]);
                        sum += &xnor_ij;
                    }
                }
                sum.MultiplyByTwoTo();
                threshold= Random_Numbers[step][i] + &Neighbor_Count[i];
                mask = (sum <= threshold);
                Neurons_Set[i] ^= &mask;
            }
        }
        if ((step + 1) % progress_stride == 0) {
            cout << "\rStep: " << step + 1 << " (" << ((step + 1) * 100 / N_sweeps)<< "%)    " << flush;
        }
    }
    cout << "\n";
    toCanonical();
}
