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
void IsingBits::fromCanonical(){
    Neurons_Set.clear();
    Neighbor_Count.clear();

    Neurons_Set.reserve(N_neurons);
    Neighbor_Count.reserve(N_neurons);

    for (int i = 0; i < N_neurons; ++i) {

        Bits neuron_tmp;

        for (int r = 0; r < n_bits; ++r) {
            int bit = (neurons_set[r*N_neurons+i] + 1) / 2;
            neuron_tmp.Set(r, bit);
        }
        UnsignedInt neighbor_tmp(neighbor_count[i]);
        neighbor_tmp.SetAll(neighbor_count[i]);

        Neurons_Set.push_back(neuron_tmp);
        Neighbor_Count.push_back(neighbor_tmp);
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void IsingBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N_neurons; ++i)
            neurons_set[r*N_neurons+i] = -1 + 2 * Neurons_Set[i].Get(r);
}


// =====================================================
// RANDOM THRESHOLDS
// =====================================================

/*

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

    for (int sweep = 0; sweep < N_sweeps; ++sweep)
        for (int i = 0; i < N_neurons; ++i) {
            unsigned long long v =
                (unsigned long long) random_numbers[sweep][i];
            Random_Numbers[sweep][i].SetAll(v);
        }
}
*/

/*
void IsingBits::initRandomNumbers(gsl_rng* ran){
    IsingModel::initRandomNumbers(ran);
    convertRandomNumbers();
}

void IsingBits::initRandomNumbersFromExp(const vector<vector<double>>& exp_base){
    IsingModel::initRandomNumbersFromExp(exp_base);
    convertRandomNumbers();
}
*/

// =====================================================
// EVOLUTION CONTEXT
// =====================================================

/*
void IsingBits::initEvolveContext(){
    fromCanonical();
    convertRandomNumbers();
}
*/

// =====================================================
// CORE UPDATE KERNEL
// =====================================================
/*
//TO BE FIXED
void IsingBits::evolveOneSweep(int sweep, BitSet& threshold, Bits& xnor_ij, Bits& mask, gsl_rng* ran){
    for (int i = 0; i < N_neurons; ++i) {
        // --------------------------------
        // Branch 1: unconditional flip
        // --------------------------------
        int rng = randomNumber(ran);
        if (rng >= neighbor_count[i]) {
            Neurons_Set[i].ComplementTo();
        }
        // --------------------------------
        // Branch 2: conditional flip
        // --------------------------------
        else {
            UnsignedInt sum(neighbor_count[i]);
            sum.SetAll(0);
            for (int j : neighbors[i]){
                xnor_ij = (Neurons_Set[i] == Neurons_Set[j]);
                sum += &xnor_ij;
            }
        
            sum.MultiplyByTwoTo();
            //threshold = Random_Numbers[sweep][i] + &Neighbor_Count[i];
            mask = (sum <= threshold);
            Neurons_Set[i] ^= &mask;
        }
    }
}
*/
// =====================================================
// EVOLUTION DRIVERS
// =====================================================
/*
// Refactored version using reusable sweep kernel TO BE FIXED
void IsingBits::evolve_modular(gsl_rng* ran){
    fromCanonical();
    Bits xnor_ij;
    Bits mask;
    BitSet threshold;
    UnsignedInt sum(neighbor_count[0]);

    int progress_stride = max(1, N_sweeps / 10);

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {

        evolveOneSweep(sweep, threshold, xnor_ij, mask, ran);
        if ((sweep + 1) % progress_stride == 0) {
            cout << "\rSweep: " << sweep + 1 << " (" << ((sweep + 1) * 100 / N_sweeps)<< "%)    " << flush;
        }
    }
    cout << "\n";
    toCanonical();
}
*/

// Monolithic reference implementation
void IsingBits::evolve_monolithic(gsl_rng* ran) {
    fromCanonical();
    Bits xnor_ij, mask;
    BitSet threshold(N_neurons);

    UnsignedInt sum((unsigned long long int) (neighbor_count[0]*2));
    int progress_stride = max(1, N_sweeps / 10);

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int i = 0; i < N_neurons; ++i) {
            const int rng = randomNumber(ran);
            const int nc    = neighbor_count[i];
            Bits& neuron_i  = Neurons_Set[i];

            if (rng >= nc) {
                neuron_i.ComplementTo();
            }
            else {
                //sum.Resize((unsigned long long int) (nc*2));  //not truly needed as the size of BitSet dynamically increases
                //cout << "Initial size" << sum.GetSize() <<endl;
                
                sum.SetAll(0);

                for (int j : neighbors[i]) {
                    xnor_ij = (neuron_i == Neurons_Set[j]);
                    sum += &xnor_ij;
                }

                sum.MultiplyByTwoTo();
                threshold.SetAll((unsigned long long int) rng);
                threshold += &Neighbor_Count[i];
                mask = (sum <= threshold);
                neuron_i ^= &mask;
            }
            //cout << sum.GetSize() <<endl;

        }
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep+1
                 << " (" << ((sweep+1)*100/N_sweeps) << "%)    " << flush;
    }
    cout << "\n";
    toCanonical();
}