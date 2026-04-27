//
//  ising_bits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_bits.hpp"
#include "unsigned_int.hpp"
#include <numeric>

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"

// =====================================================
// STATE CONVERSIONS
// =====================================================

// canonical spins {-1,+1} -> bit representation {0,1}
void IsingBits::fromCanonical(){
    spins_Set.clear();
    Neighbor_Count.clear();

    spins_Set.reserve(N_spins);
    Neighbor_Count.reserve(N_spins);

    for (int i = 0; i < N_spins; ++i) {

        Bits spin_tmp;

        for (int r = 0; r < n_bits; ++r) {
            int bit = (spins_set[r*N_spins+i] + 1) / 2;
            spin_tmp.Set(r, bit);
        }
        UnsignedInt neighbor_tmp(neighbor_count[i]);
        neighbor_tmp.SetAll(neighbor_count[i]);

        spins_Set.push_back(spin_tmp);
        Neighbor_Count.push_back(neighbor_tmp);
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void IsingBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N_spins; ++i)
            spins_set[r*N_spins+i] = -1 + 2 * spins_Set[i].Get(r);
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
            N_spins,
            UnsignedInt(N_spins)
        )
    );

    for (int sweep = 0; sweep < N_sweeps; ++sweep)
        for (int i = 0; i < N_spins; ++i) {
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
    for (int i = 0; i < N_spins; ++i) {
        // --------------------------------
        // Branch 1: unconditional flip
        // --------------------------------
        int rng = randomNumber(ran);
        if (rng >= neighbor_count[i]) {
            spins_Set[i].ComplementTo();
        }
        // --------------------------------
        // Branch 2: conditional flip
        // --------------------------------
        else {
            UnsignedInt sum(neighbor_count[i]);
            sum.SetAll(0);
            for (int j : neighbors[i]){
                xnor_ij = (spins_Set[i] == spins_Set[j]);
                sum += &xnor_ij;
            }
        
            sum.MultiplyByTwoTo();
            //threshold = Random_Numbers[sweep][i] + &Neighbor_Count[i];
            mask = (sum <= threshold);
            spins_Set[i] ^= &mask;
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
/*
// Monolithic reference implementation
void IsingBits::evolve_save(gsl_rng* ran) {
    fromCanonical();
    Bits xnor_ij, mask;
    BitSet threshold(N_spins);

    UnsignedInt sum((unsigned long long int) (neighbor_count[0]*2));
    int progress_stride = max(1, N_sweeps / 10);

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int i = 0; i < N_spins; ++i) {
            const int rng = randomNumber(ran);
            const int nc    = neighbor_count[i];
            Bits& spin_i  = spins_Set[i];

            if (rng >= nc) {
                spin_i.ComplementTo();
            }
            else {
                //sum.Resize((unsigned long long int) (nc*2));  //not truly needed as the size of BitSet dynamically increases
                //cout << "Initial size" << sum.GetSize() <<endl;
                
                sum.SetAll(0);

                for (int j : neighbors[i]) {
                    xnor_ij = (spin_i == spins_Set[j]);
                    sum += &xnor_ij;
                }

                sum.MultiplyByTwoTo();
                threshold.SetAll((unsigned long long int) rng);
                threshold += &Neighbor_Count[i];
                mask = (sum <= threshold);
                spin_i ^= &mask;
            }
            //cout << sum.GetSize() <<endl;
            toCanonical();
            SaveSpins("../results/spins/spin_config");

        }
       // if ((sweep + 1) % progress_stride == 0)
         //   cout << "\rSweep: " << sweep+1
           //      << " (" << ((sweep+1)*100/N_sweeps) << "%)    " << flush;
    }
    //cout << "\n";
    toCanonical();
}
*/


// Monolithic reference implementation
void IsingBits::evolve(gsl_rng* ran) {
    fromCanonical();
    Bits xnor_ij, mask;
    BitSet threshold(N_spins);
    UnsignedInt sum((unsigned long long int) (neighbor_count[0]*2));
    int progress_stride = max(1, N_sweeps / 10);

    // Ordre aléatoire défini une seule fois
    vector<int> order(N_spins);
    iota(order.begin(), order.end(), 0);
    gsl_ran_shuffle(ran, order.data(), N_spins, sizeof(int));

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int idx = 0; idx < N_spins; ++idx) {
            const int i = order[idx];
            const int rng = randomNumber(ran);
            const int nc    = neighbor_count[i];
            Bits& spin_i  = spins_Set[i];
            if (rng >= nc) {
                spin_i.ComplementTo();
            }
            else {
                sum.SetAll(0);
                for (int j : neighbors[i]) {
                    xnor_ij = (spin_i == spins_Set[j]);
                    sum += &xnor_ij;
                }
                sum.MultiplyByTwoTo();
                threshold.SetAll((unsigned long long int) rng);
                threshold += &Neighbor_Count[i];
                mask = (sum <= threshold);
                spin_i ^= &mask;
            }
        }
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep+1
            << " (" << ((sweep+1)*100/N_sweeps) << "%)    " << flush;
    }
    toCanonical();
}


