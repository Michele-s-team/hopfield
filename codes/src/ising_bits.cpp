//
//  ising_bits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_bits.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"

// ──────────────────────────────────────────────
// fromCanonical
// neurons_set (±1) → Neurons_Set (Bits)
// bit r de Neurons_Set[i] = 1 if spin +1, 0 si spin -1
// ──────────────────────────────────────────────
void IsingBits::fromCanonical() {
    Neurons_Set.clear();
    Neurons_Set.reserve(N_neurons);
    Neighbor_Count.clear();
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
}


void IsingBits::toCanonical() {
    for (int r = 0; r < n_bits; r++)
        for (int i = 0; i < N_neurons; i++)
            neurons_set[r][i] = -1 + 2 * Neurons_Set[i].Get(r);
}

void IsingBits::convertRandomNumbers() {
    Random_Numbers.resize(N_sweeps, vector<UnsignedInt>(N_neurons, UnsignedInt(N_neurons)));
    for (int step = 0; step < N_sweeps; step++)
        for (int i = 0; i < N_neurons; i++) {
            unsigned long long int v = (unsigned long long int) random_numbers[step][i];
            Random_Numbers[step][i].SetAll(v);
        }
}

void IsingBits::setrandom(double new_BJ, gsl_rng* ran) {
    BJ = new_BJ;
    initRandomNumbers(ran);
    convertRandomNumbers(); 
}

// dans ising_bits.cpp
void IsingBits::setFromExp(double new_BJ, const vector<vector<double>>& exp_base) {
    BJ = new_BJ;
    initRandomNumbersFromExp(exp_base);
    convertRandomNumbers();
}

void IsingBits::evolve() {
    fromCanonical();          // spins + Neighbor_Count
    // Random_Numbers déjà à jour via setrandom() — pas besoin de les reconvertir
    
    Bits xnor_ij;
    Bits mask;
    for (int step = 0; step < N_sweeps; step++) {
        for (int i = 0; i < N_neurons; i++) {
            // BRANCH 1
            if (neighbor_count[i] <= random_numbers[step][i]) {  // ← corrigé
                Neurons_Set[i].ComplementTo();
            }
            else {
                // BRANCH 2
                UnsignedInt sum(1);
                sum.SetAll(0);
                for (int j = 0; j < N_neurons; j++) {
                    if (i != j && connections[i][j]) {
                        xnor_ij = Neurons_Set[i] == Neurons_Set[j];
                        sum += &xnor_ij;
                    }
                }
                sum.MultiplyByTwoTo();
                BitSet temp = Random_Numbers[step][i] + &Neighbor_Count[i];
                mask = sum <= temp;
                Neurons_Set[i] ^= &mask;
            }
        }
        if ((step+1) % (N_sweeps / 10) == 0)
            cout << "\rStep: " << step+1
            << " (" << ((step+1) * 100 / N_sweeps) << "%)    " << flush;
    }
    cout << "\n";
    toCanonical();
}