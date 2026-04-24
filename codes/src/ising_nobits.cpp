//
//  isingnobits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_nobits.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"

// =====================================================
// ENERGY COMPUTATION
// =====================================================


// ΔE(i, r) = σ_i * Σ_j A_ij σ_j
double IsingNoBits::DeltaE(int neuron, int realization) {
    int sum = 0;
    for (int j : neighbors[neuron])
        sum += neurons_set[realization*n_bits+j];
    return neurons_set[realization*n_bits+neuron] * sum;
}



// =====================================================
// EVOLUTION (SINGLE SWEEP)
// =====================================================
/*
//TO BE FIXED
void IsingNoBits::evolveOneSweep(int sweep, gsl_rng* ran) {
    for (int i = 0; i < N_neurons; ++i) {
        double rng = random_numbers[sweep][i];
        if (rng >= neighbor_count[i]) {
            for (int r = 0; r < n_bits; ++r) neurons_set[r*n_bits+i]  = -neurons_set[r*n_bits+i] ;
        }
        else {
            for (int r = 0; r < n_bits; ++r)
                if (rng >= DeltaE(i, r)) neurons_set[r*n_bits+i]  = -neurons_set[r*n_bits+i] ;
        }
    }
}
*/
// =====================================================
// FULL EVOLUTION
// =====================================================
/*
void IsingNoBits::evolve_modular(gsl_rng* ran) {
    int progress_stride = max(1, N_sweeps / 10);
    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        evolveOneSweep(sweep, ran);
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1 << " (" << ((sweep + 1) * 100 / N_sweeps) << "%) " << flush;
    }
    cout << "\n";
}
*/

void IsingNoBits::evolve_monolithic(gsl_rng* ran) {
    int progress_stride = max(1, N_sweeps / 10);
    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int i = 0; i < N_neurons; ++i) {
            double rng = randomNumber(ran);
            if (rng >= neighbor_count[i]) {
                for (int r = 0; r < n_bits; ++r) neurons_set[r*n_bits+i] = -neurons_set[r*n_bits+i] ;
            }
            else {
                for (int r = 0; r < n_bits; ++r)
                    if (rng >= DeltaE(i, r)) neurons_set[r*n_bits+i]  = -neurons_set[r*n_bits+i] ;
            }
        }
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1 << " (" << ((sweep + 1) * 100 / N_sweeps) << "%)    " << flush;
    }
    cout << "\n";
}