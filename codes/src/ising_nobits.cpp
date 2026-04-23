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
        sum += neurons_set[realization][j];
    return neurons_set[realization][neuron] * sum;
}



// =====================================================
// EVOLUTION (SINGLE SWEEP)
// =====================================================


void IsingNoBits::evolveOneSweep(int sweep) {
    for (int i = 0; i < N_neurons; ++i) {
        double rho = random_numbers[sweep][i];
        if (rho >= neighbor_count[i]) {
            for (int r = 0; r < n_bits; ++r) neurons_set[r][i] = -neurons_set[r][i];
        }
        else {
            for (int r = 0; r < n_bits; ++r)
                if (rho >= DeltaE(i, r)) neurons_set[r][i] = -neurons_set[r][i];
        }
    }
}
// =====================================================
// FULL EVOLUTION
// =====================================================

void IsingNoBits::evolve_modular() {
    int progress_stride = max(1, N_sweeps / 10);
    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        evolveOneSweep(sweep);
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1 << " (" << ((sweep + 1) * 100 / N_sweeps) << "%) " << flush;
    }
    cout << "\n";
}

void IsingNoBits::evolve_monolithic() {
    int progress_stride = max(1, N_sweeps / 10);
    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int i = 0; i < N_neurons; ++i) {
            double rho = random_numbers[sweep][i];
            if (rho >= neighbor_count[i]) {
                for (int r = 0; r < n_bits; ++r) neurons_set[r][i] = -neurons_set[r][i];
            }
            else {
                for (int r = 0; r < n_bits; ++r)
                    if (rho >= DeltaE(i, r)) neurons_set[r][i] = -neurons_set[r][i];
            }
        }
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1 << " (" << ((sweep + 1) * 100 / N_sweeps) << "%)    " << flush;
    }
    cout << "\n";
}