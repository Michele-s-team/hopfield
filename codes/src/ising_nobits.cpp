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

    for (int j = 0; j < N_neurons; ++j) {

        if (connections[neuron][j]) {

            sum += neurons_set[realization][j];
        }
    }

    return neurons_set[realization][neuron] * sum;
}

// =====================================================
// EVOLUTION (SINGLE SWEEP)
// =====================================================

void IsingNoBits::evolveOneSweep(int step) {
    for (int i = 0; i < N_neurons; ++i) {

        double rho = random_numbers[step][i];
        for (int r = 0; r < n_bits; ++r) {

            if (rho >= DeltaE(i, r)) {
                neurons_set[r][i] =
                    -neurons_set[r][i];
            }
        }
    }
}

// =====================================================
// EVOLUTION DRIVER
// =====================================================

void IsingNoBits::evolve_modular() {
    int progress_stride = max(1, N_sweeps / 10);

    for (int step = 0; step < N_sweeps; ++step) {

        evolveOneSweep(step);

        if ((step + 1) % progress_stride == 0) {

            cout << "\rStep: " << step + 1 << " (" << ((step + 1) * 100 / N_sweeps)<< "%) " << flush;
        }
    }

    cout << "\n";
}
void IsingNoBits::evolve_monolithic() {
    for (int step = 0; step < N_sweeps; step++) {
        if (step % (N_sweeps / 10) == 0)
            cout << "\rStep: " << step
                 << " (" << (step * 100 / N_sweeps) << "%)    " << flush;

        for (int i = 0; i < N_neurons; i++) {
            double rho = random_numbers[step][i];
            for (int r = 0; r < n_bits; r++) {
                if (rho >= DeltaE(i, r))
                    neurons_set[r][i] = -neurons_set[r][i];
            }
        }
    }
    cout << "\n";
}