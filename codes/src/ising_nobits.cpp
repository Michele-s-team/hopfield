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


// ──────────────────────────────────────────────
// DeltaE
// returns sigma_i * sum_{j neighbouring i} sigma_j
// ──────────────────────────────────────────────
double IsingNoBits::DeltaE(int neuron, int realization) {
    int sum = 0;
    for (int j = 0; j < N_neurons; j++)
        if (connections[neuron][j] == 1)
            sum += neurons_set[realization][j];
    return neurons_set[realization][neuron] * sum;
}

// ──────────────────────────────────────────────
// evolve
// at each step, for each neuron i:
// a threshold rho is drawn, and the spin is flipped
// in all realisations where rho >= DeltaE
// ──────────────────────────────────────────────
void IsingNoBits::evolve() {
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