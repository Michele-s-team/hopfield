//
//  isingmodel.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_model.hpp"
#include "gsl_rng.h"
#include "gsl_randist.h"

#include "gsl_math.h"
#include <vector>

#include "lib.hpp"
#include "main.hpp"


// ──────────────────────────────────────────────
// Constructeur
// ──────────────────────────────────────────────
IsingModel::IsingModel(int L, double BJ, int N_sweeps)
    : L(L),
      N_neurons(L * L),
      N_sweeps(N_sweeps),
      BJ(BJ),
      connections(L * L, vector<int>(L * L, 0)),
      neighbor_count(L * L, 0),
      random_numbers(N_sweeps, vector<int>(L * L, 0)),
      neurons_set(n_bits, vector<int>(L * L, 0))  // ← ici
{}

// ──────────────────────────────────────────────
// Initialisation
// ──────────────────────────────────────────────
void IsingModel::init(gsl_rng* ran) {
    // Connexions 2D avec conditions aux bords périodiques
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            int i = x + L * y;

            int neighbors[4] = {
                (x + 1) % L       + L * y,
                (x - 1 + L) % L   + L * y,
                x + L * ((y + 1) % L),
                x + L * ((y - 1 + L) % L)
            };

            for (int k = 0; k < 4; k++) {
                connections[i][neighbors[k]] = 1;
            }
            neighbor_count[i] = 4;
        }
    }

    // Random initial states (spins ±1)
    for (int r = 0; r < n_bits; r++)
        for (int i = 0; i < N_neurons; i++)
            neurons_set[r][i] = 2 * gsl_rng_uniform_int(ran, 2) - 1;

    // Random number for each step (not sweep) (same for all 64 parallel realizations)
    for (int step = 0; step < N_sweeps; step++)
        for (int i = 0; i < N_neurons; i++)
            random_numbers[step][i] = (int) min(
                (double) N_neurons,
                1.0 / (2.0 * BJ) * gsl_ran_exponential(ran, 1.0)
            );
}

// ──────────────────────────────────────────────
// Getter
// ──────────────────────────────────────────────
const vector<vector<int>>& IsingModel::getState() const {
    return neurons_set;
}