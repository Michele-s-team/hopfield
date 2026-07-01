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
#include <algorithm>
#include <iostream>

// =====================================================
// ENERGY
// =====================================================

int IsingNoBits::DeltaE(int spin, int realization) {

    int sum = 0;

    for (int j : neighbors[spin])
        sum += spins_set[realization * N + j];

    return spins_set[realization * N + spin] * sum;
}

void IsingNoBits::DeltaE_all(int spin, std::vector<int>& delta_E) {

    delta_E.assign(n_bits, 0);

    // Loop over neighbors once
    for (int k = 0; k < neighbors[spin].size(); ++k) {

        const int j = neighbors[spin][k];

        // replica loop is contiguous in memory
        for (int r = 0; r < n_bits; ++r) {

            delta_E[r] += spins_set[r * N + j];
        }
    }

    // multiply by central spin
    for (int r = 0; r < n_bits; ++r) {
        delta_E[r] *= spins_set[r * N + spin];
    }
}

// =====================================================
// SHARED RNG
// =====================================================

void IsingNoBits::runSweepsSharedRNG(gsl_rng* ran, bool save, double freq) {

    const int progress_stride = std::max(1, getNSweeps() / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < getNSweeps(); ++sweep) {

        for (int step = 0; step < N; ++step) {

            const int spin = gsl_rng_uniform_int(ran, N);
            const int rng = randomNumber(ran, neighbor_count[spin]);

            DeltaE_all(spin, delta_E);

            for (int r = 0; r < n_bits; ++r) {

                const int dE = delta_E[r];

                if (dE <= 0 || rng >= dE) {
                    spins_set[r * N + spin] *= -1;
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << sweep + 1
                      << " (" << (sweep + 1) * 100 / getNSweeps() << "%)    "
                      << std::flush;
    }

    std::cout << "\n";
}

// =====================================================
// INDEPENDENT RNG
// =====================================================

void IsingNoBits::runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq) {

    const int progress_stride = std::max(1, getNSweeps() / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < getNSweeps(); ++sweep) {

        for (int step = 0; step < N; ++step) {

            const int spin = gsl_rng_uniform_int(ran, N);

            DeltaE_all(spin, delta_E);

            for (int r = 0; r < n_bits; ++r) {

                const int dE = delta_E[r];

                if (dE <= 0) {
                    spins_set[r * N + spin] *= -1;
                } else {
                    const int rng = randomNumber(ran, neighbor_count[spin]);
                    if (rng >= dE)
                        spins_set[r * N + spin] *= -1;
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << sweep + 1
                      << " (" << (sweep + 1) * 100 / getNSweeps() << "%)    "
                      << std::flush;
    }

    std::cout << "\n";
}
// =====================================================
// PUBLIC API
// =====================================================

void IsingNoBits::evolve(gsl_rng* ran) {
    runSweepsSharedRNG(ran, false, 0);
}

void IsingNoBits::evolveIndependentRNG(gsl_rng* ran) {
    runSweepsIndependentRNG(ran, false, 0);
}

void IsingNoBits::evolve_save(gsl_rng* ran, double freq, const std::string& filename) {

    OpenSpinFiles(filename);

    runSweepsSharedRNG(ran, true, freq);

    CloseSpinFiles();
}