//
//  hopfield_nobits.cpp
//  hopfield
//
//  Created by Bastien on 12/05/2026.
//

#include "hopfield_nobits.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"

// =====================================================
// ENERGY COMPUTATION
// =====================================================

// Local energy cost of flipping spin i in realization r:
//   ΔE(i, r) = 2 * σ_i(r) * Σ_j J_ij(r) σ_j(r)
// For Hopfield: J_ij = (1/N) * Σ_p ξ_i^p ξ_j^p
int HopfieldNoBits::DeltaE(int spin, int r) {
    int sum = 0.0;
    for (int k = 0; k < neighbors[spin].size(); ++k) {
        int j = neighbors[spin][k];
        sum += couplings[spin][k][r] * spins_set[r * N + j];
    }
    return spins_set[r * N + spin] * sum;
}
// =====================================================
// ENERGY COMPUTATION (all replicas at once)
// =====================================================
// Calcule DeltaE(spin, r) pour tous les réplicas r en une seule passe sur les
// voisins, au lieu de reboucler sur neighbors[spin] à chaque r séparément.
// Boucle k externe, r interne : accès contigu à couplings[spin][k][r].
void HopfieldNoBits::DeltaE_all(int spin, vector<int>& delta_E) {
    delta_E.assign(n_bits, 0);
    for (int k = 0; k < neighbors[spin].size(); ++k) {
        int j = neighbors[spin][k];
        for (int r = 0; r < n_bits; ++r) {
            delta_E[r] += couplings[spin][k][r] * spins_set[r * N + j];
        }
    }
    for (int r = 0; r < n_bits; ++r) {
        delta_E[r] *= spins_set[r * N + spin];
    }
}

// =====================================================
// SHARED RNG (same random threshold among replicas)
// =====================================================
void HopfieldNoBits::runSweepsSharedRNG(gsl_rng* ran, bool save, double freq) {

    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {

        for (int step = 0; step < N; ++step) {

            const int spin = gsl_rng_uniform_int(ran, N);
            const int threshold = P * neighbor_count[spin];
            const int rng = randomNumber(ran, threshold, N);

            // Branch 1: unconditional flip
            if (rng >= threshold) {

                for (int r = 0; r < n_bits; ++r)
                    spins_set[r * N + spin] *= -1;

            } else {

                // Branch 2: compute all ΔE once
                DeltaE_all(spin, delta_E);

                for (int r = 0; r < n_bits; ++r) {

                    const int dE = delta_E[r];

                    if (dE <= 0 || rng >= dE) {
                        spins_set[r * N + spin] *= -1;
                    }
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << "\n";
}
// =====================================================
// INDEPENDENT RNG (different random threshold among replicas)
// =====================================================
void HopfieldNoBits::runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq) {

    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {

        for (int step = 0; step < N; ++step) {

            const int spin = gsl_rng_uniform_int(ran, N);

            // Compute ΔE once per spin
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
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << "\n";
}
// =====================================================
// PUBLIC API
// =====================================================

void HopfieldNoBits::evolve(gsl_rng* ran) {
    runSweepsSharedRNG(ran, false, 0.0);
}

void HopfieldNoBits::evolveIndependentRNG(gsl_rng* ran) {
    runSweepsIndependentRNG(ran, false, 0.0);
}

void HopfieldNoBits::evolve_save(gsl_rng* ran, double freq, const string& folder) {
    OpenSpinFiles();
    runSweepsSharedRNG(ran, true, freq);
    CloseSpinFiles();
}

