//
//  spinglass_nobits.cpp
//  hopfield
//
//  Created by Bastien on 12/05/2026.
//
#include "spinglass_nobits.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"

// =====================================================
// ENERGY COMPUTATION
// =====================================================

// Local energy cost of flipping spin i in realization r:
//   ΔE(i, r) =   Σ_j J_ij(r) σ_i(r) σ_j(r)
int SpinGlassNoBits::DeltaE(int spin, int r) {
    int sum = 0;
    for (int k = 0; k < neighbors[spin].size(); ++k) {
        int j = neighbors[spin][k];                
        sum += spins_set[r*N+j]*couplings[spin][k][r];
    }
    return spins_set[r*N+spin] * sum;
}

// =====================================================
// SHARED RNG
// =====================================================

void SpinGlassNoBits::runSweepsSharedRNG(gsl_rng* ran, bool save, double freq) {

    const int progress_stride = std::max(1, getNSweeps() / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    int rng;
    int spin;

    for (int sweep = 0; sweep < getNSweeps(); ++sweep) {

        for (int step = 0; step < N; ++step) {

            spin = gsl_rng_uniform_int(ran, N);

            rng = randomNumber(ran, neighbor_count[spin]);

            if (rng >= neighbor_count[spin]) {

                for (int r = 0; r < n_bits; ++r)
                    spins_set[r * N + spin] *= -1;
            }
            else {

                for (int r = 0; r < n_bits; ++r)
                    if (rng >= DeltaE(spin, r))
                        spins_set[r * N + spin] *= -1;
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveMagnetizations(sweep);

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

void SpinGlassNoBits::runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq) {

    const int progress_stride = std::max(1, getNSweeps() / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    int rng;
    int spin;

    for (int sweep = 0; sweep < getNSweeps(); ++sweep) {

        for (int step = 0; step < N; ++step) {

            spin = gsl_rng_uniform_int(ran, N);

            for (int r = 0; r < n_bits; ++r) {

                rng = randomNumber(ran, neighbor_count[spin]);

                if (rng >= DeltaE(spin, r))
                    spins_set[r * N + spin] *= -1;
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveMagnetizations(sweep);

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

void SpinGlassNoBits::evolve(gsl_rng* ran) {
    runSweepsSharedRNG(ran, false, 0);
}

void SpinGlassNoBits::evolveIndependentRNG(gsl_rng* ran) {
    runSweepsIndependentRNG(ran, false, 0);
}

void SpinGlassNoBits::evolve_save(gsl_rng* ran, double freq, const std::string& filename) {

    OpenSpinFiles(filename);

    runSweepsSharedRNG(ran, true, freq);

    CloseSpinFiles();
}