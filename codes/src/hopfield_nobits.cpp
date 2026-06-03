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
        int contrib = 0;
        for (int p = 0; p < P; ++p)
            contrib += patterns[p][spin][r] * patterns[p][j][r];
        sum += contrib * spins_set[r * N + j];
    }
    return spins_set[r * N + spin] * sum;
}
// =====================================================
// SHARED RNG (same random threshold among replicas)
// =====================================================

void HopfieldNoBits::runSweepsSharedRNG(gsl_rng* ran, bool save, double freq) {
    const int total_sweeps = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            int spin = gsl_rng_uniform_int(ran, N);
            int rng = randomNumber(ran, P * neighbor_count[spin], N);

            if (rng >= P * neighbor_count[spin]) {
                // 1ST BRANCH: unconditional flip in all replicas
                for (int r = 0; r < n_bits; ++r)
                    spins_set[r * N + spin] *= -1;
            } else {
                // 2ND BRANCH: flip replica r only if rng >= DeltaE(spin, r)
                for (int r = 0; r < n_bits; ++r)
                    if (rng >= DeltaE(spin, r))
                        spins_set[r * N + spin] *= -1;
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveMagnetizations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                 << flush;
    }
    cout << "\n";
}

// =====================================================
// INDEPENDENT RNG (different random threshold among replicas)
// =====================================================

void HopfieldNoBits::runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq) {
    const int total_sweeps = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            int spin = gsl_rng_uniform_int(ran, N);

            for (int r = 0; r < n_bits; ++r) {
                int rng = randomNumber(ran, neighbor_count[spin]);
                if (rng >= DeltaE(spin, r))
                    spins_set[r * N + spin] *= -1;
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveMagnetizations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                 << flush;
    }
    cout << "\n";
}

// =====================================================
// PUBLIC API
// =====================================================

void HopfieldNoBits::evolveSharedRNG(gsl_rng* ran) {
    runSweepsSharedRNG(ran, false, 0.0);
}

void HopfieldNoBits::evolveIndependentRNG(gsl_rng* ran) {
    runSweepsIndependentRNG(ran, false, 0.0);
}

void HopfieldNoBits::evolveSharedRNG_save(gsl_rng* ran, double freq, const string& folder) {
    OpenCSVFiles(folder);
    runSweepsSharedRNG(ran, true, freq);
    CloseCSVFiles();
}

