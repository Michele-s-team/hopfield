//
//  hopfield_nobits.cpp
//  hopfield
//
//  Created by Bastien on 12/05/2026.
//  Condensed on 23/07/2026: the three ΔE formulations (neighbors / overlaps /
//  overlaps-non-neighbors) and the shared-RNG / independent-RNG sweep loops
//  are each merged into a single parameterized implementation, selected by
//  SweepMode. runSweepsSharedRNG and runSweepsIndependentRNG remain the only
//  two public entry points.
//

#include "hopfield_nobits.hpp"

#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"
#include <numeric>
#include <algorithm>

// =====================================================
// ENERGY COMPUTATION (all replicas at once, dispatched by mode)
// =====================================================
//   NEIGHBORS              : ΔE from direct neighbor couplings   (sparse graphs)
//   OVERLAPS               : ΔE from pure pattern overlaps       (dense/fully-connected graphs)
//   OVERLAPS_NON_NEIGHBORS : overlaps corrected by non-neighbor couplings (dense, non-complete graphs)
void HopfieldNoBits::DeltaE_all(int spin, SweepMode mode, vector<int>& delta_E) {
    delta_E.assign(n_bits, 0);

    switch (mode) {
        case SweepMode::NEIGHBORS:
            for (int k = 0; k < neighbors[spin].size(); ++k) {
                int j = neighbors[spin][k];
                for (int r = 0; r < n_bits; ++r)
                    delta_E[r] += couplings[spin][k * n_bits + r] * spins_set[r * N + j];
            }
            for (int r = 0; r < n_bits; ++r)
                delta_E[r] *= spins_set[r * N + spin];
            break;

        case SweepMode::OVERLAPS:
            for (int mu = 0; mu < P; ++mu)
                for (int r = 0; r < n_bits; ++r)
                    delta_E[r] += patterns[spin * P * n_bits + mu * n_bits + r] * overlaps[mu * n_bits + r];
            for (int r = 0; r < n_bits; ++r) {
                delta_E[r] *= spins_set[r * N + spin];
                delta_E[r] -= P;
            }
            break;

        case SweepMode::OVERLAPS_NON_NEIGHBORS:
            for (int r = 0; r < n_bits; ++r) {
                for (int mu = 0; mu < P; ++mu)
                    delta_E[r] += patterns[spin * P * n_bits + mu * n_bits + r] * overlaps[mu * n_bits + r];
                for (int i = 0; i < non_neighbors[spin].size(); ++i) {
                    int j = non_neighbors[spin][i];
                    delta_E[r] -= couplings_nonneighbors[spin][i * n_bits + r] * spins_set[r * N + j];
                }
                delta_E[r] *= spins_set[r * N + spin];
                delta_E[r] -= P;
            }
            break;
    }
}

// Flips spin `spin` for replica `r`, updating the pattern overlaps first
// whenever the mode depends on them (every mode except NEIGHBORS).
void HopfieldNoBits::FlipSpin(int spin, int r, SweepMode mode) {
    if (mode != SweepMode::NEIGHBORS) {
        for (int mu = 0; mu < P; ++mu) {
            overlaps[mu * n_bits + r] -=
                2 * patterns[spin * P * n_bits + mu * n_bits + r] * spins_set[r * N + spin];
        }
    }
    spins_set[r * N + spin] *= -1;
}

// =====================================================
// CORE SWEEP LOOPS (mode-agnostic)
// =====================================================

void HopfieldNoBits::runSweepsSharedRNG_core(gsl_rng* ran, bool save, double freq, SweepMode mode) {
    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            const int spin = gsl_rng_uniform_int(ran, N);
            const int deg_spin = degrees[spin];
            const int rng = randomNumber(ran, P * deg_spin, N);

            if (rng >= P * deg_spin) {
                // Same threshold shared by every replica: unconditional flip
                for (int r = 0; r < n_bits; ++r)
                    FlipSpin(spin, r, mode);
            } else {
                DeltaE_all(spin, mode, delta_E);
                for (int r = 0; r < n_bits; ++r) {
                    if (delta_E[r] <= 0 || rng >= delta_E[r])
                        FlipSpin(spin, r, mode);
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
    std::cout << endl;
}

void HopfieldNoBits::runSweepsIndependentRNG_core(gsl_rng* ran, bool save, double freq, SweepMode mode) {
    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            const int spin = gsl_rng_uniform_int(ran, N);
            const int deg_spin = degrees[spin];

            DeltaE_all(spin, mode, delta_E);
            for (int r = 0; r < n_bits; ++r) {
                const int rng = randomNumber(ran, P * deg_spin, N);
                const bool flip = (rng >= P * deg_spin) || (delta_E[r] <= 0 || rng >= delta_E[r]);
                if (flip)
                    FlipSpin(spin, r, mode);
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }
    std::cout << endl;
}

// =====================================================
// PUBLIC ENTRY POINTS
// =====================================================

void HopfieldNoBits::runSweepsSharedRNG(gsl_rng* ran, bool save, double freq) {
    const double alpha = (double)P / N;
    const double mean_degree = std::accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
    const double mean_density = mean_degree / (N - 1);

    if (alpha + 0.5 <= mean_density) {
        compute_overlaps();
        const int min_deg = *std::min_element(degrees.begin(), degrees.end());
        if (min_deg == N - 1) {
            cout << "Shared RNG: Overlaps spin update algorithm" << endl;
            runSweepsSharedRNG_core(ran, save, freq, SweepMode::OVERLAPS);
        } else {
            initCouplings_nonNeighbors();
            cout << "Shared RNG: Overlaps and non-neighbors couplings spin update algorithm" << endl;
            runSweepsSharedRNG_core(ran, save, freq, SweepMode::OVERLAPS_NON_NEIGHBORS);
        }
    } else {
        initCouplings();
        cout << "Shared RNG: Neighbors couplings spin update algorithm" << endl;
        runSweepsSharedRNG_core(ran, save, freq, SweepMode::NEIGHBORS);
    }
}

void HopfieldNoBits::runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq) {
    const double alpha = (double)P / N;
    const double mean_degree = std::accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
    const double mean_density = mean_degree / (N - 1);

    if (alpha + 0.5 <= mean_density) {
        compute_overlaps();
        const int min_deg = *std::min_element(degrees.begin(), degrees.end());
        if (min_deg == N - 1) {
            cout << "Independent RNG: Overlaps spin update algorithm" << endl;
            runSweepsIndependentRNG_core(ran, save, freq, SweepMode::OVERLAPS);
        } else {
            initCouplings_nonNeighbors();
            cout << "Independent RNG: Overlaps and non-neighbors couplings spin update algorithm" << endl;
            runSweepsIndependentRNG_core(ran, save, freq, SweepMode::OVERLAPS_NON_NEIGHBORS);
        }
    } else {
        initCouplings();
        cout << "Independent RNG: Neighbors couplings spin update algorithm" << endl;
        runSweepsIndependentRNG_core(ran, save, freq, SweepMode::NEIGHBORS);
    }
}