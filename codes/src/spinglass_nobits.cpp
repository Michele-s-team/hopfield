//
//  spinglassnobits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
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
double SpinglassNoBits::DeltaE(int spin, int r) {
    int sum = 0;
    for (int k = 0; k < neighbors[spin].size(); ++k) {
        int j = neighbors[spin][k];                
        sum += spins_set[r*N_spins+j]*couplings[spin][k][r];
    }
    return spins_set[r*N_spins+spin] * sum;
}

// =====================================================
// SWEEP LOOP
// =====================================================

// Core simulation loop: N_sweeps sweeps of N_spins random flip attempts each.
// random numbers are shared among the 64 realizations
// Saves magnetizations every save_stride sweeps if save=true.
void SpinglassNoBits::runSweepsSharedRNG(gsl_rng* ran, bool save, double freq) {
    const int progress_stride = max(1, N_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;
    int rng;
    int spin;

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int step = 0; step < N_spins; ++step) {
            spin = gsl_rng_uniform_int(ran, N_spins);  // pick a random spin
            rng = randomNumber(ran);
            if (rng >= neighbor_count[spin]) {
                // 1ST BRANCH: unconditional flip: rng exceeds max possible local field
                for (int r = 0; r < n_bits; ++r)
                    spins_set[r*N_spins+spin] *= -1;
            }
            else {
                // 2ND BRANCH:flip realization r only if rng >= ΔE(i, r)
                for (int r = 0; r < n_bits; ++r)
                    if (rng >= DeltaE(spin, r))
                        spins_set[r*N_spins+spin] *= -1;
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveMagnetizations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / N_sweeps << "%)    "
                 << flush;
    }
    cout << "\n";
}


// Core simulation loop: N_sweeps of N_spins random flip attempts each.
// each realization has its own random number
// Saves magnetizations every save_stride sweeps if save=true.
void SpinglassNoBits::runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq) {
    const int progress_stride = max(1, N_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;
    int rng;
    int spin;

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int step = 0; step < N_spins; ++step) {
            spin = gsl_rng_uniform_int(ran, N_spins);  // pick a random spin

                for (int r = 0; r < n_bits; ++r){
                    rng = randomNumber(ran);
                    if (rng >= DeltaE(spin, r))
                        spins_set[r*N_spins+spin] *= -1;
                }
            }

        if (save && (sweep % save_stride == 0))
            SaveMagnetizations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / N_sweeps << "%)    "
                 << flush;
    }
    cout << "\n";
}


// =====================================================
// PUBLIC API
// =====================================================

// Run simulation without saving (thermalization)
void SpinglassNoBits::evolveSharedRNG(gsl_rng* ran) {
    runSweepsSharedRNG(ran, /*save=*/false, 0);
}

// Run simulation without saving (thermalization)
void SpinglassNoBits::evolveIndependentRNG(gsl_rng* ran) {
    runSweepsIndependentRNG(ran, /*save=*/false, 0);
}

// Run simulation and save magnetizations at the given frequency
void SpinglassNoBits::evolveSharedRNG_save(gsl_rng* ran, double freq, const string& filename) {
    OpenCSVFiles(filename); 
    runSweepsSharedRNG(ran, /*save=*/true, freq);
    CloseCSVFiles();
}