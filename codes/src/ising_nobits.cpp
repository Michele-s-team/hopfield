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
double IsingNoBits::DeltaE(int spin, int realization) {
    int sum = 0;
    for (int j : neighbors[spin])
        sum += spins_set[realization*N_spins+j];
    return spins_set[realization*N_spins+spin] * sum;
}



// =====================================================
// EVOLUTION (SINGLE SWEEP)
// =====================================================
/*
//TO BE FIXED
void IsingNoBits::evolveOneSweep(int sweep, gsl_rng* ran) {
    for (int i = 0; i < N_spins; ++i) {
        double rng = random_numbers[sweep][i];
        if (rng >= neighbor_count[i]) {
            for (int r = 0; r < n_bits; ++r) spins_set[r*N_spins+i]  = -spins_set[r*N_spins+i] ;
        }
        else {
            for (int r = 0; r < n_bits; ++r)
                if (rng >= DeltaE(i, r)) spins_set[r*N_spins+i]  = -spins_set[r*N_spins+i] ;
        }
    }
}
*/
// =====================================================
// FULL EVOLUTION
// =====================================================
/*
void IsingNoBits::evolve_modular(gsl_rng* ran) {
    int progress_stride = max(1, N_sweeps / 10);
    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        evolveOneSweep(sweep, ran);
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1 << " (" << ((sweep + 1) * 100 / N_sweeps) << "%) " << flush;
    }
    cout << "\n";
}
*/

void IsingNoBits::evolve(gsl_rng* ran) {
    int progress_stride = max(1, N_sweeps / 10);
    int rng;
    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int i = 0; i < N_spins; ++i) {
            rng = randomNumber(ran);
            if (rng >= neighbor_count[i]) {
                for (int r = 0; r < n_bits; ++r) spins_set[r*N_spins+i] = -spins_set[r*N_spins+i] ;
            }
            else {
                for (int r = 0; r < n_bits; ++r)
                    if (rng >= DeltaE(i, r)) spins_set[r*N_spins+i]  = -spins_set[r*N_spins+i] ;
            }
        }
        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1 << " (" << ((sweep + 1) * 100 / N_sweeps) << "%)    " << flush;
    }
    cout << "\n";
}