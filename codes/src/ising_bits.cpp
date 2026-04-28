//
//  ising_bits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_bits.hpp"
#include "unsigned_int.hpp"
#include <numeric>
#include <filesystem>
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"

// =====================================================
// STATE CONVERSIONS
// =====================================================

// canonical spins {-1,+1} -> bit representation {0,1}
void IsingBits::fromCanonical(){
    spins_Set.clear();
    Neighbor_Count.clear();

    spins_Set.reserve(N_spins);
    Neighbor_Count.reserve(N_spins);

    for (int i = 0; i < N_spins; ++i) {
        Bits spin_tmp;
        for (int r = 0; r < n_bits; ++r) {
            int bit = (spins_set[r*N_spins+i] + 1) / 2;
            spin_tmp.Set(r, bit);
        }
        UnsignedInt neighbor_tmp(neighbor_count[i]);
        neighbor_tmp.SetAll(neighbor_count[i]);

        spins_Set.push_back(spin_tmp);
        Neighbor_Count.push_back(neighbor_tmp);
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void IsingBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N_spins; ++i)
            spins_set[r*N_spins+i] = -1 + 2 * spins_Set[i].Get(r);
}

// =====================================================
// RANDOM THRESHOLDS (disabled)
// =====================================================

/*
// Pre-convert double RNG values to UnsignedInt bitwise format
void IsingBits::convertRandomNumbers()
{
    Random_Numbers.clear();
    Random_Numbers.resize(
        N_sweeps,
        vector<UnsignedInt>(
            N_spins,
            UnsignedInt(N_spins)
        )
    );

    for (int sweep = 0; sweep < N_sweeps; ++sweep)
        for (int i = 0; i < N_spins; ++i) {
            unsigned long long v =
                (unsigned long long) random_numbers[sweep][i];
            Random_Numbers[sweep][i].SetAll(v);
        }
}
*/

/*
void IsingBits::initRandomNumbers(gsl_rng* ran){
    IsingModel::initRandomNumbers(ran);
    convertRandomNumbers();
}

void IsingBits::initRandomNumbersFromExp(const vector<vector<double>>& exp_base){
    IsingModel::initRandomNumbersFromExp(exp_base);
    convertRandomNumbers();
}
*/

// =====================================================
// EVOLUTION CONTEXT (disabled)
// =====================================================

/*
// Sync canonical <-> bitwise representation and prepare RNG
void IsingBits::initEvolveContext(){
    fromCanonical();
    convertRandomNumbers();
}
*/

// =====================================================
// METROPOLIS DYNAMICS
// =====================================================

// Attempt a spin flip at site i using the bitwise Metropolis rule:
//   - if rng >= nc: unconditional flip (infinite temperature limit)
//   - otherwise: flip only realizations where 2*aligned_neighbors <= rng + nc
void IsingBits::tryFlip(int i, int rng,
                         Bits& xnor_ij, Bits& mask,
                         UnsignedInt& sum, BitSet& threshold){
    Bits& spin_i = spins_Set[i];
    const int nc = neighbor_count[i];

    // unconditional flip: rng exceeds max possible local field
    if (rng >= nc) {
        spin_i.ComplementTo();
        return;
    }

    // count aligned neighbors across all realizations simultaneously
    sum.SetAll(0);
    for (int j : neighbors[i]) {
        xnor_ij = (spin_i == spins_Set[j]);
        sum += &xnor_ij;
    }
    sum.MultiplyByTwoTo();

    // threshold = rng + nc, flip where sum <= threshold
    threshold.SetAll(static_cast<unsigned long long>(rng));
    threshold += &Neighbor_Count[i];

    mask = (sum <= threshold);
    spin_i ^= &mask;
}

// =====================================================
// OBSERVABLES
// =====================================================

// Compute magnetization m = (2*ones - N) / N for each realization
void IsingBits::GetMagnetizations(vector<double>& magnetizations){
    vector<int> ones(n_bits, 0);

    for (int i = 0; i < N_spins; ++i)
        for (int r = 0; r < n_bits; ++r)
            ones[r] += spins_Set[i].Get(r);

    for (int r = 0; r < n_bits; ++r)
        magnetizations[r] = (2.0 * ones[r] - N_spins) / N_spins;
}

// =====================================================
// SWEEP LOOP
// =====================================================

// Core simulation loop: N_sweeps sweeps of N_spins random flip attempts each.
// Saves magnetizations every save_stride sweeps if save=true.
void IsingBits::runSweeps(gsl_rng* ran, bool save, double freq){
    // temporaries allocated once for all sweeps and all flips
    Bits        xnor_ij, mask;
    UnsignedInt sum((unsigned long long int)(neighbor_count[0] * 2));
    BitSet      threshold(N_spins);

    const int progress_stride = max(1, N_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {

        for (int step = 0; step < N_spins; ++step) {
            int i = gsl_rng_uniform_int(ran, N_spins);  // pick a random spin
            tryFlip(i, randomNumber(ran), xnor_ij, mask, sum, threshold);
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
void IsingBits::evolve(gsl_rng* ran){
    fromCanonical();
    runSweeps(ran, /*save=*/false, 0);
    toCanonical();
}

// Run simulation and save magnetizations at the given frequency
void IsingBits::evolve_save(gsl_rng* ran, double freq){
    fromCanonical();
    OpenCSVFiles("../results/magnetizations/magnetizations_bits.csv");
    runSweeps(ran, /*save=*/true, freq);
    CloseCSVFiles();
    toCanonical();
}