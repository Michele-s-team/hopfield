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
    Bits_Spins_Set.clear();
    Neighbor_Count.clear();

    Bits_Spins_Set.reserve(N);
    Neighbor_Count.reserve(N);
    Bits spin_tmp;

    for (int i = 0; i < N; ++i) {
        for (int r = 0; r < n_bits; ++r) {
            int bit = (spins_set[r*N+i] + 1) / 2;
            spin_tmp.Set(r, bit);
        }
        UnsignedInt neighbor_tmp(neighbor_count[i]);
        neighbor_tmp.SetAll((unsigned long long int) neighbor_count[i]);
        Bits_Spins_Set.push_back(spin_tmp);
        Neighbor_Count.push_back(neighbor_tmp);
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void IsingBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N; ++i)
            spins_set[r*N+i] = -1 + 2 * Bits_Spins_Set[i].Get(r);
}

// =====================================================
// OBSERVABLES
// =====================================================

// Compute magnetization m = (2*ones - N)/N for each realization 
void IsingBits::GetMagnetizations(vector<double>& magnetizations){
    magnetizations.resize(n_bits);
    vector<int> ones(n_bits, 0);

    for (int i = 0; i < N; ++i)
        for (int r = 0; r < n_bits; ++r)
            ones[r] += Bits_Spins_Set[i].Get(r);

    for (int r = 0; r < n_bits; ++r)
        magnetizations[r] = (2.0 * ones[r] - N) / N;
}

// =====================================================
// METROPOLIS DYNAMICS
// =====================================================

// Core simulation loop: N_sweeps sweeps of N random flip attempts each.
void IsingBits::runSweeps(gsl_rng* ran, bool save, double freq){
    // temporaries allocated once for all sweeps and all flips
    Bits xnor_ij, mask;
    UnsignedInt sum((unsigned long long int)(neighbor_count[0] * 2));
    UnsignedInt threshold((unsigned long long int)(neighbor_count[0] * 2));
    int rng;
    int spin;

    const int N_sweeps = getNSweeps();  // Use base class method
    const int progress_stride = max(1, N_sweeps / 10);
    const int save_stride = (save && freq > 0) ? max(1, (int)round(1.0 / freq)) : 0;

    for (int sweep = 0; sweep < N_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            spin = gsl_rng_uniform_int(ran, N);
            // Use base class randomNumber method
            rng = randomNumber(ran, neighbor_count[0]);
            Bits& Bits_Spin_i = Bits_Spins_Set[spin];

            // 1ST BRANCH: UNCONDITIONAL FLIP IN EVERY REPLICA
            if (rng >= neighbor_count[spin]) {  
                Bits_Spin_i.ComplementTo();
                continue;
            }

            // 2ND BRANCH: NEIGHBOR-DEPENDENT FLIP
            sum.SetAll(0);
            for (int j : neighbors[spin]) {
                xnor_ij = ~(Bits_Spin_i ^ Bits_Spins_Set[j]);
                sum += &xnor_ij;
            }
            sum.MultiplyByTwoTo();  

            threshold.SetAll((unsigned long long int) rng);
            threshold += &Neighbor_Count[spin];

            mask = (sum <= threshold);
            Bits_Spin_i ^= &mask;
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
    runSweeps(ran, /*save=*/false, 0.0);
    toCanonical();
}

// Run simulation and save magnetizations at the given frequency
void IsingBits::evolve_save(gsl_rng* ran, double freq, const string& folder){
    fromCanonical();
    OpenSpinFiles(folder);  // Use base class method
    runSweeps(ran, /*save=*/true, freq);
    CloseSpinFiles();       // Use base class method
    toCanonical();
}