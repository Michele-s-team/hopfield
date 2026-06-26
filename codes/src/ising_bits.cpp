//
//  ising_bits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_bits.hpp"
#include "unsigned_int.hpp"
#include <numeric>
#include <algorithm>
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
// DIRECT BITWISE INITIALIZATION
// =====================================================


// Initialize spins directly in bit-sliced representation.
// Replaces: initSpins() + fromCanonical() step 1.
void IsingBits::initSpinsBits(gsl_rng* ran) {
    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    Bits_Spins_Set.clear();
    Bits_Spins_Set.resize(N);
    Neighbor_Count.clear();
    Neighbor_Count.reserve(N);

    for (int i = 0; i < N; ++i) {
        for (int r = 0; r < n_bits; ++r)
            Bits_Spins_Set[i].Set(r, randomBit(ran));

        UnsignedInt nc_tmp((unsigned long long) max_deg);
        nc_tmp.SetAll((unsigned long long) neighbor_count[i]);
        Neighbor_Count.push_back(nc_tmp);
    }
}

void IsingBits::initSpinsFromConfigBits(vector<Bits> Config) {
    Bits_Spins_Set = Config;
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

// Converts spin configurations (already in bits) into packed blocks for each realisation
// Uses the same MSB-first convention as PackBlock (and SavePatterns), so that
// spins and patterns are bit-comparable.
void IsingBits::GetSpinConfigurations(vector<vector<uint64_t>>& configs){
    const int n_blocks = num_blocks(N);
    configs.assign(n_bits, vector<uint64_t>(n_blocks, 0));

    for (int b = 0; b < n_blocks; ++b){
        int start = b * BITS_PER_BLOCK;
        int end   = min(N, start + BITS_PER_BLOCK);

        for (int i = start; i < end; ++i){
            for (int r = 0; r < n_bits; ++r){
                configs[r][b] <<= 1;
                if (Bits_Spins_Set[i].Get(r))
                    configs[r][b] |= 1ULL;
            }
        }
    }
}

// =====================================================
// METROPOLIS DYNAMICS
// =====================================================

// Core simulation loop: N_sweeps sweeps of N random flip attempts each.
void IsingBits::runSweeps(gsl_rng* ran, bool save, double freq, int shift=0){
    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());
    // temporaries allocated once for all sweeps and all flips
    Bits xnor_ij, mask;
    UnsignedInt sum((unsigned long long int)(max_deg * 2));
    UnsignedInt threshold((unsigned long long int)(max_deg * 2));
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
            SaveSpinConfigurations(shift + sweep);

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

// Run simulation and save at given frequency — fully bitwise initialization
void IsingBits::evolve_save_bits(gsl_rng* ran, double freq, const string& filename){
    initSpinsBits(ran);
    cout << "Bitwise initialization done" << endl;

    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    cout << "evolve_save_bits called, opening: " << filename << endl;
    runSweeps(ran, /*save=*/true, freq);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save_bits called, closing: " << filename << endl;
}

// Run simulation without saving — fully bitwise initialization
void IsingBits::evolve_bits(gsl_rng* ran, const string& filename){
    initSpinsBits(ran);

    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_bits called, closing: " << filename << endl;
}