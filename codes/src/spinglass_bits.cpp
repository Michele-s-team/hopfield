//
//  spinglass_bits.cpp
//  hopfield
//
//  Created by Bastien on 7/05/2026.
//

#include "spinglass_bits.hpp"
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
void SpinGlassBits::fromCanonical(){
    Bits_Spins_Set.clear();
    Neighbor_Count.clear();
    Couplings.clear();
    Bits_Spins_Set.reserve(N);
    Neighbor_Count.reserve(N);
    Couplings.reserve(N);

    for (int i = 0; i < N; ++i) {
        Bits spin_tmp;                          // déclaré ici : réinitialisé à chaque i
        for (int r = 0; r < n_bits; ++r) {
            int spin = (spins_set[r*N+i] + 1) / 2;
            spin_tmp.Set(r, spin);
        }

        Couplings.push_back(vector<Bits>());
        Couplings.back().reserve(couplings[i].size());

        for (int j = 0; j < (int)couplings[i].size(); ++j) {
            Bits coupling_tmp;                  // déclaré ici : réinitialisé à chaque j
            for (int r = 0; r < n_bits; ++r) {
                int coupling = (couplings[i][j][r] + 1) / 2;
                coupling_tmp.Set(r, coupling);
            }
            Couplings.back().push_back(coupling_tmp);
        }

        UnsignedInt neighbor_tmp(neighbor_count[i]);
        neighbor_tmp.SetAll((unsigned long long int) neighbor_count[i]);
        Bits_Spins_Set.push_back(spin_tmp);
        Neighbor_Count.push_back(neighbor_tmp);
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void SpinGlassBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N; ++i)
            spins_set[r*N+i] = -1 + 2 * Bits_Spins_Set[i].Get(r);
}

// =====================================================
// DIRECT BITWISE INITIALIZATION
// =====================================================


// Initialize spins directly in bit-sliced representation.
// Replaces: initSpins() + fromCanonical() step 1.
void SpinGlassBits::initSpinsBits(gsl_rng* ran) {
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

void SpinGlassBits::initSpinsFromConfigBits(vector<Bits> Config) {
    Bits_Spins_Set = Config;
}


// Initialize Couplings directly in bit-sliced representation
void SpinGlassBits::initCouplingsBits(gsl_rng* ran) {
    Couplings.clear();
    Couplings.resize(N);
    for (int i = 0; i < N; ++i) {
        Couplings[i].clear();
        Couplings[i].resize(neighbors[i].size());
    }

    // Generate once for i < j, then mirror (J_ij = J_ji)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < (int)neighbors[i].size(); ++k) {
            int j = neighbors[i][k];
            if (j <= i) continue;

            Bits coupling_tmp;
            for (int r = 0; r < n_bits; ++r)
                coupling_tmp.Set(r, randomBit(ran));

            Couplings[i][k] = coupling_tmp;

            int k_mirror = neighbor_index(j, i);
            Couplings[j][k_mirror] = coupling_tmp;
        }
    }
}

// Overwrite the patterns tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void SpinGlassBits::initCouplingsFromConfig(vector<vector<Bits>> Config) {
    Couplings = Config;
}
// Return a copy of the full coupling tensor
vector<vector<Bits>> SpinGlassBits::getCouplingsConfigBits() {
    return Couplings;
}


// =====================================================
// OBSERVABLES
// =====================================================

// Compute magnetization m = (2*ones - N)/N for each realization 
void SpinGlassBits::GetMagnetizations(vector<double>& magnetizations){
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
void SpinGlassBits::GetSpinConfigurations(vector<vector<uint64_t>>& configs){
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
// Saves magnetizations every save_stride sweeps if save=true.
void SpinGlassBits::runSweeps(gsl_rng* ran, bool save, double freq){
    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    // temporaries allocated once for all sweeps and all flips
    Bits xnor_ij, mask; //used in branch2
    UnsignedInt sum((unsigned long long int)(max_deg * 2)); // max value of sum= neighbor_count[i] * 2 and all spons have same number of neighbors
    UnsignedInt threshold((unsigned long long int)(max_deg * 2)); // no need for more space allocation
    int rng;
    int i;

    const int total_sweeps = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            i = gsl_rng_uniform_int(ran, N);  // pick a random spin
            rng = randomNumber(ran, neighbor_count[i]); //rng does not have to go higher than neighbor_count[i]
            Bits& Bits_Spin_i = Bits_Spins_Set[i];

            // 1ST BRANCH: UNCONDITIONNAL FLIP IN EVERY REPLICA
            if (rng >= neighbor_count[i]) {  
                Bits_Spin_i.ComplementTo();
                continue;
            }

            // 2ND BRANCH: NEIGHBOR-DEPENDENT FLIP
            sum.SetAll(0);
            for (int k = 0; k < neighbors[i].size(); ++k) {
                int j = neighbors[i][k];                  // id of the neighbor
                xnor_ij = (Bits_Spin_i ^ Bits_Spins_Set[j] ^ Couplings[i][k]);
                sum += &xnor_ij;
            }
            sum.MultiplyByTwoTo();  

            threshold.SetAll((unsigned long long int) rng);
            threshold += &Neighbor_Count[i];

            mask = (sum <= threshold);     //at max, sum is equal to 2*neigbohrs_count=8, a higher threshold value is useless
            Bits_Spin_i ^= &mask;
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

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

// Run simulation without saving (thermalization)
void SpinGlassBits::evolve(gsl_rng* ran){
    fromCanonical();
    runSweeps(ran, /*save=*/false, 0.0);
    toCanonical();
}

// Run simulation and save magnetizations at the given frequency
void SpinGlassBits::evolve_save(gsl_rng* ran, double freq, const string& filename){
    fromCanonical();
    OpenSpinFiles();
    cout << "evolve_save called, opening: " << filename << endl;
    runSweeps(ran, /*save=*/true, freq);
    CloseSpinFiles();
    cout << "evolve_save called, closing: " << filename << endl;
    toCanonical();
}


// Run simulation and save at given frequency — fully bitwise initialization
void SpinGlassBits::evolve_save_bits(gsl_rng* ran, double freq, const string& filename){
    initCouplingsBits(ran);
    initSpinsBits(ran);
    cout << "Bitwise initialization done" << endl;

    OpenSpinFiles();
    SaveSpinConfigurations(0);
    cout << "evolve_save_bits called, opening: " << filename << endl;
    runSweeps(ran, /*save=*/true, freq);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save_bits called, closing: " << filename << endl;
}

// Run simulation without saving — fully bitwise initialization
void SpinGlassBits::evolve_bits(gsl_rng* ran, const string& filename){
    initCouplingsBits(ran);
    initSpinsBits(ran);

    OpenSpinFiles();
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_bits called, closing: " << filename << endl;
}