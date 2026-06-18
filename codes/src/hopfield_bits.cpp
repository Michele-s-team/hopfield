//
//  hopfield_bits.cpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//

#include "hopfield_bits.hpp"
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

void HopfieldBits::fromCanonical(){
    Bits_Spins_Set.clear();
    Neighbor_Count.clear();
    P_times_Neighbor_Count.clear();
    Patterns.clear();
    Couplings.clear();

    Bits_Spins_Set.reserve(N);
    Neighbor_Count.reserve(N);
    P_times_Neighbor_Count.reserve(N);
    Couplings.reserve(N);
    Patterns.reserve(P);

    // =====================================================
    // 1. Convert spins
    // =====================================================
    Bits_Spins_Set.clear();
    Bits_Spins_Set.reserve(N);

    Neighbor_Count.clear();
    P_times_Neighbor_Count.clear();
    Neighbor_Count.reserve(N);
    P_times_Neighbor_Count.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        cout << "\r" << i << flush;

        Bits spin_tmp;

        for (int r = 0; r < n_bits; ++r)
        {
            int spin = (spins_set[r * N + i] + 1) / 2;
            spin_tmp.Set(r, spin);
        }

        Bits_Spins_Set.push_back(spin_tmp);

        UnsignedInt neighbor_tmp(neighbor_count[i]);
        UnsignedInt P_times_neighbor_tmp(P * neighbor_count[i]);

        neighbor_tmp.SetAll((unsigned long long)neighbor_count[i]);
        P_times_neighbor_tmp.SetAll((unsigned long long)(P * neighbor_count[i]));

        Neighbor_Count.push_back(neighbor_tmp);
        P_times_Neighbor_Count.push_back(P_times_neighbor_tmp);
    }

    // =====================================================
    // 2. Build D_Coupling and P - D_Coupling (bitwise)
    // =====================================================
    D_Coupling.clear();
    PminusD_Coupling.clear();

    D_Coupling.resize(N);
    PminusD_Coupling.resize(N);

    for (int i = 0; i < N; ++i)
    {
        int deg = neighbors[i].size();

        D_Coupling[i].resize(deg);
        PminusD_Coupling[i].resize(deg);

        for (int k = 0; k < deg; ++k)
        {
            int j = neighbors[i][k];

            UnsignedInt Dij((unsigned long long)0);

            for (int p = 0; p < P; ++p)
            {
                Bits diff = Patterns[p][i] ^ Patterns[p][j];
                Dij += &diff;
            }

            UnsignedInt PD((unsigned long long)P);
            PD.SubstractTo(&Dij, nullptr);

            D_Coupling[i][k] = Dij;
            PminusD_Coupling[i][k] = PD;
        }


    // =====================================================
    // 3. Convert couplings (optional)
    // =====================================================
    /*
    Couplings.clear();
    Couplings.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        Couplings.emplace_back();
        Couplings.back().reserve(couplings[i].size());

        for (int j = 0; j < (int)couplings[i].size(); ++j)
        {
            Bits coupling_tmp;

            for (int r = 0; r < n_bits; ++r)
            {
                int val = (couplings[i][j][r] + 1) / 2;
                coupling_tmp.Set(r, val);
            }

            Couplings.back().push_back(coupling_tmp);
        }
    }
    */
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void HopfieldBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N; ++i)
            spins_set[r*N+i] = -1 + 2 * Bits_Spins_Set[i].Get(r);
}

// =====================================================
// OBSERVABLES
// =====================================================

// Compute magnetization m = (2*ones - N)/N for each realization 
void HopfieldBits::GetMagnetizations(vector<double>& magnetizations){
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
void HopfieldBits::GetSpinConfigurations(vector<vector<uint64_t>>& configs){
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
void HopfieldBits::runSweeps(gsl_rng* ran, bool save, double freq){
    // temporaries allocated once for all sweeps and all flips
    Bits xnor_ij, mask, s_ij, not_s_ij;; //used in branch2
    UnsignedInt sum((unsigned long long int)(2* neighbor_count[0] * P)); // max value of sum = 4* neighbor_count[i] * P and all spons have same number of neighbors 
                                                                         //works only for regular networks, else have to specify max(neighbor_count[i])
    UnsignedInt threshold((unsigned long long int)(2* neighbor_count[0] * P)); // no need for more space allocation
    UnsignedInt contrib((unsigned long long int) P);
    UnsignedInt other((unsigned long long int) P);
    int rng;
    int i;

    const int total_sweeps = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
         cout << "\r" << sweep << flush;

        for (int step = 0; step < N; ++step) {
            i = gsl_rng_uniform_int(ran, N);  // pick a random spin
            rng = randomNumber(ran, neighbor_count[i]*P, N);  //works only for regular networks, else have to specify neighbor_count[i]
            Bits& Bits_Spin_i = Bits_Spins_Set[i];

            // 1ST BRANCH: UNCONDITIONNAL FLIP IN EVERY REPLICA
            if (rng >= P * neighbor_count[i]) {  
                Bits_Spin_i.ComplementTo();
                continue;
            }

            // 2ND BRANCH: NEIGHBOR-DEPENDENT FLIP
            sum.SetAll(0);
            for (int k = 0; k < neighbors[i].size(); ++k) {
                int j = neighbors[i][k];
                s_ij     = Bits_Spin_i ^ Bits_Spins_Set[j];
                not_s_ij = ~s_ij;

                contrib = D_Coupling[i][k];
                contrib &= &s_ij;          // keep lanes where spins differ

                other = PminusD_Coupling[i][k];
                other &= &not_s_ij;        // keep lanes where spins agree

                contrib += &other;
                sum += &contrib;
            }
            sum.MultiplyByTwoTo(); 

            threshold.SetAll((unsigned long long int) rng);
            threshold += &P_times_Neighbor_Count[i];

            mask = (sum <= threshold);   
            Bits_Spin_i ^= &mask;
        }

        if (save && sweep > 0 && sweep < total_sweeps && (sweep % save_stride == 0))
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
void HopfieldBits::evolve(gsl_rng* ran, const string& filename){
    fromCanonical();
    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve called, closing: " << filename << endl;
    toCanonical();
    toCanonical();
}

// Run simulation and save magnetizations at the given frequency
void HopfieldBits::evolve_save(gsl_rng* ran, double freq, const string& filename){
    fromCanonical();
    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    cout << "evolve_save called, opening: " << filename << endl;
    runSweeps(ran, /*save=*/true, freq);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save called, closing: " << filename << endl;
    toCanonical();
}