//
//  hopfield_bits.cpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//

#include "hopfield_bits.hpp"
#include "unsigned_int.hpp"
#include <numeric>
#include <algorithm>
#include <filesystem>
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"

using namespace std;


// =====================================================
// STATE CONVERSIONS WHEN WORKING FROM CANONICAL FORMAT
// =====================================================

void HopfieldBits::fromCanonical() {
    // =====================================================
    // 1. Convert spins
    // =====================================================
    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    Bits_Spins_Set.clear();
    Bits_Spins_Set.reserve(N);
    Neighbor_Count.clear();
    Neighbor_Count.reserve(N);
    P_times_Neighbor_Count.clear();
    P_times_Neighbor_Count.reserve(N);

    for (int i = 0; i < N; ++i) {
        Bits spin_tmp;
        for (int r = 0; r < n_bits; ++r) {
            int spin = (spins_set[r * N + i] + 1) / 2;
            spin_tmp.Set(r, spin);
        }
        Bits_Spins_Set.push_back(spin_tmp);

        UnsignedInt nc_tmp((unsigned long long)max_deg);
        nc_tmp.SetAll((unsigned long long)neighbor_count[i]);
        Neighbor_Count.push_back(nc_tmp);

        UnsignedInt pnk_tmp((unsigned long long)(P * max_deg));
        pnk_tmp.SetAll((unsigned long long)(P * neighbor_count[i]));
        P_times_Neighbor_Count.push_back(pnk_tmp);
    }

    // =====================================================
    // 2. Convert patterns
    // =====================================================
    Patterns.clear();
    Patterns.resize(P);
    for (int p = 0; p < P; ++p) {
        Patterns[p].resize(N);
        for (int i = 0; i < N; ++i) {
            Bits pat_tmp;
            for (int r = 0; r < n_bits; ++r) {
                pat_tmp.Set(r, (patterns[p][i][r] + 1) / 2);
            }
            Patterns[p][i] = pat_tmp;
        }
    }

    // =====================================================
    // 3. Convert couplings
    // =====================================================
    Couplings.clear();
    Couplings.resize(N);
    for (int i = 0; i < N; ++i) {
        Couplings[i].clear();
        Couplings[i].reserve(couplings[i].size());
        for (int j = 0; j < (int)couplings[i].size(); ++j) {
            UnsignedInt coupling_tmp((unsigned long long)2 * P);
            for (int r = 0; r < n_bits; ++r) {
                int val = couplings[i][j][r] + P;
                coupling_tmp.Set(r, val);
            }
            Couplings[i].push_back(coupling_tmp);
        }
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void HopfieldBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N; ++i)
            spins_set[r*N+i] = -1 + 2 * Bits_Spins_Set[i].Get(r);
}

// =====================================================
// DIRECT BITWISE INITIALIZATION
// =====================================================


// Initialize spins directly in bit-sliced representation.
// Replaces: initSpins() + fromCanonical() step 1.
void HopfieldBits::initSpinsBits(gsl_rng* ran) {
    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    Bits_Spins_Set.clear();
    Bits_Spins_Set.resize(N);
    Neighbor_Count.clear();
    Neighbor_Count.reserve(N);
    P_times_Neighbor_Count.clear();
    P_times_Neighbor_Count.reserve(N);

    for (int i = 0; i < N; ++i) {
        for (int r = 0; r < n_bits; ++r)
            Bits_Spins_Set[i].Set(r, randomBit(ran));

        UnsignedInt nc_tmp((unsigned long long) max_deg);
        nc_tmp.SetAll((unsigned long long) neighbor_count[i]);
        Neighbor_Count.push_back(nc_tmp);

        UnsignedInt pnk_tmp((unsigned long long)(P * max_deg));
        pnk_tmp.SetAll((unsigned long long)(P * neighbor_count[i]));
        P_times_Neighbor_Count.push_back(pnk_tmp);
    }
}


// Initialize Patterns directly in bit-sliced representation.
// Replaces: initPatterns() + fromCanonical() step 2.
// patterns[p][i] is a Bits word: bit r = (xi^p_i^r + 1) / 2 in {0,1}
void HopfieldBits::initPatternsBits(gsl_rng* ran) {
    Patterns.clear();
    Patterns.resize(P);
    for (int p = 0; p < P; ++p) {
        Patterns[p].resize(N);
        for (int i = 0; i < N; ++i) {
            for (int r = 0; r < n_bits; ++r)
                Patterns[p][i].Set(r, (randomSpin(ran) + 1) / 2);
        }
    }
}

// Initialize Couplings directly in bit-sliced representation via Hebb rule,
// without ever building the scalar couplings[][][] tensor.
// g_ij^r = P + sum_mu xi_i^mu^r * xi_j^mu^r  in [0, 2P]
// Replaces: initCouplings() + fromCanonical() step 3.
void HopfieldBits::initCouplingsBits() {
    Couplings.clear();
    Couplings.resize(N);
    for (int i = 0; i < N; ++i) {
        Couplings[i].clear();
        Couplings[i].reserve(neighbors[i].size());
        for (int k = 0; k < (int)neighbors[i].size(); ++k)
            Couplings[i].emplace_back((unsigned long long) 2 * P);
    }

    // Hebb rule: compute once for i < j, then mirror
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < (int)neighbors[i].size(); ++k) {
            int j = neighbors[i][k];
            if (j <= i) continue;

            // Accumulate G_ij^r = sum_mu xi_i^mu^r * xi_j^mu^r
            // xi_i^mu^r in {0,1} as Bits -> product = XNOR = ~(a^b)
            // sum_mu (2*bit-1)(2*bit-1) = sum_mu (1 - 2*(a^b))
            //                           = P - 2 * popcount(a^b) per replica
            // g_ij^r = P + G_ij^r = 2P - 2*popcount(XOR)
            // But here we accumulate directly in UnsignedInt bit-sliced:
            // g_ij starts at P (SetAll), then += XNOR for each pattern

            Couplings[i][k].SetAll((unsigned long long) P);

            for (int mu = 0; mu < P; ++mu) {
                Bits prod = ~(Patterns[mu][i] ^ Patterns[mu][j]);  // XNOR = [xi_i == xi_j]
                // prod bit r = 1 if xi_i^mu^r == xi_j^mu^r -> contributes +1
                // prod bit r = 0                            -> contributes -1
                // net: G_ij += 2*prod - 1  per replica
                // i.e. g_ij += prod (the -1+P offset is already in SetAll(P))
                // Actually: xi*xi = (2b-1)(2b-1) = 1 - 2*(b XOR b') 
                //           sum_mu xi*xi = P - 2*sum_mu XOR
                // so g_ij = P + sum_mu xi*xi = 2P - 2*sum_mu XOR
                // equivalently: g_ij = P + sum_mu (2*XNOR - 1)
                //                    = P - P + 2*sum_mu XNOR = 2*sum_mu XNOR
                // --> reset to 0 and accumulate XNOR twice
                Couplings[i][k] += &prod;
            }
            // At this point sum = sum_mu XNOR in [0,P]
            // g_ij = 2*sum_mu XNOR, so multiply by 2
            // But we want g_ij = P + G_ij = P + sum_mu xi*xi
            //                  = P + (P - 2*(P - sum_mu XNOR))
            //                  = 2*sum_mu XNOR  ✓
            Couplings[i][k].MultiplyByTwoTo();

            // Mirror onto j
            int k_mirror = neighbor_index(j, i);
            Couplings[j][k_mirror] = Couplings[i][k];
        }
    }
}

// Overwrite the patterns tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void HopfieldBits::initPatternsFromConfig(vector<vector<Bits>> Config) {
    Patterns = Config;
    initCouplingsBits();
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
// Convert Patterns (bit-sliced) back to canonical scalar tensor patterns[p][i][r] in {-1,+1}
vector<vector<vector<int>>> HopfieldBits::getPatternsBits() {
    vector<vector<vector<int>>> result(P, vector<vector<int>>(N, vector<int>(n_bits, 0)));
    for (int p = 0; p < P; ++p)
        for (int i = 0; i < N; ++i)
            for (int r = 0; r < n_bits; ++r)
                result[p][i][r] = -1 + 2 * Patterns[p][i].Get(r);
    return result;
}

// Convert Couplings (bit-sliced) back to canonical scalar tensor couplings[i][k][r] in [-P,+P]
vector<vector<vector<int>>> HopfieldBits::getCouplingsConfigBits() {
    vector<vector<vector<int>>> result(N);
    for (int i = 0; i < N; ++i) {
        result[i].resize(Couplings[i].size(), vector<int>(n_bits, 0));
        for (int k = 0; k < (int)Couplings[i].size(); ++k)
            for (int r = 0; r < n_bits; ++r)
                result[i][k][r] = (int)Couplings[i][k].Get(r) - P;  // g_ij = P + G_ij -> G_ij
    }
    return result;
}
// =====================================================
// METROPOLIS DYNAMICS
// =====================================================
//
// Runs Monte Carlo sweeps on 64 parallel Hopfield network instances using a
// bitwise Metropolis algorithm. All 64 replicas are updated simultaneously
// via bit-sliced arithmetic on UnsignedInt objects.
//
// Flip condition (derived from Metropolis accept/reject):
//
//   LHS >= RHS
//
// where:
//   LHS = random + sum_j g_ij + 2P * sum_j c_ij
//   RHS = P * deg_i + 2 * sum_j c_ij * g_ij
//
// with:
//   g_ij = P + G_ij  in [0, 2P],  G_ij = sum_mu xi_i^mu xi_j^mu  (precomputed)
//   c_ij = [S_i == S_j]           (1 if same spin, 0 otherwise)
//   random in [0, P * deg_i)      (scaled log-uniform random number)
//
// All UnsignedInt accumulators are preallocated before the sweep loop with
// their maximum possible values to avoid any heap allocation in the hot path.
void HopfieldBits::runSweeps(gsl_rng* ran, bool save, double freq){
    Bits c_ij, mask;

    const int total_sweeps    = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride     = save ? max(1, (int)round(1.0 / freq)) : 0;

    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    UnsignedInt sum_g          ((unsigned long long) 2 * max_deg * P);
    UnsignedInt sum_c          ((unsigned long long) max_deg);
    UnsignedInt sum_cg         ((unsigned long long) 2 * max_deg * P);
    UnsignedInt sum_c_times_2P ((unsigned long long) 2 * max_deg * P);
    UnsignedInt LHS            ((unsigned long long) 5 * max_deg * P);
    UnsignedInt RHS            ((unsigned long long) 5 * max_deg * P);
    UnsignedInt tmp            ((unsigned long long) 2 * P);

    for (int sweep = 0; sweep < total_sweeps; ++sweep){
        for (int step = 0; step < N; ++step){

            int i      = gsl_rng_uniform_int(ran, N);
            int deg_i  = neighbor_count[i];
            int random = randomNumber(ran, deg_i * P, N);

            Bits& S_i = Bits_Spins_Set[i];

            if (random >= P * deg_i) {
                S_i.ComplementTo();
                continue;
            }

            sum_g .SetAll(0);
            sum_c .SetAll(0);
            sum_cg.SetAll(0);

            for (int k = 0; k < deg_i; ++k){
                int j = neighbors[i][k];
                UnsignedInt& g_ij = Couplings[i][k];

                c_ij = ~(S_i ^ Bits_Spins_Set[j]);

                sum_g += &g_ij;
                sum_c += &c_ij;

                tmp  = g_ij;
                tmp &= &c_ij;
                sum_cg += &tmp;
            }

            // RHS = P*deg_i + 2*sum_cg
            RHS  = sum_cg;
            RHS.MultiplyByTwoTo();
            RHS += &P_times_Neighbor_Count[i];

            // LHS = random + sum_g + 2P*sum_c
            LHS  = sum_g;
            LHS.AddScalar(random);
            sum_c.MultiplyByInteger(2 * P, &sum_c_times_2P);
            LHS += &sum_c_times_2P;

            mask = (RHS <= LHS);
            S_i ^= &mask;
        }

        if (save && sweep > 0 && (sweep % save_stride == 0))
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

// Run simulation strarting from the canonical values without saving (thermalization)
void HopfieldBits::evolve(gsl_rng* ran, const string& filename){
    fromCanonical();
    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve called, closing: " << filename << endl;
    toCanonical();
}

// Run simulation and save magnetizations at the given frequency
void HopfieldBits::evolve_save(gsl_rng* ran, double freq, const string& filename){
    fromCanonical();
    cout << "Bitwise conversion done"<<endl;
    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    cout << "evolve_save called, opening: " << filename << endl;
    runSweeps(ran, /*save=*/true, freq);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save called, closing: " << filename << endl;
    toCanonical();
}


// Run simulation and save at given frequency — fully bitwise initialization
void HopfieldBits::evolve_save_bits(gsl_rng* ran, double freq, const string& filename){
    initPatternsBits(ran);
    initCouplingsBits();
    initSpinsBits(ran);
    cout << "Bitwise initialization done" << endl;

    vector<vector<vector<int>>> patterns = getPatternsBits();
    SavePatterns("patterns/", patterns);
    cout << "Patterns saved" << endl;

    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    cout << "evolve_save_bits called, opening: " << filename << endl;
    runSweeps(ran, /*save=*/true, freq);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save_bits called, closing: " << filename << endl;
}

// Run simulation without saving — fully bitwise initialization
void HopfieldBits::evolve_bits(gsl_rng* ran, const string& filename){
    initPatternsBits(ran);
    initCouplingsBits();
    initSpinsBits(ran);

    vector<vector<vector<int>>> patterns = getPatternsBits();
    SavePatterns("patterns/", patterns);
    cout << "Patterns saved" << endl;

    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_bits called, closing: " << filename << endl;
}