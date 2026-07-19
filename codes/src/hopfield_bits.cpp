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

#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"
#include <chrono>
using clk = std::chrono::high_resolution_clock;

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

    for (int spin = 0; spin < N; ++spin) {
        Bits spin_tmp;
        for (int r = 0; r < n_bits; ++r) {
            int spin = (spins_set[r * N + spin] + 1) / 2;
            spin_tmp.Set(r, spin);
        }
        Bits_Spins_Set.push_back(spin_tmp);

        UnsignedInt nc_tmp((unsigned long long)max_deg);
        nc_tmp.SetAll((unsigned long long)neighbor_count[spin]);
        Neighbor_Count.push_back(nc_tmp);

        UnsignedInt pnk_tmp((unsigned long long)(P * max_deg));
        pnk_tmp.SetAll((unsigned long long)(P * neighbor_count[spin]));
        P_times_Neighbor_Count.push_back(pnk_tmp);
    }

    // =====================================================
    // 2. Convert patterns
    // =====================================================
    Patterns.clear();
    Patterns.resize(P);
    for (int mu = 0; mu < P; ++mu) {
        Patterns[mu].resize(N);
        for (int spin = 0; spin < N; ++spin) {
            Bits pat_tmp;
            for (int r = 0; r < n_bits; ++r) {
                pat_tmp.Set(r, (patterns[mu][spin][r] + 1) / 2);
            }
            Patterns[mu][spin] = pat_tmp;
        }
    }

    // =====================================================
    // 3. Convert couplings
    // =====================================================
    Couplings.clear();
    Couplings.resize(N);
    for (int spin = 0; spin < N; ++spin) {
        Couplings[spin].clear();
        Couplings[spin].reserve(couplings[spin].size());
        for (int j = 0; j < (int)couplings[spin].size(); ++j) {
            UnsignedInt coupling_tmp((unsigned long long)2 * P);
            for (int r = 0; r < n_bits; ++r) {
                int val = couplings[spin][j][r] + P;
                coupling_tmp.Set(r, val);
            }
            Couplings[spin].push_back(coupling_tmp);
        }
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void HopfieldBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int spin = 0; spin < N; ++spin)
            spins_set[r*N+spin] = -1 + 2 * Bits_Spins_Set[spin].Get(r);
}

// =====================================================
// DIRECT BITWISE INITIALIZATION
// =====================================================


// Initialize spins directly in bit-sliced representation.
// Replaces: initSpins() + fromCanonical() step 1.
void HopfieldBits::initSpinsBits(gsl_rng* ran) {
    Bits_Spins_Set.clear();
    Bits_Spins_Set.resize(N);

    for (int spin = 0; spin < N; ++spin)
        for (int r = 0; r < n_bits; ++r)
            Bits_Spins_Set[spin].Set(r, randomBit(ran));

    initSpinMetadataBits();
}


void HopfieldBits::initSpinMetadataBits() {
    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    Neighbor_Count.clear();
    Neighbor_Count.reserve(N);

    P_times_Neighbor_Count.clear();
    P_times_Neighbor_Count.reserve(N);

    for (int spin = 0; spin < N; ++spin) {
        UnsignedInt nc_tmp((unsigned long long)max_deg);
        nc_tmp.SetAll((unsigned long long)neighbor_count[spin]);
        Neighbor_Count.push_back(nc_tmp);

        UnsignedInt pnk_tmp((unsigned long long)(P * max_deg));
        pnk_tmp.SetAll((unsigned long long)(P * neighbor_count[spin]));
        P_times_Neighbor_Count.push_back(pnk_tmp);
    }
}

// Initialize Patterns directly in bit-sliced representation.
// Replaces: initPatterns() + fromCanonical() step 2.
// patterns[mu][spin] is a Bits word: bit r = (xi^p_i^r + 1) / 2 in {0,1}
void HopfieldBits::initPatternsBits(gsl_rng* ran) {
    Patterns.clear();
    Patterns.resize(P);
    for (int mu = 0; mu < P; ++mu) {
        Patterns[mu].resize(N);
        for (int spin = 0; spin < N; ++spin) {
            for (int r = 0; r < n_bits; ++r)
                Patterns[mu][spin].Set(r, randomBit(ran));
        }
    }
    initCouplingsBits();
}


// Corrupt a single pattern by flipping each bit independently with
// probability flip_fraction. Input is one column of Patterns (spin.e.
// Patterns[mu]), a vector<Bits> of size N, where each Bits word packs
// n_bits independent replicas.
//
// By default each replica r gets its own independent random flip mask
// (spin.e. flip_fraction is the *expected* fraction of flipped spins per
// replica, but which spins get flipped differs from one replica to the
// next). This lets you run n_bits independent noise realizations for
// the same flip_fraction in a single bitwise pass.
//
// If you instead want all replicas to share the exact same corrupted
// pattern (same flipped sites for every replica), draw the flip mask
// once per spin spin (outside the r loop) and apply it to all r — see the
// commented alternative below.

vector<Bits> HopfieldBits::corruptPattern(const vector<Bits>& pattern,
                                           double flip_fraction,
                                           gsl_rng* ran) const {

    vector<Bits> corrupted = pattern; // copy, size N

    for (int spin = 0; spin < N; ++spin) {

        // --- Option 1 (default): independent noise per replica ---
        for (int r = 0; r < n_bits; ++r) {
            if (gsl_rng_uniform(ran) < flip_fraction) {
                int bit = corrupted[spin].Get(r);
                corrupted[spin].Set(r, 1 - bit);
            }
        }

        // --- Option 2: same flip mask shared across all replicas ---
        // if (gsl_rng_uniform(ran) < flip_fraction) {
        //     for (int r = 0; r < n_bits; ++r) {
        //         int bit = corrupted[spin].Get(r);
        //         corrupted[spin].Set(r, 1 - bit);
        //     }
        // }
    }

    return corrupted;
}


// Initialize Couplings directly in bit-sliced representation via Hebb rule,
// without ever building the scalar couplings[][][] tensor.
// g_ij^r = P + sum_mu xi_i^mu^r * xi_j^mu^r  in [0, 2P]
// Replaces: initCouplings() + fromCanonical() step 3.
void HopfieldBits::initCouplingsBits() {
    Couplings.clear();
    Couplings.resize(N);
    for (int spin = 0; spin < N; ++spin) {
        Couplings[spin].clear();
        Couplings[spin].reserve(neighbors[spin].size());
        for (int k = 0; k < (int)neighbors[spin].size(); ++k)
            Couplings[spin].emplace_back((unsigned long long) 2 * P);
    }

    // Hebb rule: compute once for spin < j, then mirror
    for (int spin = 0; spin < N; ++spin) {
        for (int k = 0; k < (int)neighbors[spin].size(); ++k) {
            int j = neighbors[spin][k];
            if (j <= spin) continue;

            Couplings[spin][k].SetAll((unsigned long long) 0);

            for (int mu = 0; mu < P; ++mu) {
                Bits prod = ~(Patterns[mu][spin] ^ Patterns[mu][j]);
                Couplings[spin][k] += &prod;
            }
            Couplings[spin][k].MultiplyByTwoTo();

            // Mirror onto j
            int k_mirror = neighbor_index(j, spin);
            Couplings[j][k_mirror] = Couplings[spin][k];
            
        }
    }
}

// Overwrite the patterns tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void HopfieldBits::initPatternsFromConfigBits(const vector<vector<Bits>>& Config) {
    Patterns = Config;
    initCouplingsBits();
}

void HopfieldBits::initSpinsFromConfigBits(const vector<Bits>& Config) {
    Bits_Spins_Set = Config;
    initSpinMetadataBits();
}

// Return a copy of the full pattern tensor
 vector<vector<Bits>> HopfieldBits::getPatternsBits() {
    return Patterns;
}

// Return a copy of the full coupling tensor
vector<vector<UnsignedInt>> HopfieldBits::getCouplingsConfigBits() {
    return Couplings;
}

// =====================================================
// OBSERVABLES
// =====================================================

// Compute magnetization m = (2*ones - N)/N for each realization 
void HopfieldBits::GetMagnetizations(vector<double>& magnetizations){
    magnetizations.resize(n_bits);
    vector<int> ones(n_bits, 0);

    for (int spin = 0; spin < N; ++spin)
        for (int r = 0; r < n_bits; ++r)
            ones[r] += Bits_Spins_Set[spin].Get(r);

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

        for (int spin = start; spin < end; ++spin){
            for (int r = 0; r < n_bits; ++r){
                configs[r][b] <<= 1;
                if (Bits_Spins_Set[spin].Get(r))
                    configs[r][b] |= 1ULL;
            }
        }
    }
}
// Convert Patterns (bit-sliced) back to canonical scalar tensor patterns[mu][spin][r] in {-1,+1}
vector<vector<vector<int>>> HopfieldBits::getPatternsBitsToCanonical() {
    vector<vector<vector<int>>> result(P, vector<vector<int>>(N, vector<int>(n_bits, 0)));
    for (int mu = 0; mu < P; ++mu)
        for (int spin = 0; spin < N; ++spin)
            for (int r = 0; r < n_bits; ++r)
                result[mu][spin][r] = -1 + 2 * Patterns[mu][spin].Get(r);
    return result;
}

void HopfieldBits::compute_shifted_overalps(){
    UnsignedInt sum(2*N);
    Bits c_ij, carry;
    for (int mu=0; mu < P; mu++){
        sum.SetAll(0);
        c_ij.Set(0);
        for (int spin = 0; spin < N; spin++){
            carry.Set(0);
            c_ij = ~(Patterns[mu][spin]^ Bits_Spins_Set[spin]);
            sum.AddTo(&c_ij, &carry);
      
        }
        sum.MultiplyByTwoTo();
        Shifted_Overlaps.push_back(sum);
    }
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
//   RHS = P * deg_spin + 2 * sum_j c_ij * g_ij
//
// with:
//   g_ij = P + G_ij  in [0, 2P],  G_ij = sum_mu xi_i^mu xi_j^mu  (precomputed)
//   c_ij = [S_i == S_j]           (1 if same spin, 0 otherwise)
//   random in [0, P * deg_spin)      (scaled log-uniform random number)
//
// All UnsignedInt accumulators are preallocated before the sweep loop with
// their maximum possible values to avoid any heap allocation in the hot path.

void HopfieldBits::runSweeps(gsl_rng* ran, bool save, double freq, int shift){

    const int total_sweeps    = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride     = save ? max(1, (int)round(1.0 / freq)) : 0;

    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    Bits c_ij, mask, carry;

    UnsignedInt sum_c          ((unsigned long long) max_deg);
    UnsignedInt sum_cg         ((unsigned long long) 2 * max_deg * P);
    UnsignedInt sum_c_times_2P ((unsigned long long) 2 * max_deg * P);
    UnsignedInt LHS            ((unsigned long long) 5 * max_deg * P);
    UnsignedInt RHS            ((unsigned long long) 5 * max_deg * P);
    UnsignedInt tmp          ((unsigned long long) max_deg * P);

    Bits         c_terms[MaxDeg];
    UnsignedInt  cg_terms[MaxDeg];
    
    const unsigned long long twoP = 2 * P;

    // Buffers scratch pour CSAReduceBits, alloués une seule fois
    Bits bucketsScratch[MaxWidth][MaxDeg];
    Bits resultBitScratch[MaxWidth];

     // Pool CSA alloué une seule fois, réutilisé pour tous les sweeps / tous les spins
    UnsignedInt csaScratchSum[MaxDeg];
    UnsignedInt csaScratchCarry[MaxDeg];
    UnsignedInt csaBuf[MaxDeg];
    for (int i = 0; i < MaxDeg; ++i) {
        csaScratchSum[i]   = UnsignedInt(2 * max_deg * P);
        csaScratchCarry[i] = UnsignedInt(2 * max_deg * P);
        csaBuf[i]          = UnsignedInt(2 * max_deg * P);
    }

    // Precompute Σ_j g_ij once
    vector<UnsignedInt> sum_g_persist(N);
    for (int spin = 0; spin < N; ++spin) {
        sum_g_persist[spin] = UnsignedInt((unsigned long long)2 * max_deg * P);
        sum_g_persist[spin].SetAll(0);

        for (int k = 0; k < neighbor_count[spin]; ++k)
            sum_g_persist[spin] += &Couplings[spin][k];
    }

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {

        for (int step = 0; step < N; ++step) {

            int spin      = gsl_rng_uniform_int(ran, N);
            int deg_spin  = neighbor_count[spin];
            int random = randomNumber(ran, deg_spin * P, N);

            Bits& S_i = Bits_Spins_Set[spin];

            if (random >= P * deg_spin) {
                S_i.ComplementTo();
                continue;
            }

            sum_c.SetAll(0);
            sum_cg.SetAll(0);
            LHS.SetAll(0);
            RHS.SetAll(0);
            sum_c_times_2P.SetAll(0);

            /*

            for (int k = 0; k < deg_spin; ++k) {
                int j = neighbors[spin][k];

                // Agreement mask between spin i and neighbor j
                c_terms[k] = ~(S_i ^ Bits_Spins_Set[j]);

                // Broadcast-AND: apply the 1-bit mask across every bit-plane of the coupling
                cg_terms[k] = Couplings[spin][k];
                cg_terms[k].AndTo(&c_terms[k], 0, cg_terms[k].GetSize());
            }

            sum_c  = UnsignedInt::CSAReduceBits<MaxDeg, MaxWidth>(c_terms, deg_spin, (unsigned long long) max_deg, bucketsScratch, resultBitScratch);
            sum_cg = UnsignedInt::CSAReduceUnsignedInt<MaxDeg>(cg_terms, deg_spin, csaBuf, csaScratchSum, csaScratchCarry);
            */
            //old loop

            for (int k = 0; k < deg_spin; ++k) {
                carry.Set(0);

                int j = neighbors[spin][k];

                c_ij = ~(S_i ^ Bits_Spins_Set[j]);

                sum_c.AddTo(&c_ij, &carry); //assumes that the carry won't overflow the size of "sum_c"
                sum_cg.AddAnd(&Couplings[spin][k], &c_ij);
            }
        

            // RHS = 2*sum_cg + P*deg_spin
            RHS.CopyValues(sum_cg);
            RHS.MultiplyByTwoTo();
            RHS += &P_times_Neighbor_Count[spin];

            // 2P*sum_c
            sum_c.MultiplyByConstant(twoP, &sum_c_times_2P);

            // LHS = random + sum_g + 2P*sum_c
            LHS.CopyValues(sum_g_persist[spin]);
            LHS.AddScalar(random, tmp);
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

    cout << endl; 
}

// Recalcule Shifted_Overlaps à partir de l'état courant des spins,
// et compare avec les valeurs maintenues incrémentalement.
// Retourne true si tout est cohérent, false au premier écart trouvé (et l'affiche).
bool HopfieldBits::check_shifted_overlaps_consistency(int sweep, int step) {

    bool ok = true;
    Bits eq_mask;

    for (int mu = 0; mu < P; mu++) {

        UnsignedInt fresh(2 * N);
        Bits c_ij, carry;

        fresh.SetAll(0);

        for (int spin = 0; spin < N; spin++) {
            carry.Set(0);
            c_ij = ~(Patterns[mu][spin] ^ Bits_Spins_Set[spin]);
            fresh.AddTo(&c_ij, &carry);
        }
        fresh.MultiplyByTwoTo();

        eq_mask = fresh == Shifted_Overlaps[mu];

        if (!eq_mask.equal(Bits_one)) {   // tous les 64 bits ne sont pas à 1 -> au moins un replica diverge

            unsigned long long mismatches = ~eq_mask.Get();  // bit à 1 = replica en désaccord

            cout << "[MISMATCH] sweep=" << sweep
                 << " step=" << step
                 << " mu=" << mu
                 << "  replicas en desaccord (masque)=0x" << hex << mismatches << dec
                 << endl;

            ok = false;
        }
    }

    return ok;
}

void HopfieldBits::runSweeps_overlaps(gsl_rng* ran, bool save, double freq, int shift){

    compute_shifted_overalps();
    cout << "Shifted overlaps computed"<< endl;
    const int total_sweeps    = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride     = save ? max(1, (int)round(1.0 / freq)) : 0;

    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    Bits c_i_mu, mask;
    Bits carry_g, borrow_g;   // carry/borrow dédiés à Shifted_Overlaps[mu]
    Bits carry_L, borrow_L;   // carry/borrow dédiés à sum_L
    Bits carry_c;

    Bits carry_delta;
    Bits borrow_delta;

    UnsignedInt sum_c          ((unsigned long long) P);
    UnsignedInt sum_cl         ((unsigned long long) 2 * N* P);
    UnsignedInt sum_c_times_2N ((unsigned long long) 2 * N * P);
    UnsignedInt LHS            ((unsigned long long) P * (5*N - 1));
    UnsignedInt RHS            ((unsigned long long) P * (5*N - 1));
    UnsignedInt tmp            ((unsigned long long) P * (N-1));
    UnsignedInt fourSum        ((unsigned long long) 4 * P);
    UnsignedInt TwoMask        ((unsigned long long) 2);
    UnsignedInt Sub            ((unsigned long long) 4);

    UnsignedInt CST ((unsigned long long) (N-1)*P);
    CST.SetAll((N-1)*P);

    UnsignedInt Two ((unsigned long long) 2);
    Two.SetAll(2);

    UnsignedInt TwoP  ((unsigned long long) 2*P);
    TwoP.SetAll(2*P);

    UnsignedInt delta((unsigned long long)2 * P);
    delta.SetAll(2 * P);

    UnsignedInt sum_L ((unsigned long long) 2*N*P);
    sum_L.SetAll(0);

    const unsigned long long twoN = 2 * N;

    vector<Bits> c_cache(P);

    cout <<"initialization done"<<endl;

    // Precompute Σ_mu L^mu
    for (int mu = 0; mu < P; ++mu) {
        sum_L+= &Shifted_Overlaps[mu];
    }
    
    for (int sweep = 0; sweep < total_sweeps; ++sweep) {

        for (int step = 0; step < N; ++step) {

            //cout << "    " << step << endl;

            int spin      = gsl_rng_uniform_int(ran, N);
            int deg_spin  = neighbor_count[spin];
            int random = randomNumber(ran, deg_spin * P, N);

            Bits& S_i = Bits_Spins_Set[spin];

            if (random >= P * deg_spin) {
                cout << "unconditionnal flip"<<endl;
                for (int mu = 0; mu < P; mu++) {
                    carry_g.Set(0);
                    carry_L.Set(0);
                    borrow_g.Set(0);
                    borrow_L.Set(0);

                    c_i_mu = ~(S_i ^ Patterns[mu][spin]);

                    Shifted_Overlaps[mu].AddTo(&Two, &carry_g);
                    sum_L.AddTo(&Two, &carry_L);

                    Sub.SetAll(0);
                    Sub.CopyValues(c_i_mu);
                    Sub.MultiplyByPowerOfTwo(2); 

                    Shifted_Overlaps[mu].SubtractTo(&Sub, &borrow_g);
                    sum_L.SubtractTo(&Sub, &borrow_L);
                }
                S_i.ComplementTo();
                continue;
            }

            sum_c.SetAll(0);
            sum_cl.SetAll(0);
            LHS.SetAll(0);
            RHS.SetAll(0);
            sum_c_times_2N.SetAll(0);

            for (int mu = 0; mu < P; ++mu) {
                //cout << "         " << mu << endl;
                carry_c.Set(0);

                c_cache[mu] = ~(S_i ^ Patterns[mu][spin]);

                sum_c.AddTo(&c_cache[mu], &carry_c);
                sum_cl.AddAnd(&Shifted_Overlaps[mu], &c_cache[mu]);
            }
        
            // RHS = 2*sum_cg + P*deg_spin
            RHS.CopyValues(sum_cl);
            RHS.MultiplyByTwoTo();
            RHS += &CST;

            // LHS = random + sum_g + 2P*sum_c
            sum_c.MultiplyByConstant(twoN, &sum_c_times_2N);
            LHS.CopyValues(sum_c_times_2N);
            LHS.AddScalar(random, tmp);
            LHS += &sum_L;

            mask = (RHS <= LHS);

            TwoMask.SetAll(0);
            TwoMask.CopyValues(mask);
            TwoMask.MultiplyByTwoTo();

            for (int mu = 0; mu < P; mu++){
                carry_g.Set(0);
                carry_L.Set(0);
                borrow_g.Set(0);
                borrow_L.Set(0);

                c_cache[mu] &= mask;

                Shifted_Overlaps[mu].AddTo(&TwoMask, &carry_g);
                Shifted_Overlaps[mu].SubtractShifted(&c_cache[mu], 2, &borrow_g);
            }

            sum_L.AddMasked(&TwoP, &mask);

            /*
            TwoMask.SetAll(0);
            TwoMask.CopyValues(mask);
            TwoMask.MultiplyByTwoTo();

            for(int i=0;i<P;i++)
                sum_L += &TwoMask;

            */
            sum_c &= &mask;

            fourSum.CopyValues(sum_c);
            fourSum.MultiplyByPowerOfTwo(2);

            borrow_delta.Set(0);
            sum_L.SubtractTo(&fourSum, &borrow_delta); 

            S_i ^= &mask;
        }

        if (save && sweep > 0 && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                 << flush;
    }

    cout << endl; 
    cout <<" bitwise done"<< endl;
}

void HopfieldBits::runSweeps_overlaps_old(gsl_rng* ran, bool save, double freq, int shift){

    compute_shifted_overalps();
    cout << "Shifted overlaps computed" << endl;

    const int total_sweeps = getNSweeps();
    const int progress_stride = max(1,total_sweeps/10);
    const int save_stride = save ? max(1,(int)round(1.0/freq)) : 0;

    int max_deg = *max_element(neighbor_count.begin(),neighbor_count.end());

    Bits c_i_mu, mask;
    Bits carry_g, borrow_g;
    Bits carry_L, borrow_L;
    Bits carry_c;

    UnsignedInt sum_c((unsigned long long)P);
    UnsignedInt sum_cl((unsigned long long)2*N*P);
    UnsignedInt sum_c_times_2N((unsigned long long)2*N*P);
    UnsignedInt LHS((unsigned long long)P*(5*N-1));
    UnsignedInt RHS((unsigned long long)P*(5*N-1));
    UnsignedInt tmp((unsigned long long)max_deg*P);

    UnsignedInt CST((unsigned long long)(N-1)*P);
    CST.SetAll((N-1)*P);

    UnsignedInt Two((unsigned long long)2);
    Two.SetAll(2);

    UnsignedInt TwoMask((unsigned long long)2);

    UnsignedInt Sub((unsigned long long)4);

    const unsigned long long twoN = 2*N;

    cout << "initialization done" << endl;

    UnsignedInt sum_L((unsigned long long)2*N*P);
    sum_L.SetAll(0);

    for(int mu=0;mu<P;mu++)
        sum_L += &Shifted_Overlaps[mu];


    for(int sweep=0;sweep<total_sweeps;sweep++){

        for(int step=0;step<N;step++){

            int spin = gsl_rng_uniform_int(ran,N);
            int deg_spin = neighbor_count[spin];
            int random = randomNumber(ran,deg_spin*P,N);

            Bits& S_i = Bits_Spins_Set[spin];


            if(random >= P*deg_spin){

                for(int mu=0;mu<P;mu++){

                    carry_g.Set(0);
                    carry_L.Set(0);
                    borrow_g.Set(0);
                    borrow_L.Set(0);

                    c_i_mu = ~(S_i ^ Patterns[mu][spin]);

                    Shifted_Overlaps[mu].AddTo(&Two,&carry_g);
                    sum_L.AddTo(&Two,&carry_L);

                    Sub.SetAll(0);
                    Sub.CopyValues(c_i_mu);
                    Sub.MultiplyByPowerOfTwo(2);

                    Shifted_Overlaps[mu].SubtractTo(&Sub,&borrow_g);
                    sum_L.SubtractTo(&Sub,&borrow_L);
                }

                S_i.ComplementTo();
                continue;
            }


            sum_c.SetAll(0);
            sum_cl.SetAll(0);
            LHS.SetAll(0);
            RHS.SetAll(0);
            sum_c_times_2N.SetAll(0);


            for(int mu=0;mu<P;mu++){

                carry_c.Set(0);

                c_i_mu = ~(S_i ^ Patterns[mu][spin]);

                sum_c.AddTo(&c_i_mu,&carry_c);
                sum_cl.AddAnd(&Shifted_Overlaps[mu],&c_i_mu);
            }


            RHS.CopyValues(sum_cl);
            RHS.MultiplyByTwoTo();
            RHS += &CST;


            sum_c.MultiplyByConstant(twoN,&sum_c_times_2N);

            LHS.CopyValues(sum_c_times_2N);
            LHS.AddScalar(random,tmp);
            LHS += &sum_L;


            mask = (RHS <= LHS);


            TwoMask.SetAll(0);
            TwoMask.CopyValues(mask);
            TwoMask.MultiplyByTwoTo();


            for(int mu=0;mu<P;mu++){

                carry_g.Set(0);
                borrow_g.Set(0);

                c_i_mu = ~(S_i ^ Patterns[mu][spin]);
                c_i_mu &= mask;


                Shifted_Overlaps[mu].AddTo(&TwoMask,&carry_g);


                Sub.SetAll(0);
                Sub.CopyValues(c_i_mu);
                Sub.MultiplyByPowerOfTwo(2);


                Shifted_Overlaps[mu].SubtractTo(&Sub,&borrow_g);

                sum_L.AddTo(&TwoMask,&carry_L);
                sum_L.SubtractTo(&Sub,&borrow_L);
            }


            S_i ^= &mask;
        }


        if(save && sweep>0 && sweep%save_stride==0)
            SaveSpinConfigurations(sweep);


        if((sweep+1)%progress_stride==0)
            cout << "\rSweep: "
                 << sweep+1
                 << " ("
                 << (sweep+1)*100/total_sweeps
                 << "%)"
                 << flush;
    }

    cout << endl;
    cout << "bitwise done" << endl;
}

void HopfieldBits::runSweeps_DEBUG(gsl_rng* ran, bool save, double freq, int shift) {
    using namespace std::chrono;
    
    // ============================================================
    // Configuration
    // ============================================================
    const int total_sweeps = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;
    const int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());
    
    // ============================================================
    // Working variables
    // ============================================================
    Bits c_ij, mask;
    UnsignedInt sum_c((unsigned long long) max_deg);
    UnsignedInt sum_cg((unsigned long long) 2 * max_deg * P);
    UnsignedInt sum_c_times_2P((unsigned long long) 2 * max_deg * P);
    UnsignedInt LHS((unsigned long long) 5 * max_deg * P);
    UnsignedInt RHS((unsigned long long) 5 * max_deg * P);
    UnsignedInt tmp           ((unsigned long long) max_deg * P);
    const unsigned long long twoP = 2 * P;
    
    // ============================================================
    // Profiling
    // ============================================================
    struct TimerData {
        double time = 0.0;
        unsigned long long calls = 0;
        
        void add(double t) { time += t; calls++; }
        double average() const { return calls ? time / calls : 0.0; }
    };
    
    TimerData t_precompute, t_cij, t_sum_c, t_addand;
    TimerData t_rhs_copy, t_rhs_mult, t_rhs_add;
    TimerData t_multiply, t_lhs_copy, t_lhs_scalar, t_lhs_add;
    TimerData t_compare, t_spin_update, t_save;
    
    auto total_start = high_resolution_clock::now();
    
    // ============================================================
    // Precompute sum_g_persist
    // ============================================================
    auto pre_start = high_resolution_clock::now();
    vector<UnsignedInt> sum_g_persist(N);
    
    for (int spin = 0; spin < N; ++spin) {
        sum_g_persist[spin] = UnsignedInt((unsigned long long)2 * max_deg * P);
        sum_g_persist[spin].SetAll(0);
        
        for (int k = 0; k < neighbor_count[spin]; ++k)
            sum_g_persist[spin] += &Couplings[spin][k];
    }
    
    t_precompute.add(duration<double>(high_resolution_clock::now() - pre_start).count());
    
    // ============================================================
    // Main simulation
    // ============================================================
    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            int spin = gsl_rng_uniform_int(ran, N);
            int deg_spin = neighbor_count[spin];
            int random = randomNumber(ran, deg_spin * P, N);
            Bits& S_i = Bits_Spins_Set[spin];
            
            // Random spin flip without neighbor calculation
            if (random >= P * deg_spin) {
                auto start = high_resolution_clock::now();
                S_i.ComplementTo();
                t_spin_update.add(duration<double>(high_resolution_clock::now() - start).count());
                continue;
            }

            sum_c.SetAll(0);
            sum_cg.SetAll(0);
            LHS.SetAll(0);
            RHS.SetAll(0);
            tmp.SetAll(0);

            // ====================================================
            // Neighbor loop
            // ====================================================
            for (int k = 0; k < deg_spin; ++k) {
                int j = neighbors[spin][k];
                volatile unsigned long long sink = 0;
                
                // Compute c_ij = ~(S_i ^ S_j)
                auto cij_start = high_resolution_clock::now();
                for (int r = 0; r < 1000; ++r) {
                    c_ij = ~(S_i ^ Bits_Spins_Set[j]);
                    sink ^= c_ij.Get();
                }
                t_cij.add(duration<double>(high_resolution_clock::now() - cij_start).count() / 1000.0);
                
                // Accumulate sum_c
                auto sum_start = high_resolution_clock::now();
                UnsignedInt tmp = sum_c;
                for (int r = 0; r < 1000; ++r) {
                    tmp = sum_c;
                    tmp += &c_ij;
                    sink ^= tmp.Get(0);
                }
                t_sum_c.add(duration<double>(high_resolution_clock::now() - sum_start).count() / 1000.0);
                
                // Accumulate sum_cg
                auto add_start = high_resolution_clock::now();
                sum_cg.AddAnd(&Couplings[spin][k], &c_ij);
                t_addand.add(duration<double>(high_resolution_clock::now() - add_start).count());
            }
            
            // ====================================================
            // RHS calculation: RHS = (2 * sum_cg) + P_times_Neighbor_Count[spin]
            // ====================================================
            auto rhs_copy_start = high_resolution_clock::now();
            RHS.CopyValues(sum_cg);
            t_rhs_copy.add(duration<double>(high_resolution_clock::now() - rhs_copy_start).count());
            
            auto rhs_mult_start = high_resolution_clock::now();
            RHS.MultiplyByTwoTo();
            t_rhs_mult.add(duration<double>(high_resolution_clock::now() - rhs_mult_start).count());
            
            auto rhs_add_start = high_resolution_clock::now();
            RHS += &P_times_Neighbor_Count[spin];
            t_rhs_add.add(duration<double>(high_resolution_clock::now() - rhs_add_start).count());
            
            // ====================================================
            // LHS calculation: LHS = sum_g_persist[spin] + random + (2P * sum_c)
            // ====================================================
            auto mult_start = high_resolution_clock::now();
            sum_c.MultiplyByConstant(twoP, &sum_c_times_2P);
            t_multiply.add(duration<double>(high_resolution_clock::now() - mult_start).count());
            
            auto lhs_copy_start = high_resolution_clock::now();
            LHS.CopyValues(sum_g_persist[spin]);
            t_lhs_copy.add(duration<double>(high_resolution_clock::now() - lhs_copy_start).count());
            
            auto lhs_scalar_start = high_resolution_clock::now();
            LHS.AddScalar(random, tmp);
            t_lhs_scalar.add(duration<double>(high_resolution_clock::now() - lhs_scalar_start).count());
            
            auto lhs_add_start = high_resolution_clock::now();
            LHS += &sum_c_times_2P;
            t_lhs_add.add(duration<double>(high_resolution_clock::now() - lhs_add_start).count());
            
            // ====================================================
            // Decision and spin update
            // ====================================================
            auto compare_start = high_resolution_clock::now();
            mask = (RHS <= LHS);
            t_compare.add(duration<double>(high_resolution_clock::now() - compare_start).count());
            
            auto update_start = high_resolution_clock::now();
            S_i ^= &mask;
            t_spin_update.add(duration<double>(high_resolution_clock::now() - update_start).count());
            
        } // end step loop
        
        // ========================================================
        // Save and progress
        // ========================================================
        if (save && sweep > 0 && (sweep % save_stride == 0)) {
            auto save_start = high_resolution_clock::now();
            SaveSpinConfigurations(sweep);
            t_save.add(duration<double>(high_resolution_clock::now() - save_start).count());
        }
        
        if ((sweep + 1) % progress_stride == 0) {
            cout << "\rSweep: " << sweep + 1 
                 << " (" << (sweep + 1) * 100 / total_sweeps << "%)    " << flush;
        }
    } // end sweep loop
    
    // ============================================================
    // Final profiling report
    // ============================================================
    auto total_end = high_resolution_clock::now();
    double total_time = duration<double>(total_end - total_start).count();
    
    auto print_timer = [&](const string& name, const TimerData& t) {
        double percent = total_time > 0 ? 100.0 * t.time / total_time : 0.0;
        double avg_ns = t.average() * 1e9;
        
        cout << setw(35) << left << name
             << " calls = " << setw(12) << t.calls
             << " time = " << setw(12) << scientific << t.time << " s   "
             << fixed << setprecision(3) << percent << "%   "
             << "avg = " << avg_ns << " ns" << endl;
    };
    
    cout << "\n\n";
    cout << "==================================================\n";
    cout << "                 PROFILING REPORT\n";
    cout << "==================================================\n\n";
    
    cout << "TOTAL EXECUTION TIME : " << fixed << setprecision(6) 
         << total_time << " s\n\n";
    
    cout << "---------------- PRECOMPUTATION ----------------\n";
    print_timer("sum_g_persist construction", t_precompute);
    
    cout << "\n---------------- NEIGHBOURS --------------------\n";
    print_timer("c_ij = ~(S_i ^ S_j)", t_cij);
    print_timer("sum_c += c_ij", t_sum_c);
    print_timer("sum_cg.AddAnd", t_addand);
    
    cout << "\n---------------- RHS ----------------------------\n";
    print_timer("RHS.CopyValues", t_rhs_copy);
    print_timer("RHS.MultiplyByTwoTo", t_rhs_mult);
    print_timer("RHS += P_times_neighbor", t_rhs_add);
    
    cout << "\n---------------- LHS ----------------------------\n";
    print_timer("sum_c.MultiplyByConstant", t_multiply);
    print_timer("LHS.CopyValues", t_lhs_copy);
    print_timer("LHS.AddScalar", t_lhs_scalar);
    print_timer("LHS += sum_c_times_2P", t_lhs_add);
    
    cout << "\n---------------- UPDATE -------------------------\n";
    print_timer("RHS <= LHS comparison", t_compare);
    print_timer("Spin update ^= mask", t_spin_update);
    
    cout << "\n---------------- OUTPUT -------------------------\n";
    print_timer("SaveSpinConfigurations", t_save);
    cout << "\n";
}

// =====================================================
// PUBLIC API
// =====================================================

// Run simulation without saving (thermalization)
void HopfieldBits::evolve(gsl_rng* ran){
    fromCanonical();
    OpenSpinFiles();
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0, 0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve terminated"<<endl;
    toCanonical();
}

// Run simulation and save magnetizations at the given frequency
void HopfieldBits::evolve_save(gsl_rng* ran, double freq){
    fromCanonical();
    cout << "Bitwise conversion done"<<endl;
    OpenSpinFiles();
    SaveSpinConfigurations(0);
    cout << "evolve_save called"<<endl;
    runSweeps(ran, /*save=*/true, freq, 0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save terminated"<<endl;
    toCanonical();
}


// Run simulation and save at given frequency — fully bitwise initialization
void HopfieldBits::evolve_save_bits(gsl_rng* ran, double freq){ 
    initPatternsBits(ran);
    initSpinsBits(ran);
    cout << "Bitwise initialization done" << endl;

    vector<vector<vector<int>>> patterns = getPatternsBitsToCanonical();
    SavePatterns(patterns);
    cout << "Patterns saved" << endl;

    OpenSpinFiles();
    SaveSpinConfigurations(0);
    cout << "evolve_save_bits called"<<endl;
    runSweeps(ran, /*save=*/true, freq, 0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save_bits terminated"<<endl;
}
// Run simulation without saving — fully bitwise initialization
void HopfieldBits::evolve_bits(gsl_rng* ran){
    initPatternsBits(ran);
    initCouplingsBits();
    initSpinsBits(ran);

    vector<vector<vector<int>>> patterns = getPatternsBitsToCanonical();
    SavePatterns(patterns);
    cout << "Patterns saved" << endl;

    OpenSpinFiles();
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0, 0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_bits terminated"<<endl;
}