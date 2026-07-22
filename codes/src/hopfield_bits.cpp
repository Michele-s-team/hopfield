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

// Convert the canonical (scalar) representation of the model
// (spins_set, patterns, couplings) into the bit-sliced representation
// used by the bitwise Metropolis routines below.
void HopfieldBits::fromCanonical() {
    // =====================================================
    // 1. Convert spins
    // =====================================================
    int max_deg = *max_element(degrees.begin(), degrees.end());

    Bits_Spins_Set.clear();
    Bits_Spins_Set.reserve(N);
    Degrees.clear();
    Degrees.reserve(N);
    P_times_Degrees.clear();
    P_times_Degrees.reserve(N);

    for (int spin = 0; spin < N; ++spin) {
        Bits spin_tmp;
        for (int r = 0; r < n_bits; ++r) {
            int bit = (spins_set[r * N + spin] + 1) / 2;
            spin_tmp.Set(r, bit);
        }
        Bits_Spins_Set.push_back(spin_tmp);

        UnsignedInt nc_tmp((unsigned long long)max_deg);
        nc_tmp.SetAll((unsigned long long)degrees[spin]);
        Degrees.push_back(nc_tmp);

        UnsignedInt pnk_tmp((unsigned long long)(P * max_deg));
        pnk_tmp.SetAll((unsigned long long)(P * degrees[spin]));
        P_times_Degrees.push_back(pnk_tmp);
    }

    // =====================================================
    // 2. Convert patterns
    // =====================================================
    Patterns.clear();
    Patterns.resize(P * N);
    for (int mu = 0; mu < P; ++mu) {
        for (int spin = 0; spin < N; ++spin) {
            Bits pat_tmp;
            for (int r = 0; r < n_bits; ++r) {
                pat_tmp.Set(r, (patterns[spin * P * n_bits + mu * n_bits + r] + 1) / 2);
            }
            Patterns[spin * P + mu] = pat_tmp;
        }
    }

    /*
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
    */
}

// Bit representation {0,1} -> canonical spins {-1,+1}.
void HopfieldBits::toCanonical() {
    for (int r = 0; r < n_bits; ++r)
        for (int spin = 0; spin < N; ++spin)
            spins_set[r * N + spin] = -1 + 2 * Bits_Spins_Set[spin].Get(r);
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

// (Re)compute per-spin bit-sliced metadata (Degrees, P_times_Degrees)
// from the current degrees[] array. Called after spin/coupling
// initialization or whenever the topology-dependent metadata needs
// to be refreshed.
void HopfieldBits::initSpinMetadataBits() {
    int max_deg = *max_element(degrees.begin(), degrees.end());

    Degrees.clear();
    Degrees.reserve(N);

    P_times_Degrees.clear();
    P_times_Degrees.reserve(N);

    for (int spin = 0; spin < N; ++spin) {
        UnsignedInt nc_tmp((unsigned long long)max_deg);
        nc_tmp.SetAll((unsigned long long)degrees[spin]);
        Degrees.push_back(nc_tmp);

        UnsignedInt pnk_tmp((unsigned long long)(P * max_deg));
        pnk_tmp.SetAll((unsigned long long)(P * degrees[spin]));
        P_times_Degrees.push_back(pnk_tmp);
    }
}

// Initialize Patterns directly in bit-sliced representation.
// Replaces: initPatterns() + fromCanonical() step 2.
// Patterns[spin*P+mu] is a Bits word: bit r = (xi_spin^mu_r + 1) / 2 in {0,1}.
void HopfieldBits::initPatternsBits(gsl_rng* ran) {
    Patterns.clear();
    Patterns.resize(P * N);
    for (int mu = 0; mu < P; ++mu) {
        for (int spin = 0; spin < N; ++spin) {
            for (int r = 0; r < n_bits; ++r)
                Patterns[spin * P + mu].Set(r, randomBit(ran));
        }
    }
}

// Corrupt a single pattern by flipping each bit independently with
// probability flip_fraction. Input is one column of Patterns (i.e.
// Patterns[mu]), a vector<Bits> of size N, where each Bits word packs
// n_bits independent replicas.
//
// By default each replica r gets its own independent random flip mask
// (i.e. flip_fraction is the *expected* fraction of flipped spins per
// replica, but which spins get flipped differs from one replica to the
// next). This lets you run n_bits independent noise realizations for
// the same flip_fraction in a single bitwise pass.
//
// If you instead want all replicas to share the exact same corrupted
// pattern (same flipped sites for every replica), draw the flip mask
// once per spin (outside the r loop) and apply it to all r — see the
// commented alternative below.
vector<Bits> HopfieldBits::corruptPattern(const vector<Bits>& Patterns,
                                           int mu, double flip_fraction, gsl_rng* ran) const {
    vector<Bits> corrupted = Patterns;

    for (int spin = 0; spin < N; ++spin) {
        for (int r = 0; r < n_bits; ++r) {
            if (gsl_rng_uniform(ran) < flip_fraction) {
                int bit = corrupted[spin * P + mu].Get(r);
                corrupted[spin * P + mu].Set(r, 1 - bit);
            }
        }
    }

    return corrupted;
}

// Initialize Couplings directly in bit-sliced representation via Hebb rule,
// without ever building the scalar couplings[][][] tensor.
// g_ij^r = P + sum_mu xi_i^mu_r * xi_j^mu_r  in [0, 2P]
// Replaces: initCouplings() + fromCanonical() step 3.
// Only neighbor pairs (as given by neighbors[]) are computed; each pair
// is computed once (for spin < j) and mirrored onto the symmetric entry.
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
                Bits prod = ~(Patterns[spin * P + mu] ^ Patterns[j * P + mu]);
                Couplings[spin][k] += &prod;
            }
            Couplings[spin][k].MultiplyByTwoTo();

            // Mirror onto j
            int k_mirror = neighbor_index(j, spin);
            Couplings[j][k_mirror] = Couplings[spin][k];
        }
    }
}

// Same as initCouplingsBits(), but for non-neighbor pairs (as given by
// non_neighbors[]). Used by the non-neighbor overlap sweep variant, where
// the flip condition is expressed in terms of couplings over non-neighbors
// rather than neighbors.
void HopfieldBits::initCouplings_nonNeighbors_Bits() {
    Couplings_nonNeighbors.clear();
    Couplings_nonNeighbors.resize(N);
    for (int spin = 0; spin < N; ++spin) {
        Couplings_nonNeighbors[spin].clear();
        Couplings_nonNeighbors[spin].reserve(non_neighbors[spin].size());
        for (int k = 0; k < (int)non_neighbors[spin].size(); ++k)
            Couplings_nonNeighbors[spin].emplace_back((unsigned long long) 2 * P);
    }

    // Hebb rule: compute once for spin < j, then mirror
    for (int spin = 0; spin < N; ++spin) {
        for (int k = 0; k < (int)non_neighbors[spin].size(); ++k) {
            int j = non_neighbors[spin][k];
            if (j <= spin) continue;

            Couplings_nonNeighbors[spin][k].SetAll((unsigned long long) 0);

            for (int mu = 0; mu < P; ++mu) {
                Bits prod = ~(Patterns[spin * P + mu] ^ Patterns[j * P + mu]);
                Couplings_nonNeighbors[spin][k] += &prod;
            }
            Couplings_nonNeighbors[spin][k].MultiplyByTwoTo();

            // Mirror onto j
            int k_mirror = non_neighbor_index(j, spin);
            Couplings_nonNeighbors[j][k_mirror] = Couplings_nonNeighbors[spin][k];
        }
    }
}

// Overwrite the patterns tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void HopfieldBits::initPatternsFromConfigBits(const vector<Bits>& Config) {
    Patterns = Config;
}

// Overwrite the spin configuration with an externally provided one, and
// refresh the associated bit-sliced metadata.
void HopfieldBits::initSpinsFromConfigBits(const vector<Bits>& Config) {
    Bits_Spins_Set = Config;
    initSpinMetadataBits();
}

// Return a copy of the full pattern tensor.
vector<Bits> HopfieldBits::getPatternsBits() {
    return Patterns;
}

// Return a copy of the full coupling tensor.
vector<vector<UnsignedInt>> HopfieldBits::getCouplingsConfigBits() {
    return Couplings;
}

// =====================================================
// OBSERVABLES
// =====================================================

// Compute magnetization m = (2*ones - N)/N for each of the n_bits
// parallel replicas.
void HopfieldBits::GetMagnetizations(vector<double>& magnetizations) {
    magnetizations.resize(n_bits);
    vector<int> ones(n_bits, 0);

    for (int spin = 0; spin < N; ++spin)
        for (int r = 0; r < n_bits; ++r)
            ones[r] += Bits_Spins_Set[spin].Get(r);

    for (int r = 0; r < n_bits; ++r)
        magnetizations[r] = (2.0 * ones[r] - N) / N;
}

// Convert spin configurations (already in bits) into packed blocks for
// each realization. Uses the same MSB-first convention as PackBlock (and
// SavePatterns), so that spins and patterns are bit-comparable.
void HopfieldBits::GetSpinConfigurations(vector<vector<uint64_t>>& configs) {
    const int n_blocks = num_blocks(N);
    configs.assign(n_bits, vector<uint64_t>(n_blocks, 0));

    for (int b = 0; b < n_blocks; ++b) {
        int start = b * BITS_PER_BLOCK;
        int end   = min(N, start + BITS_PER_BLOCK);

        for (int spin = start; spin < end; ++spin) {
            for (int r = 0; r < n_bits; ++r) {
                configs[r][b] <<= 1;
                if (Bits_Spins_Set[spin].Get(r))
                    configs[r][b] |= 1ULL;
            }
        }
    }
}

// Convert Patterns (bit-sliced) back to the canonical scalar tensor
// patterns[spin * P * n_bits + mu * n_bits + r] in {-1,+1}.
vector<int> HopfieldBits::getPatternsBitsToCanonical() {
    vector<int> result(P * n_bits * N);
    for (int spin = 0; spin < N; ++spin)
        for (int mu = 0; mu < P; ++mu)
            for (int r = 0; r < n_bits; ++r)
                result[spin * P * n_bits + mu * n_bits + r] = -1 + 2 * Patterns[spin * P + mu].Get(r);
    return result;
}

// Compute the shifted extensive overlaps M~^mu = P_r c_i^mu + N for each
// pattern mu, from the current spin configuration. Used to (re)initialize
// Shifted_Overlaps before running the overlap-based sweep variants, and to
// cross-check the incrementally maintained values (see
// check_shifted_overlaps_consistency below).
void HopfieldBits::compute_shifted_overlaps() {
    Shifted_Overlaps.clear();
    Shifted_Overlaps.reserve(P);
    Bits c_ij;
    for (int mu = 0; mu < P; mu++) {
        UnsignedInt sum(2 * N);
        sum.SetAll(0);
        for (int spin = 0; spin < N; spin++) {
            c_ij = ~(Patterns[spin * P + mu] ^ Bits_Spins_Set[spin]);
            sum += &c_ij;
        }
        sum.MultiplyByTwoTo();
        Shifted_Overlaps.push_back(sum);
    }
}

// =====================================================
// METROPOLIS DYNAMICS
// =====================================================

// Dispatches to the fastest sweep variant depending on the mean density
// of the interaction graph, compared against the pattern-loading ratio
// alpha = P/N (see report theory: overlap formulation pays off once
// alpha + 1/2 <= mean_density).
void HopfieldBits::runSweeps(gsl_rng* ran, bool save, double freq, int burn_in) {
    double alpha = (double)P / N;

    double mean_degree  = std::accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
    double mean_density = mean_degree / N;

    if (alpha + 0.5 <= mean_density) {
        // Dense graph: overlap formulation over non-neighbors is cheaper
        compute_shifted_overlaps();
        runSweeps_overlaps_non_neighbors(ran, save, freq, burn_in);
    } else {
        // Sparse graph: direct neighbor-coupling formulation is cheaper
        initCouplingsBits();
        runSweeps_neighbors(ran, save, freq, burn_in);
    }
}

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
//   random in [0, P * deg_spin)   (scaled log-uniform random number)
//
// All UnsignedInt accumulators are preallocated before the sweep loop with
// their maximum possible values to avoid any heap allocation in the hot path.
void HopfieldBits::runSweeps_neighbors(gsl_rng* ran, bool save, double freq, int burn_in) {
    cout << "Neighbors couplings spin update algorithm" << endl;

    const int total_sweeps    = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride     = save ? max(1, (int)round(1.0 / freq)) : 0;

    int max_deg = *max_element(degrees.begin(), degrees.end());

    Bits c_ij, mask;

    UnsignedInt sum_c_ki                     ((unsigned long long) max_deg);
    UnsignedInt two_times_sum_c_ki_couplings ((unsigned long long) 2 * max_deg * P);
    UnsignedInt sum_c_ki_times_2P            ((unsigned long long) 2 * max_deg * P);
    UnsignedInt LHS                          ((unsigned long long) 5 * max_deg * P);
    UnsignedInt RHS                          ((unsigned long long) 5 * max_deg * P);
    UnsignedInt tmp                          ((unsigned long long) max_deg * P);

    const unsigned long long twoP = 2 * P;

    // Precompute Σ_j g_ij once
    vector<UnsignedInt> sum_couplings(N);
    for (int spin = 0; spin < N; ++spin) {
        sum_couplings[spin] = UnsignedInt((unsigned long long)2 * max_deg * P);
        sum_couplings[spin].SetAll(0);

        for (int k = 0; k < degrees[spin]; ++k)
            sum_couplings[spin] += &Couplings[spin][k];
    }

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {

        for (int step = 0; step < N; ++step) {

            int spin     = gsl_rng_uniform_int(ran, N);
            int deg_spin = degrees[spin];
            int random   = randomNumber(ran, deg_spin * P, N);

            Bits& S_i = Bits_Spins_Set[spin];

            // Unconditional flip: unbiased spin, always accepted
            if (random >= P * deg_spin) {
                S_i.ComplementTo();
                continue;
            }

            sum_c_ki.SetAll(0);
            two_times_sum_c_ki_couplings.SetAll(0);
            LHS.SetAll(0);
            RHS.SetAll(0);
            sum_c_ki_times_2P.SetAll(0);

            for (int k = 0; k < deg_spin; ++k) {
                int j = neighbors[spin][k];
                c_ij = ~(S_i ^ Bits_Spins_Set[j]);

                sum_c_ki += &c_ij; // assumes the sum won't overflow the size of "sum_c_ki"
                two_times_sum_c_ki_couplings.AddAnd(&Couplings[spin][k], &c_ij);
            }

            // RHS = 2*two_times_sum_c_ki_couplings + P*deg_spin
            RHS.CopyValues(two_times_sum_c_ki_couplings);
            RHS.MultiplyByTwoTo();
            RHS += &P_times_Degrees[spin];

            // 2P*sum_c_ki
            sum_c_ki.MultiplyByConstant(twoP, &sum_c_ki_times_2P);

            // LHS = random + sum_g + 2P*sum_c_ki
            LHS.CopyValues(sum_couplings[spin]);
            LHS.AddScalar(random, tmp);
            LHS += &sum_c_ki_times_2P;

            mask = (RHS <= LHS);
            S_i ^= &mask;
        }

        if (save && sweep > burn_in && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                 << flush;
    }

    cout << endl;
}

// Recompute Shifted_Overlaps from the current spin state and compare it
// against the values maintained incrementally during the sweep loop.
// Returns true if everything is consistent, false at the first mismatch
// found (which is also printed).
bool HopfieldBits::check_shifted_overlaps_consistency(int sweep, int step) {

    bool ok = true;
    Bits eq_mask;

    for (int mu = 0; mu < P; mu++) {

        UnsignedInt fresh(2 * N);
        Bits c_ij;

        fresh.SetAll(0);

        for (int spin = 0; spin < N; spin++) {
            c_ij = ~(Patterns[spin * P + mu] ^ Bits_Spins_Set[spin]);
            fresh += &c_ij;
        }
        fresh.MultiplyByTwoTo();

        eq_mask = fresh == Shifted_Overlaps[mu];

        if (!eq_mask.equal(Bits_one)) {  // not all 64 bits set -> at least one replica diverges

            unsigned long long mismatches = ~eq_mask.Get();  // bit set to 1 = replica in disagreement

            cout << "[MISMATCH] sweep=" << sweep
                 << " step=" << step
                 << " mu=" << mu
                 << "  disagreeing replicas (mask)=0x" << hex << mismatches << dec
                 << endl;

            ok = false;
        }
    }

    return ok;
}

// Runs Monte Carlo sweeps on 64 parallel Hopfield network instances using a
// bitwise Metropolis algorithm based on the *non-neighbor* couplings and
// overlap formulation.
//
// Instead of summing the local field h_k directly over the neighbors of k
// (cost O(deg_k)), this variant rewrites S_k h_k using the extensive
// (shifted) overlaps M~^mu, then subtracts off the correction due to the
// non-neighbors of k. The resulting flip condition only requires summing
// over the *non-neighbors* of k and (twice) over the patterns, which is
// cheaper than summing over all N spins whenever mean_degree >= alpha + 1/2
// (i.e. the graph is dense).
//
// Flip condition (Metropolis accept/reject), derived from the overlap
// decomposition of S_k h_k restricted to non-neighbors of k:
//
//   LHS >= RHS
//
// where:
//   LHS = random + 2N * sum_mu C_k^mu + sum_mu M~^mu + 2 * sum_{i not in neigh(k), i != k} (C_ik XOR G~_ki)
//   RHS = P * deg_spin + 2 * sum_mu (C_k^mu XOR M~^mu) + sum_{i not in neigh(k), i != k} G~_ki + 2P * sum_{i not in neigh(k), i != k} C_ik
//
// with:
//   M~^mu = M^mu + N  in [0, 2N],  M^mu = sum_i xi_i^mu S_i          (extensive overlap, precomputed/updated incrementally)
//   C_k^mu = (S_k xi_k^mu + 1) / 2  in {0,1}                          (binary counterpart of S_k xi_k^mu)
//   G~_ki  = P + G_ki  in [0, 2P],  G_ki = sum_mu xi_k^mu xi_i^mu     (non-neighbor coupling, precomputed)
//   C_ik   = [S_i == S_k]                                            (1 if same spin, 0 otherwise)
//   random in [0, P * deg_spin)                                      (scaled log-uniform random number)
//
// After the flip-mask computation, the shifted overlaps are updated
// incrementally to reflect the accepted flips.
//
// All UnsignedInt accumulators are preallocated before the sweep loop with
// their maximum possible values to avoid any heap allocation in the hot path.
// Couplings over non-neighbor pairs (Couplings_nonNeighbors) are precomputed
// once via initCouplings_nonNeighbors_Bits() before the sweep loop.
//
// Falls back to the neighbor-based algorithm (runSweeps_overlaps) whenever
// the graph is fully connected (min degree == N-1), since the non-neighbor
// set is then empty and this formulation offers no benefit.
void HopfieldBits::runSweeps_overlaps_non_neighbors(gsl_rng* ran, bool save, double freq, int burn_in) {

    int min_deg = *min_element(degrees.begin(), degrees.end());
    if (min_deg == N - 1) { runSweeps_overlaps(ran, save, freq, burn_in); return; }

    cout << "Overlaps and non-neighbors couplings spin update algorithm" << endl;

    initCouplings_nonNeighbors_Bits();

    const int total_sweeps    = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride     = save ? max(1, (int)round(1.0 / freq)) : 0;

    int max_size_non_neighbors = N - 1 - min_deg;

    Bits mask;
    Bits c_ij;
    Bits carry_g, borrow_g;  // carry/borrow for Shifted_Overlaps[mu]
    Bits carry_L;            // carry/borrow for sum_shifted_overlaps
    Bits carry_c;

    Bits borrow_delta;

    UnsignedInt sum_c_ki                     ((unsigned long long) max_size_non_neighbors);
    UnsignedInt two_times_sum_c_ki_couplings ((unsigned long long) 4 * max_size_non_neighbors * P);
    UnsignedInt sum_c_ki_times_2P            ((unsigned long long) 2 * max_size_non_neighbors * P);
    UnsignedInt sum_c_k_mu                   ((unsigned long long) P);
    UnsignedInt sum_c_shifted_overlap        ((unsigned long long) 2 * N * P);
    UnsignedInt sum_c_k_mu_times_2N          ((unsigned long long) 2 * N * P);

    UnsignedInt LHS     ((unsigned long long) P * (5 * N - 1));
    UnsignedInt RHS     ((unsigned long long) P * (5 * N - 1));
    UnsignedInt tmp     ((unsigned long long) P * (N - 1));
    UnsignedInt fourSum ((unsigned long long) 4 * P);
    UnsignedInt TwoMask ((unsigned long long) 2);

    UnsignedInt CST ((unsigned long long) (N - 1) * P);
    UnsignedInt Two ((unsigned long long) 2);
    Two.SetAll(2);

    UnsignedInt TwoP ((unsigned long long) 2 * P);
    TwoP.SetAll(2 * P);

    UnsignedInt sum_shifted_overlaps ((unsigned long long) 2 * N * P);
    sum_shifted_overlaps.SetAll(0);

    const unsigned long long twoN = 2 * N;
    const unsigned long long twoP = 2 * P;

    vector<Bits> c_cache(P);

    // Precompute Σ_mu L^mu
    for (int mu = 0; mu < P; ++mu) {
        sum_shifted_overlaps += &Shifted_Overlaps[mu];
    }

    // Precompute Σ_j g_ij once
    vector<UnsignedInt> sum_couplings(N);
    for (int spin = 0; spin < N; ++spin) {
        sum_couplings[spin] = UnsignedInt((unsigned long long)2 * max_size_non_neighbors * P);
        sum_couplings[spin].SetAll(0);

        for (int k = 0; k < N - 1 - degrees[spin]; ++k)
            sum_couplings[spin] += &Couplings_nonNeighbors[spin][k];
    }

    cout << "initialization done" << endl;

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {

            int spin               = gsl_rng_uniform_int(ran, N);
            int deg_spin            = degrees[spin];
            int non_neighbors_size = N - 1 - deg_spin;
            CST.SetAll(deg_spin * P);
            int random = randomNumber(ran, deg_spin * P, N);

            Bits& S_i = Bits_Spins_Set[spin];

            // Unconditional flip: unbiased spin, always accepted
            if (random >= P * deg_spin) {
                sum_c_k_mu.SetAll(0);
                for (int mu = 0; mu < P; mu++) {

                    c_cache[mu] = ~(S_i ^ Patterns[spin * P + mu]);
                    sum_c_k_mu += &c_cache[mu];

                    Shifted_Overlaps[mu] += &Two;
                    Shifted_Overlaps[mu].SubtractShifted(&c_cache[mu], 2, &borrow_g);
                }
                sum_shifted_overlaps += &TwoP;
                fourSum.CopyValues(sum_c_k_mu);
                fourSum.MultiplyByPowerOfTwo(2);

                sum_shifted_overlaps.SubtractTo(&fourSum, &borrow_delta);

                S_i.ComplementTo();
                continue;
            }

            sum_c_k_mu.SetAll(0);
            sum_c_shifted_overlap.SetAll(0);
            sum_c_ki.SetAll(0);
            two_times_sum_c_ki_couplings.SetAll(0);
            sum_c_ki_times_2P.SetAll(0);
            LHS.SetAll(0);
            RHS.SetAll(0);

            for (int k = 0; k < non_neighbors_size; ++k) {
                int j = non_neighbors[spin][k];
                c_ij = ~(S_i ^ Bits_Spins_Set[j]);

                sum_c_ki += &c_ij;  // assumes the sum won't overflow the size of "sum_c_ki"
                two_times_sum_c_ki_couplings.AddAnd(&Couplings_nonNeighbors[spin][k], &c_ij);
            }

            for (int mu = 0; mu < P; ++mu) {
                carry_c.Set(0);

                c_cache[mu] = ~(S_i ^ Patterns[spin * P + mu]);

                sum_c_k_mu += &c_cache[mu];
                sum_c_shifted_overlap.AddAnd(&Shifted_Overlaps[mu], &c_cache[mu]);
            }

            // LHS = random + 2N*sum_c_k_mu + sum_shifted_overlaps + 2*sum_c_ki_couplings
            sum_c_k_mu.MultiplyByConstant(twoN, &sum_c_k_mu_times_2N);
            LHS.CopyValues(sum_c_k_mu_times_2N);
            LHS.AddScalar(random, tmp);
            LHS += &sum_shifted_overlaps;

            two_times_sum_c_ki_couplings.MultiplyByTwoTo();
            LHS += &two_times_sum_c_ki_couplings;

            // RHS = n_k*P + 2*sum_c_shifted_overlap + sum_couplings + 2P*sum_c_ki
            RHS.CopyValues(sum_c_shifted_overlap);
            RHS.MultiplyByTwoTo();

            RHS += &P_times_Degrees[spin];
            RHS += &sum_couplings[spin];

            // 2P*sum_c_ki
            sum_c_ki.MultiplyByConstant(twoP, &sum_c_ki_times_2P);
            RHS += &sum_c_ki_times_2P;

            mask = (RHS <= LHS);

            TwoMask.SetAll(0);
            TwoMask.CopyValues(mask);
            TwoMask.MultiplyByTwoTo();

            for (int mu = 0; mu < P; mu++) {

                c_cache[mu] &= mask;

                Shifted_Overlaps[mu] += &TwoMask;
                Shifted_Overlaps[mu].SubtractShifted(&c_cache[mu], 2, &borrow_g);
            }

            sum_shifted_overlaps.AddMasked(&TwoP, &mask);

            sum_c_k_mu &= &mask;

            fourSum.CopyValues(sum_c_k_mu);
            fourSum.MultiplyByPowerOfTwo(2);

            borrow_delta.Set(0);
            sum_shifted_overlaps.SubtractTo(&fourSum, &borrow_delta);

            S_i ^= &mask;
        }

        if (save && sweep > burn_in && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                 << flush;
    }

    cout << endl;
    cout << " bitwise done" << endl;
}

// Runs Monte Carlo sweeps on 64 parallel Hopfield network instances using a
// bitwise Metropolis algorithm based on the extensive-overlap formulation,
// specialized for a fully-connected network (deg_k = N-1 for all k).
// All 64 replicas are updated simultaneously via bit-sliced arithmetic on
// UnsignedInt objects.
//
// Because the graph is fully connected, the non-neighbor correction terms of
// the general flip condition vanish entirely: summing over "non-neighbors of
// k" is summing over the empty set. The local field S_k h_k can therefore be
// expressed purely in terms of the extensive (shifted) overlaps M~^mu,
// avoiding any O(N) (or O(deg_k)) loop over other spins in the hot path.
//
// Flip condition (Metropolis accept/reject), specialized from the general
// overlap decomposition to the fully-connected case:
//
//   LHS >= RHS
//
// where:
//   LHS = random + 2N * sum_mu C_k^mu + sum_mu M~^mu
//   RHS = P * (N-1) + 2 * sum_mu (C_k^mu XOR M~^mu)
//
// with:
//   M~^mu = M^mu + N  in [0, 2N],  M^mu = sum_i xi_i^mu S_i        (extensive overlap, precomputed/updated incrementally)
//   C_k^mu = (S_k xi_k^mu + 1) / 2  in {0,1}                        (binary counterpart of S_k xi_k^mu)
//   random in [0, P * deg_spin)                                     (scaled log-uniform random number; deg_spin = N-1 here)
//
// After the flip-mask computation, the shifted overlaps are updated
// incrementally to reflect the accepted flips.
//
// This is the cheapest of the three sweep variants (O(P) per spin update,
// no dependence on N beyond the overlap terms).
//
// All UnsignedInt accumulators are preallocated before the sweep loop with
// their maximum possible values to avoid any heap allocation in the hot path.
void HopfieldBits::runSweeps_overlaps(gsl_rng* ran, bool save, double freq, int burn_in) {

    cout << "Overlaps spin update algorithm" << endl;

    const int total_sweeps    = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride     = save ? max(1, (int)round(1.0 / freq)) : 0;

    Bits mask;
    Bits carry_g, borrow_g;  // carry/borrow for Shifted_Overlaps[mu]
    Bits carry_L;            // carry/borrow for sum_shifted_overlaps
    Bits carry_c;

    Bits borrow_delta;

    UnsignedInt sum_c_k_mu            ((unsigned long long) P);
    UnsignedInt sum_c_shifted_overlap ((unsigned long long) 2 * N * P);
    UnsignedInt sum_c_k_mu_times_2N   ((unsigned long long) 2 * N * P);
    UnsignedInt LHS                   ((unsigned long long) P * (5 * N - 1));
    UnsignedInt RHS                   ((unsigned long long) P * (5 * N - 1));
    UnsignedInt tmp                   ((unsigned long long) P * (N - 1));
    UnsignedInt fourSum               ((unsigned long long) 4 * P);
    UnsignedInt TwoMask               ((unsigned long long) 2);

    UnsignedInt CST ((unsigned long long) (N - 1) * P);
    CST.SetAll((N - 1) * P);

    UnsignedInt Two ((unsigned long long) 2);
    Two.SetAll(2);

    UnsignedInt TwoP ((unsigned long long) 2 * P);
    TwoP.SetAll(2 * P);

    UnsignedInt sum_shifted_overlaps ((unsigned long long) 2 * N * P);
    sum_shifted_overlaps.SetAll(0);

    const unsigned long long twoN = 2 * N;

    vector<Bits> c_cache(P);

    cout << "initialization done" << endl;

    // Precompute Σ_mu L^mu
    for (int mu = 0; mu < P; ++mu) {
        sum_shifted_overlaps += &Shifted_Overlaps[mu];
    }

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {

        for (int step = 0; step < N; ++step) {

            int spin     = gsl_rng_uniform_int(ran, N);
            int deg_spin = degrees[spin];
            int random   = randomNumber(ran, deg_spin * P, N);

            Bits& S_i = Bits_Spins_Set[spin];

            // Unconditional flip: unbiased spin, always accepted
            if (random >= P * deg_spin) {
                sum_c_k_mu.SetAll(0);
                for (int mu = 0; mu < P; mu++) {

                    c_cache[mu] = ~(S_i ^ Patterns[spin * P + mu]);
                    sum_c_k_mu += &c_cache[mu];

                    Shifted_Overlaps[mu] += &Two;
                    Shifted_Overlaps[mu].SubtractShifted(&c_cache[mu], 2, &borrow_g);
                }
                sum_shifted_overlaps += &TwoP;
                fourSum.CopyValues(sum_c_k_mu);
                fourSum.MultiplyByPowerOfTwo(2);

                sum_shifted_overlaps.SubtractTo(&fourSum, &borrow_delta);

                S_i.ComplementTo();
                continue;
            }

            sum_c_k_mu.SetAll(0);
            sum_c_shifted_overlap.SetAll(0);
            LHS.SetAll(0);
            RHS.SetAll(0);

            for (int mu = 0; mu < P; ++mu) {
                carry_c.Set(0);

                c_cache[mu] = ~(S_i ^ Patterns[spin * P + mu]);

                sum_c_k_mu += &c_cache[mu];
                sum_c_shifted_overlap.AddAnd(&Shifted_Overlaps[mu], &c_cache[mu]);
            }

            RHS.CopyValues(sum_c_shifted_overlap);
            RHS.MultiplyByTwoTo();
            RHS += &CST;

            sum_c_k_mu.MultiplyByConstant(twoN, &sum_c_k_mu_times_2N);
            LHS.CopyValues(sum_c_k_mu_times_2N);
            LHS.AddScalar(random, tmp);
            LHS += &sum_shifted_overlaps;

            mask = (RHS <= LHS);

            TwoMask.SetAll(0);
            TwoMask.CopyValues(mask);
            TwoMask.MultiplyByTwoTo();

            for (int mu = 0; mu < P; mu++) {

                c_cache[mu] &= mask;

                Shifted_Overlaps[mu] += &TwoMask;
                Shifted_Overlaps[mu].SubtractShifted(&c_cache[mu], 2, &borrow_g);
            }

            sum_shifted_overlaps.AddMasked(&TwoP, &mask);

            sum_c_k_mu &= &mask;

            fourSum.CopyValues(sum_c_k_mu);
            fourSum.MultiplyByPowerOfTwo(2);

            borrow_delta.Set(0);
            sum_shifted_overlaps.SubtractTo(&fourSum, &borrow_delta);

            S_i ^= &mask;
        }

        if (save && sweep > burn_in && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: " << sweep + 1
                 << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                 << flush;
    }

    cout << endl;
    cout << " bitwise done" << endl;
}

void HopfieldBits::runSweeps_neighbors_DEBUG(gsl_rng* ran, bool save, double freq, int burn_in) {
    using namespace std::chrono;
    
    // ============================================================
    // Configuration
    // ============================================================
    const int total_sweeps = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride = save ? max(1, (int)round(1.0 / freq)) : 0;
    const int max_deg = *max_element(degrees.begin(), degrees.end());
    
    // ============================================================
    // Working variables
    // ============================================================
    Bits c_ij, mask;
    UnsignedInt sum_c_ki((unsigned long long) max_deg);
    UnsignedInt two_times_sum_c_ki_couplings((unsigned long long) 2 * max_deg * P);
    UnsignedInt sum_c_ki_times_2P((unsigned long long) 2 * max_deg * P);
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
    // Precompute sum_couplings
    // ============================================================
    auto pre_start = high_resolution_clock::now();
    vector<UnsignedInt> sum_couplings(N);
    
    for (int spin = 0; spin < N; ++spin) {
        sum_couplings[spin] = UnsignedInt((unsigned long long)2 * max_deg * P);
        sum_couplings[spin].SetAll(0);
        
        for (int k = 0; k < degrees[spin]; ++k)
            sum_couplings[spin] += &Couplings[spin][k];
    }
    
    t_precompute.add(duration<double>(high_resolution_clock::now() - pre_start).count());
    
    // ============================================================
    // Main simulation
    // ============================================================
    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            int spin = gsl_rng_uniform_int(ran, N);
            int deg_spin = degrees[spin];
            int random = randomNumber(ran, deg_spin * P, N);
            Bits& S_i = Bits_Spins_Set[spin];
            
            // Random spin flip without neighbor calculation
            if (random >= P * deg_spin) {
                auto start = high_resolution_clock::now();
                S_i.ComplementTo();
                t_spin_update.add(duration<double>(high_resolution_clock::now() - start).count());
                continue;
            }

            sum_c_ki.SetAll(0);
            two_times_sum_c_ki_couplings.SetAll(0);
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
                
                // Accumulate sum_c_ki
                auto sum_start = high_resolution_clock::now();
                UnsignedInt tmp = sum_c_ki;
                for (int r = 0; r < 1000; ++r) {
                    tmp = sum_c_ki;
                    tmp += &c_ij;
                    sink ^= tmp.Get(0);
                }
                t_sum_c.add(duration<double>(high_resolution_clock::now() - sum_start).count() / 1000.0);
                
                // Accumulate two_times_sum_c_ki_couplings
                auto add_start = high_resolution_clock::now();
                two_times_sum_c_ki_couplings.AddAnd(&Couplings[spin][k], &c_ij);
                t_addand.add(duration<double>(high_resolution_clock::now() - add_start).count());
            }
            
            // ====================================================
            // RHS calculation: RHS = (2 * two_times_sum_c_ki_couplings) + P_times_Degrees[spin]
            // ====================================================
            auto rhs_copy_start = high_resolution_clock::now();
            RHS.CopyValues(two_times_sum_c_ki_couplings);
            t_rhs_copy.add(duration<double>(high_resolution_clock::now() - rhs_copy_start).count());
            
            auto rhs_mult_start = high_resolution_clock::now();
            RHS.MultiplyByTwoTo();
            t_rhs_mult.add(duration<double>(high_resolution_clock::now() - rhs_mult_start).count());
            
            auto rhs_add_start = high_resolution_clock::now();
            RHS += &P_times_Degrees[spin];
            t_rhs_add.add(duration<double>(high_resolution_clock::now() - rhs_add_start).count());
            
            // ====================================================
            // LHS calculation: LHS = sum_couplings[spin] + random + (2P * sum_c_ki)
            // ====================================================
            auto mult_start = high_resolution_clock::now();
            sum_c_ki.MultiplyByConstant(twoP, &sum_c_ki_times_2P);
            t_multiply.add(duration<double>(high_resolution_clock::now() - mult_start).count());
            
            auto lhs_copy_start = high_resolution_clock::now();
            LHS.CopyValues(sum_couplings[spin]);
            t_lhs_copy.add(duration<double>(high_resolution_clock::now() - lhs_copy_start).count());
            
            auto lhs_scalar_start = high_resolution_clock::now();
            LHS.AddScalar(random, tmp);
            t_lhs_scalar.add(duration<double>(high_resolution_clock::now() - lhs_scalar_start).count());
            
            auto lhs_add_start = high_resolution_clock::now();
            LHS += &sum_c_ki_times_2P;
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
        if (save && sweep > burn_in && (sweep % save_stride == 0)) {
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
    print_timer("sum_couplings construction", t_precompute);
    
    cout << "\n---------------- NEIGHBOURS --------------------\n";
    print_timer("c_ij = ~(S_i ^ S_j)", t_cij);
    print_timer("sum_c_ki += c_ij", t_sum_c);
    print_timer("two_times_sum_c_ki_couplings.AddAnd", t_addand);
    
    cout << "\n---------------- RHS ----------------------------\n";
    print_timer("RHS.CopyValues", t_rhs_copy);
    print_timer("RHS.MultiplyByTwoTo", t_rhs_mult);
    print_timer("RHS += P_times_neighbor", t_rhs_add);
    
    cout << "\n---------------- LHS ----------------------------\n";
    print_timer("sum_c_ki.MultiplyByConstant", t_multiply);
    print_timer("LHS.CopyValues", t_lhs_copy);
    print_timer("LHS.AddScalar", t_lhs_scalar);
    print_timer("LHS += sum_c_ki_times_2P", t_lhs_add);
    
    cout << "\n---------------- UPDATE -------------------------\n";
    print_timer("RHS <= LHS comparison", t_compare);
    print_timer("Spin update ^= mask", t_spin_update);
    
    cout << "\n---------------- OUTPUT -------------------------\n";
    print_timer("SaveSpinConfigurations", t_save);
    cout << "\n";
}


void HopfieldBits::runSweeps_overlaps_DEBUG(gsl_rng* ran, bool save, double freq, int burn_in){
using namespace std::chrono;

// ============================================================
// Configuration
// ============================================================
const int total_sweeps = getNSweeps();
const int progress_stride = max(1,total_sweeps/10);
const int save_stride = save ? max(1,(int)round(1.0/freq)) : 0;

// ============================================================
// Working variables
// ============================================================
Bits mask;
Bits carry_g, borrow_g;
Bits carry_L;
Bits carry_c;
Bits borrow_delta;

UnsignedInt sum_c_k_mu((unsigned long long)P);
UnsignedInt sum_c_shifted_overlap((unsigned long long)2*N*P);
UnsignedInt sum_c_k_mu_times_2N((unsigned long long)2*N*P);
UnsignedInt LHS((unsigned long long)P*(5*N-1));
UnsignedInt RHS((unsigned long long)P*(5*N-1));
UnsignedInt tmp((unsigned long long)P*(N-1));
UnsignedInt fourSum((unsigned long long)4*P);
UnsignedInt TwoMask((unsigned long long)2);

UnsignedInt CST((unsigned long long)(N-1)*P);
CST.SetAll((N-1)*P);

UnsignedInt Two((unsigned long long)2);
Two.SetAll(2);

UnsignedInt TwoP((unsigned long long)2*P);
TwoP.SetAll(2*P);

UnsignedInt sum_shifted_overlaps((unsigned long long)2*N*P);
sum_shifted_overlaps.SetAll(0);

const unsigned long long twoN = 2*N;

vector<Bits> c_cache(P);

// ============================================================
// Profiling structure
// ============================================================
struct TimerData{
    double time=0.0;
    unsigned long long calls=0;

    void add(double t){
        time+=t;
        calls++;
    }

    double average() const{
        return calls ? time/calls : 0.0;
    }
};

// Global
TimerData t_total_loop;

// Initialization
TimerData t_sumL_init;

// Random and selection
TimerData t_random_spin;
TimerData t_random_number;

// Unconditional flip
TimerData t_unc_total;
TimerData t_unc_ccache;
TimerData t_unc_sumc;
TimerData t_unc_overlap_update;
TimerData t_unc_sumL_update;
TimerData t_unc_spin_update;

// Conditional flip
TimerData t_cond_total;
TimerData t_cond_ccache;
TimerData t_cond_sumc;
TimerData t_cond_sumcl;

// RHS
TimerData t_rhs_copy;
TimerData t_rhs_mult;
TimerData t_rhs_add;

// LHS
TimerData t_lhs_mult;
TimerData t_lhs_copy;
TimerData t_lhs_scalar;
TimerData t_lhs_add;

// Decision
TimerData t_compare;

// Mask and overlap update
TimerData t_mask_prepare;
TimerData t_overlap_update;
TimerData t_sumL_mask;
TimerData t_sumc_mask;
TimerData t_fourSum;
TimerData t_sumL_subtract;

// Spin
TimerData t_spin_flip;

// Output
TimerData t_save;
TimerData t_progress;

auto total_start = high_resolution_clock::now();

// ============================================================
// Precompute sum_shifted_overlaps
// ============================================================
auto start_sumL = high_resolution_clock::now();

for(int mu=0;mu<P;++mu)
    sum_shifted_overlaps += &Shifted_Overlaps[mu];

t_sumL_init.add(duration<double>(high_resolution_clock::now()-start_sumL).count());

cout<<"initialization done"<<endl;

// ============================================================
// Main simulation
// ============================================================
for(int sweep=0;sweep<total_sweeps;++sweep){

    auto loop_start = high_resolution_clock::now();

    for(int step=0;step<N;++step){

        auto random_start = high_resolution_clock::now();
        int spin = gsl_rng_uniform_int(ran,N);
        t_random_spin.add(duration<double>(high_resolution_clock::now()-random_start).count());

        int deg_spin = degrees[spin];

        auto randnum_start = high_resolution_clock::now();
        int random = randomNumber(ran,deg_spin*P,N);
        t_random_number.add(duration<double>(high_resolution_clock::now()-randnum_start).count());

        Bits& S_i = Bits_Spins_Set[spin];


        // ====================================================
        // Unconditional flip
        // ====================================================
        if(random >= P*deg_spin){

            auto unc_total_start = high_resolution_clock::now();

            sum_c_k_mu.SetAll(0);

            auto unc_ccache_start = high_resolution_clock::now();

            for(int mu=0;mu<P;mu++){
                c_cache[mu]=~(S_i ^ Patterns[spin*P+mu]);
            }

            t_unc_ccache.add(duration<double>(high_resolution_clock::now()-unc_ccache_start).count());


            auto unc_sumc_start = high_resolution_clock::now();

            for(int mu=0;mu<P;mu++)
                sum_c_k_mu += &c_cache[mu];

            t_unc_sumc.add(duration<double>(high_resolution_clock::now()-unc_sumc_start).count());


            auto unc_overlap_start = high_resolution_clock::now();

            for(int mu=0;mu<P;mu++){

                Shifted_Overlaps[mu]+=&Two;

                Shifted_Overlaps[mu].SubtractShifted(
                    &c_cache[mu],
                    2,
                    &borrow_g
                );
            }

            t_unc_overlap_update.add(duration<double>(high_resolution_clock::now()-unc_overlap_start).count());


            auto unc_sumL_start = high_resolution_clock::now();

            sum_shifted_overlaps += &TwoP;

            fourSum.CopyValues(sum_c_k_mu);
            fourSum.MultiplyByPowerOfTwo(2);

            sum_shifted_overlaps.SubtractTo(&fourSum,&borrow_delta);

            t_unc_sumL_update.add(duration<double>(high_resolution_clock::now()-unc_sumL_start).count());


            auto unc_spin_start = high_resolution_clock::now();

            S_i.ComplementTo();

            t_unc_spin_update.add(duration<double>(high_resolution_clock::now()-unc_spin_start).count());


            t_unc_total.add(duration<double>(high_resolution_clock::now()-unc_total_start).count());

            continue;
        }


        // ====================================================
        // Conditional flip
        // ====================================================
        auto cond_total_start = high_resolution_clock::now();

        sum_c_k_mu.SetAll(0);
        sum_c_shifted_overlap.SetAll(0);
        LHS.SetAll(0);
        RHS.SetAll(0);


        auto cond_ccache_start = high_resolution_clock::now();

        for(int mu=0;mu<P;++mu){

            c_cache[mu]=~(S_i ^ Patterns[spin*P+mu]);

        }

        t_cond_ccache.add(duration<double>(high_resolution_clock::now()-cond_ccache_start).count());


        auto cond_sumc_start = high_resolution_clock::now();

        for(int mu=0;mu<P;++mu)
            sum_c_k_mu += &c_cache[mu];

        t_cond_sumc.add(duration<double>(high_resolution_clock::now()-cond_sumc_start).count());


        auto cond_sumcl_start = high_resolution_clock::now();

        for(int mu=0;mu<P;++mu)
            sum_c_shifted_overlap.AddAnd(&Shifted_Overlaps[mu],&c_cache[mu]);

        t_cond_sumcl.add(duration<double>(high_resolution_clock::now()-cond_sumcl_start).count());
// ====================================================
// RHS calculation
// ====================================================
auto rhs_copy_start = high_resolution_clock::now();

RHS.CopyValues(sum_c_shifted_overlap);

t_rhs_copy.add(duration<double>(high_resolution_clock::now()-rhs_copy_start).count());


auto rhs_mult_start = high_resolution_clock::now();

RHS.MultiplyByTwoTo();

t_rhs_mult.add(duration<double>(high_resolution_clock::now()-rhs_mult_start).count());


auto rhs_add_start = high_resolution_clock::now();

RHS += &CST;

t_rhs_add.add(duration<double>(high_resolution_clock::now()-rhs_add_start).count());


// ====================================================
// LHS calculation
// ====================================================
auto lhs_mult_start = high_resolution_clock::now();

sum_c_k_mu.MultiplyByConstant(twoN,&sum_c_k_mu_times_2N);

t_lhs_mult.add(duration<double>(high_resolution_clock::now()-lhs_mult_start).count());


auto lhs_copy_start = high_resolution_clock::now();

LHS.CopyValues(sum_c_k_mu_times_2N);

t_lhs_copy.add(duration<double>(high_resolution_clock::now()-lhs_copy_start).count());


auto lhs_scalar_start = high_resolution_clock::now();

LHS.AddScalar(random,tmp);

t_lhs_scalar.add(duration<double>(high_resolution_clock::now()-lhs_scalar_start).count());


auto lhs_add_start = high_resolution_clock::now();

LHS += &sum_shifted_overlaps;

t_lhs_add.add(duration<double>(high_resolution_clock::now()-lhs_add_start).count());


// ====================================================
// Decision
// ====================================================
auto compare_start = high_resolution_clock::now();

mask=(RHS<=LHS);

t_compare.add(duration<double>(high_resolution_clock::now()-compare_start).count());


// ====================================================
// Prepare doubled mask
// ====================================================
auto mask_start = high_resolution_clock::now();

TwoMask.SetAll(0);
TwoMask.CopyValues(mask);
TwoMask.MultiplyByTwoTo();

t_mask_prepare.add(duration<double>(high_resolution_clock::now()-mask_start).count());


// ====================================================
// Update shifted overlaps
// ====================================================
auto overlap_start = high_resolution_clock::now();

for(int mu=0;mu<P;mu++){

    c_cache[mu] &= mask;

    Shifted_Overlaps[mu] += &TwoMask;

    Shifted_Overlaps[mu].SubtractShifted(
        &c_cache[mu],
        2,
        &borrow_g
    );
}

t_overlap_update.add(duration<double>(high_resolution_clock::now()-overlap_start).count());


// ====================================================
// Update sum_shifted_overlaps
// ====================================================
auto sumL_mask_start = high_resolution_clock::now();

sum_shifted_overlaps.AddMasked(&TwoP,&mask);

t_sumL_mask.add(duration<double>(high_resolution_clock::now()-sumL_mask_start).count());


auto sumc_mask_start = high_resolution_clock::now();

sum_c_k_mu &= &mask;

t_sumc_mask.add(duration<double>(high_resolution_clock::now()-sumc_mask_start).count());


auto fourSum_start = high_resolution_clock::now();

fourSum.CopyValues(sum_c_k_mu);
fourSum.MultiplyByPowerOfTwo(2);

t_fourSum.add(duration<double>(high_resolution_clock::now()-fourSum_start).count());


auto subtract_start = high_resolution_clock::now();

borrow_delta.Set(0);

sum_shifted_overlaps.SubtractTo(&fourSum,&borrow_delta);

t_sumL_subtract.add(duration<double>(high_resolution_clock::now()-subtract_start).count());


// ====================================================
// Spin update
// ====================================================
auto spin_update_start = high_resolution_clock::now();

S_i ^= &mask;

t_spin_flip.add(duration<double>(high_resolution_clock::now()-spin_update_start).count());


t_cond_total.add(duration<double>(high_resolution_clock::now()-cond_total_start).count());

} // end step loop

// ========================================================
// End of sweep
// ========================================================
auto loop_end = high_resolution_clock::now();

t_total_loop.add(duration<double>(loop_end-loop_start).count());


if(save && sweep>0 && (sweep%save_stride==0)){

    auto save_start = high_resolution_clock::now();

    SaveSpinConfigurations(sweep);

    t_save.add(duration<double>(high_resolution_clock::now()-save_start).count());
}


if((sweep+1)%progress_stride==0){

    auto progress_start = high_resolution_clock::now();

    cout<<"\rSweep: "<<sweep+1
        <<" ("<<(sweep+1)*100/total_sweeps<<"%)    "
        <<flush;

    t_progress.add(duration<double>(high_resolution_clock::now()-progress_start).count());
}

} // end sweep loop


cout<<endl;


// ============================================================
// Final profiling report
// ============================================================
auto total_end = high_resolution_clock::now();

double total_time = duration<double>(total_end-total_start).count();


auto print_timer = [&](const string& name,const TimerData& t){

    double percent = total_time>0 ? 100.0*t.time/total_time : 0.0;
    double avg_ns = t.average()*1e9;

    cout<<setw(40)<<left<<name
        <<" calls = "<<setw(12)<<t.calls
        <<" time = "<<setw(12)<<scientific<<t.time<<" s   "
        <<fixed<<setprecision(3)<<percent<<"%   "
        <<"avg = "<<avg_ns<<" ns"
        <<endl;
};


cout<<"\n\n";
cout<<"==================================================\n";
cout<<"              OVERLAPS PROFILING REPORT\n";
cout<<"==================================================\n\n";


cout<<"TOTAL EXECUTION TIME : "
    <<fixed<<setprecision(6)
    <<total_time<<" s\n\n";


// ============================================================
// Initialization
// ============================================================
cout<<"---------------- INITIALIZATION ----------------\n";

print_timer("Precompute sum_shifted_overlaps",t_sumL_init);


// ============================================================
// Main loop
// ============================================================
cout<<"\n---------------- MAIN LOOP ----------------------\n";

print_timer("Complete sweep loop",t_total_loop);

print_timer("Random spin generation",t_random_spin);

print_timer("Random number generation",t_random_number);


// ============================================================
// Unconditional flip
// ============================================================
cout<<"\n---------------- UNCONDITIONAL FLIP -------------\n";

print_timer("Total unconditional flip",t_unc_total);

print_timer("Compute c_cache",t_unc_ccache);

print_timer("Compute sum_c_k_mu",t_unc_sumc);

print_timer("Update Shifted_Overlaps",t_unc_overlap_update);

print_timer("Update sum_shifted_overlaps",t_unc_sumL_update);

print_timer("Spin ComplementTo",t_unc_spin_update);


// ============================================================
// Conditional flip
// ============================================================
cout<<"\n---------------- CONDITIONAL FLIP ---------------\n";

print_timer("Total conditional flip",t_cond_total);

print_timer("Compute c_cache",t_cond_ccache);

print_timer("Compute sum_c_k_mu",t_cond_sumc);

print_timer("Compute sum_c_shifted_overlap",t_cond_sumcl);


// ============================================================
// RHS
// ============================================================
cout<<"\n---------------- RHS ----------------------------\n";

print_timer("RHS.CopyValues",t_rhs_copy);

print_timer("RHS.MultiplyByTwoTo",t_rhs_mult);

print_timer("RHS += CST",t_rhs_add);


// ============================================================
// LHS
// ============================================================
cout<<"\n---------------- LHS ----------------------------\n";

print_timer("sum_c_k_mu.MultiplyByConstant",t_lhs_mult);

print_timer("LHS.CopyValues",t_lhs_copy);

print_timer("LHS.AddScalar",t_lhs_scalar);

print_timer("LHS += sum_shifted_overlaps",t_lhs_add);


// ============================================================
// Decision and updates
// ============================================================
cout<<"\n---------------- UPDATE -------------------------\n";

print_timer("RHS <= LHS comparison",t_compare);

print_timer("Prepare TwoMask",t_mask_prepare);

print_timer("Shifted_Overlaps update",t_overlap_update);

print_timer("sum_shifted_overlaps.AddMasked",t_sumL_mask);

print_timer("sum_c_k_mu &= mask",t_sumc_mask);

print_timer("fourSum preparation",t_fourSum);

print_timer("sum_shifted_overlaps subtraction",t_sumL_subtract);

print_timer("Spin XOR mask",t_spin_flip);


// ============================================================
// Output
// ============================================================
cout<<"\n---------------- OUTPUT -------------------------\n";

print_timer("SaveSpinConfigurations",t_save);

print_timer("Progress display",t_progress);

cout<<endl;

}