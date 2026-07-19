//
//  hopfield_model.cpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//

#include "hopfield_model.hpp"

#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"
#include <algorithm>
using namespace std;

// =====================================================
// CONSTRUCTION
// =====================================================

HopfieldModel::HopfieldModel(int N, double beta, int N_sweeps, int P)
    : SimulationBase(N, beta, N_sweeps), P(P)
{
    // Initialize patterns and couplings
    patterns.assign(n_bits * N* P, 0);
    couplings.assign(N, {});
}

// =====================================================
// UTILITIES
// =====================================================

// Return the local index of 'target' in the neighbor list of 'spin'.
// Used to mirror couplings: if couplings[spin][k] = J, then
// couplings[target][neighbor_index(target, spin)] must also be set to J.
int HopfieldModel::neighbor_index(int spin, int target) const {
    const auto& nb = neighbors[spin];
    return find(nb.begin(), nb.end(), target) - nb.begin();
}

// =====================================================
// PATTERNS INITIALIZATION
// =====================================================

// Initialize P random patterns (±1 random value for each neuron of the network) 
// for the n_bits realizations
void HopfieldModel::initPatterns(gsl_rng* ran) {
    for (int mu = 0; mu < P; mu++){
        for (int i = 0; i < N; i++) {
            for (int r = 0; r < n_bits; r++)
                patterns[i * P * n_bits + mu * n_bits+ r] = randomSpin(ran);
        }
    }
    //initCouplings();
}

// =====================================================
// COUPLINGS INITIALIZATION 
// =====================================================

// Initializes the Couplings depending on the patterns (Hebb rule)
void HopfieldModel::initCouplings(){
    // Allocate one flat array per spin
    couplings.resize(N);
    for (int spin = 0; spin < N; ++spin) {
        couplings[spin].assign(neighbor_count[spin] * n_bits, 0);
    }

    // Hebb rule: compute each edge once (spin < nb), then mirror
    for (int spin = 0; spin < N; ++spin) {
        for (int k = 0; k < neighbor_count[spin]; ++k) {
            int nb = neighbors[spin][k];
            if (nb <= spin)
                continue;
            for (int r = 0; r < n_bits; ++r) {
                int J = 0;
                for (int mu = 0; mu < P; ++mu) {
                    J += patterns[spin * P * n_bits + mu * n_bits + r]
                       * patterns[nb   * P * n_bits + mu * n_bits + r];
                }
                couplings[spin][k * n_bits + r] = J;
            }
            // Mirror onto neighbour
            int k_mirror = neighbor_index(nb, spin);
            std::copy_n(
                &couplings[spin][k * n_bits],
                n_bits,
                &couplings[nb][k_mirror * n_bits]
            );
        }
    }
}


// Overwrite the patterns tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void HopfieldModel::initPatternsFromConfig(vector<int> config) {
    patterns = config;
    initCouplings();
}

// Return a copy of the full pattern tensor
vector<int> HopfieldModel::getPatterns() {
    return patterns;
}

// Return a copy of the full coupling tensor
vector<vector<int>> HopfieldModel::getCouplingsConfig() {
    return couplings;
}


void HopfieldModel::compute_overlaps(){
    for (int mu=0; mu<P; mu++){
        for (int r=0; r<n_bits; r++){
            for (int spin=0; spin<N; spin++){
                overlaps[mu *n_bits +r]+=spins_set[r*N+spin]*patterns[spin * P * n_bits + mu * n_bits+ r];
            }
        }
 
    }
}