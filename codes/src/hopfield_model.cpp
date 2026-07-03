//
//  hopfield_model.cpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//

#include "hopfield_model.hpp"
#include "lib.hpp"
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
    patterns.assign(P, vector<vector<int>>(N, vector<int>(n_bits, 0)));
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
    for (int p = 0; p < P; p++){
        for (int i = 0; i < N; i++) {
            for (int r = 0; r < n_bits; r++)
                patterns[p][i][r] = randomSpin(ran);
        }
    }
    initCouplings();
}

// =====================================================
// COUPLINGS INITIALIZATION 
// =====================================================

// Initializes the Couplings depending on the patterns (Hebb rule)
void HopfieldModel::initCouplings(){
    // Allocate couplings: N spins, each with neighbor list size, times n_bits replicas
    for (int spin = 0; spin < N; ++spin) {
        couplings[spin].assign(
            neighbors[spin].size(),
            vector<int>(n_bits, 0)
        );
    }

    // Hebb rule: each edge computed once (spin < nb), then mirrored
    for (int spin = 0; spin < N; ++spin){
        for (int i = 0; i < (int)neighbors[spin].size(); ++i){
            int nb = neighbors[spin][i];

            if (nb <= spin)
                continue;  // skip already-filled edges

            auto& J = couplings[spin][i];

            for (int r = 0; r < n_bits; ++r){
                J[r] = 0;

                for (int p = 0; p < P; ++p)
                {
                    J[r] += patterns[p][spin][r] * patterns[p][nb][r];
                }
            }

            // Mirror onto neighbor
            couplings[nb][neighbor_index(nb, spin)] = J;
        }
    }
    cout << endl;
}


// Overwrite the patterns tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void HopfieldModel::initPatternsFromConfig(vector<vector<vector<int>>> config) {
    patterns = config;
    initCouplings();
}

// Return a copy of the full pattern tensor
vector<vector<vector<int>>> HopfieldModel::getPatterns() {
    return patterns;
}

// Return a copy of the full coupling tensor
vector<vector<vector<int>>> HopfieldModel::getCouplingsConfig() {
    return couplings;
}