//
//  spinglass_model.cpp
//  hopfield
//
//  Created by Bastien on 7/05/2026.
//

#include "spinglass_model.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"
#include <algorithm>
using namespace std;

// =====================================================
// CONSTRUCTION
// =====================================================

SpinGlassModel::SpinGlassModel(int N, double beta, int N_sweeps)
    : SimulationBase(N, beta, N_sweeps)
{
    // Initialize couplings: N spins, each with its neighbor list size, times n_bits replicas
    couplings.assign(N, {});
}

// =====================================================
// SETTERS
// =====================================================

// Return the local index of 'target' in the neighbor list of 'spin'.
int SpinGlassModel::neighbor_index(int spin, int target) const {
    const auto& nb = neighbors[spin];
    return find(nb.begin(), nb.end(), target) - nb.begin();
}

// Initialize random ±1 couplings for all edges of the lattice.
void SpinGlassModel::initCouplings(gsl_rng* ran) {
    // Allocate one vector of n_bits integers per neighbor of each spin
    for (int spin = 0; spin < N; ++spin)
        couplings[spin].assign(neighbors[spin].size(), vector<int>(n_bits));

    // Fill each edge exactly once (spin < nb), then mirror
    for (int spin = 0; spin < N; ++spin) {
        for (int i = 0; i < neighbors[spin].size(); ++i) {
            int nb = neighbors[spin][i];
            if (nb <= spin) continue;   // skip already-filled edges

            auto& J = couplings[spin][i];
            generate(J.begin(), J.end(), [&]{ return randomBinary(ran); });
            couplings[nb][neighbor_index(nb, spin)] = J; // mirror onto neighbor
        }
    }
}

// Return a copy of the full coupling tensor
vector<vector<vector<int>>> SpinGlassModel::getCouplingsConfig() {
    return couplings;
}

// Overwrite the coupling tensor with an externally provided configuration
void SpinGlassModel::initCouplingsFromConfig(vector<vector<vector<int>>> config) {
    couplings = config;
}