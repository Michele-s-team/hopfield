//
//  spinglass_model.cpp
//  hopfield
//
//  Created by Bastien on 7/05/2026.
//
#include "spin_system.hpp"
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

SpinglassModel::SpinglassModel(int L, double beta, int N_sweeps)
    : SpinSystem(L), beta(beta), inv2beta(1.0 / (2.0 * beta)), N_sweeps(N_sweeps)
{}

// =====================================================
// SETTERS
// =====================================================

// Return the local index of 'target' in the neighbor list of 'spin'.
// Used to mirror couplings: if couplings[spin][k] = J, then
// couplings[target][neighbor_index(target, spin)] must also be set to J.
int SpinglassModel::neighbor_index(int spin, int target) const {
    const auto& nb = neighbors[spin];
    return find(nb.begin(), nb.end(), target) - nb.begin();
}

// Initialize random ±1 couplings for all edges of the lattice.
// Each edge (spin, nb) with spin < nb is generated once and then
// mirrored onto the neighbor side to guarantee J_ij = J_ji.
void SpinglassModel::initCouplings(gsl_rng* ran) {
    couplings.assign(L * L, {});

    // Allocate one vector of n_bits integers per neighbor of each spin
    for (int spin = 0; spin < L * L; ++spin)
        couplings[spin].assign(neighbors[spin].size(), std::vector<int>(n_bits));

    // Fill each edge exactly once (spin < nb), then mirror
    for (int spin = 0; spin < L * L; ++spin) {
        for (int i = 0; i < neighbors[spin].size(); ++i) {
            int nb = neighbors[spin][i];
            if (nb <= spin) continue;   // skip already-filled edges

            auto& J = couplings[spin][i];
            generate(J.begin(), J.end(), [&]{ return randomBinary(ran); });
            couplings[nb][neighbor_index(nb, spin)] = J; // mirror onto neighbor
        }
    }
}

// Return a copy of the full coupling tensor (used to share disorder
// between two model instances, e.g. SpinglassBits and SpinglassNoBits).
vector<vector<vector<int>>> SpinglassModel::getCouplingsConfig() {
    return couplings;
}

// Overwrite the coupling tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void SpinglassModel::initCouplingsFromConfig(vector<vector<vector<int>>> config) {
    couplings = config;
}

// Update the number of Monte Carlo sweeps to perform during the next run.
void SpinglassModel::setNSweeps(int n) {
    N_sweeps = n;
}

// Update the inverse temperature and recompute the derived prefactor inv2beta = 1/(2*beta),
// which scales the Metropolis threshold distribution.
void SpinglassModel::setbeta(double new_beta) {
    beta     = new_beta;
    inv2beta = 1.0 / (2.0 * beta);
}

// =====================================================
// RANDOM NUMBER GENERATION
// =====================================================

// Draw a Metropolis acceptance threshold for a single flip attempt.
// The threshold is sampled from an exponential distribution and capped
// at the maximum neighbor count:
//
//   rng = min(neighbor_count, Exp(1) / (2 * beta))
//
// A proposed flip is accepted if the local energy improvement exceeds rng,
// reproducing the standard Metropolis criterion in this parameterization.
int SpinglassModel::randomNumber(gsl_rng* ran) {
    return (int)min((double)neighbor_count[0],
                    inv2beta * gsl_ran_exponential(ran, 1.0));
}