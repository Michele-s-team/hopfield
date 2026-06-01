//
//  hopfield_model.cpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//
#include "spin_system.hpp"
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
    : SpinSystem(N), beta(beta), inv2beta(1.0 / (2.0 * beta)), N_sweeps(N_sweeps), P(P)
{}

// =====================================================
// SETTERS
// =====================================================

// Return the local index of 'target' in the neighbor list of 'spin'.
// Used to mirror couplings: if couplings[spin][k] = J, then
// couplings[target][neighbor_index(target, spin)] must also be set to J.
int HopfieldModel::neighbor_index(int spin, int target) const {
    const auto& nb = neighbors[spin];
    return find(nb.begin(), nb.end(), target) - nb.begin();
}

//Initializes the Couplings depending on the patterns
void HopfieldModel::initCouplings() {
    couplings.assign(N, {});
    for (int spin = 0; spin < N; ++spin)
        couplings[spin].assign(neighbors[spin].size(), std::vector<int>(n_bits, 0));

    // Hebb rule, each edge once (spin < nb), then mirror
    for (int spin = 0; spin < N; ++spin) {
        for (int i = 0; i < (int)neighbors[spin].size(); ++i) {
            int nb = neighbors[spin][i];
            if (nb <= spin) continue;
            auto& J = couplings[spin][i];
            for (int r = 0; r < n_bits; ++r)
                for (int p = 0; p < P; ++p)
                    J[r] += patterns[p][spin][r] * patterns[p][nb][r];
            couplings[nb][neighbor_index(nb, spin)] = J; // mirror
        }
    }
}

// Initialize P random patterns (±1 random value for each neuron of the network) for the n_bits realizations
void HopfieldModel::initPatterns(gsl_rng* ran) {
    // patterns[p][i][r] : pattern p, neurone i, realization r
    patterns.assign(P, vector<vector<int>>(N, vector<int>(n_bits)));
    for (int p = 0; p < P; p++)
        for (int i = 0; i < N; i++)
            for (int r = 0; r < n_bits; r++)
                patterns[p][i][r] = randomBinary(ran);

    initCouplings();
}


// Return a copy of the full pattern tensor (used to share disorder
// between two model instances, e.g. HopfieldBits and HopfieldNoBits).
vector<vector<vector<int>>> HopfieldModel::getPatternsConfig() {
    return patterns;
}

// Return a copy of the full pattern tensor (used to share disorder
// between two model instances, e.g. HopfieldBits and HopfieldNoBits).
vector<vector<vector<int>>> HopfieldModel::getCouplingsConfig() {
    return couplings;
}




// Overwrite the patterns tensor with an externally provided configuration.
// Allows two model instances to share the exact same disorder realization.
void HopfieldModel::initPatternsFromConfig(vector<vector<vector<int>>> config) {
    patterns = config;
    initCouplings();
}


// Update the number of Monte Carlo sweeps to perform during the next run.
void HopfieldModel::setNSweeps(int n) {
    N_sweeps = n;
}

// Update the inverse temperature and recompute the derived prefactor inv2beta = 1/(2*beta),
// which scales the Metropolis threshold distribution.
void HopfieldModel::setbeta(double new_beta) {
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
int HopfieldModel::randomNumber(gsl_rng* ran) {
    return (int)min((double)neighbor_count[0],
                    inv2beta * gsl_ran_exponential(ran, 1.0));
}