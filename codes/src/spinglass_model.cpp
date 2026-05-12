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

// Return the local index of 'target' in the neighbor list of 'spin'
int SpinglassModel::neighbor_index(int spin, int target) const {
    const auto& nb = neighbors[spin];
    return find(nb.begin(), nb.end(), target) - nb.begin();
}

void SpinglassModel::initialize_couplings(gsl_rng* ran) {
    couplings.assign(L * L, {});
    for (int spin = 0; spin < L * L; ++spin)
        couplings[spin].assign(neighbors[spin].size(), std::vector<int>(n_bits));

    for (int spin = 0; spin < L * L; ++spin) {
        for (int i = 0; i < neighbors[spin].size(); ++i) {
            int nb = neighbors[spin][i];
            if (nb <= spin) continue;

            auto& J = couplings[spin][i];
            generate(J.begin(), J.end(), [&]{ return randomBinary(ran); });
            couplings[nb][neighbor_index(nb, spin)] = J; // mirror
        }
    }
}

void SpinglassModel::setNSweeps(int n) {
    N_sweeps = n;
}

void SpinglassModel::setbeta(double new_beta) {
    beta = new_beta;
    inv2beta = 1.0 / (2.0 * beta);  
}

// =====================================================
// RANDOM NUMBER GENERATION
// =====================================================

// Draw a Metropolis threshold from an exponential distribution:
//   rng ~ min(N_spins, Exp(1) / (2*betaJ))
int SpinglassModel::randomNumber(gsl_rng* ran) {
    return (int)min((double)neighbor_count[0], inv2beta*gsl_ran_exponential(ran, 1.0));
}
