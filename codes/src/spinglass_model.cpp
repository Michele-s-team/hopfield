//
//  spinglass_model.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//
#include "spin_system.hpp"
#include "spinglass_model.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"

// =====================================================
// CONSTRUCTION
// =====================================================

SpinglassModel::SpinglassModel(int L, double BJ, int N_sweeps)
    : SpinSystem(L), beta(beta), inv2beta(1.0 / (2.0 * beta)), N_sweeps(N_sweeps), couplings(L * L)
{}

// =====================================================
// SETTERS
// =====================================================

void SpinglassModel::initialize_couplings(gsl_rng* ran){
    for (int spin=0; spin<L*L; spin++){
        for (int i=0; i<neighbors[spin].size(); i++){
            couplings[spin][i]=randomBinary(ran);
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
//   rng ~ min(N_spins, Exp(1) / (2*BJ))
int SpinglassModel::randomNumber(gsl_rng* ran) {
    return (int)min((double)neighbor_count[0], inv2beta*gsl_ran_exponential(ran, 1.0));
}
/*
// Pre-generate all random thresholds for the full simulation
void SpinglassModel::initrandomNumbers(gsl_rng* ran) {
    double val;
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N_spins; i++){
            val = randomNumber(ran);
            random_numbers[sweep][i] = (int)val;
        }
}

// Initialize thresholds from an externally provided exponential base
void SpinglassModel::initRandomNumbersFromExp(const vector<vector<double>>& exp_base) {
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N_spins; i++)
            random_numbers[sweep][i] = (int) min((double) N_spins, 1.0 / (2.0 * BJ) * exp_base[sweep][i]);
}
*/

/*
void SpinglassModel::init(gsl_rng* ran) {
    initSpins(ran);
    initrandomNumbers(ran);
}
*/