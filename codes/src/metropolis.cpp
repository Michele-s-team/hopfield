//
//  metropolis.cpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//

#include "metropolis.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"

#include <algorithm>

using namespace std;

// =====================================================
// CONSTRUCTION
// =====================================================

Metropolis::Metropolis(double beta,
                       int N_sweeps)
    : beta(beta),
      inv2beta(1.0 / (2.0 * beta)),
      N_sweeps(N_sweeps)
{}

// =====================================================
// SETTERS
// =====================================================

void Metropolis::setNSweeps(int n) {
    N_sweeps = n;
}

void Metropolis::setBeta(double new_beta) {
    beta     = new_beta;
    inv2beta = 1.0 / (2.0 * new_beta);
}

// =====================================================
// RANDOM NUMBER GENERATION
// =====================================================

int Metropolis::randomNumber(gsl_rng* ran, int max_neighbor_count, double factor) {
    return static_cast<int>(
        min(static_cast<double>(max_neighbor_count),
            factor * inv2beta * gsl_ran_exponential(ran, 1.0))
    );
}