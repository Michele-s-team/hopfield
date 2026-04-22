//
//  ising_model.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "spin_system.hpp"
#include "ising_model.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"

static constexpr int N_NEIGHBORS = 4;



IsingModel::IsingModel(int L, double BJ, int N_sweeps)
    : SpinSystem(L),
      BJ(BJ),
      N_sweeps(N_sweeps),
      random_numbers(N_sweeps, vector<int>(L * L, 0))
{}

void IsingModel::setBJ(double new_BJ) {
    BJ = new_BJ;
}

void IsingModel::initRandomNumbers(gsl_rng* ran) {
    for (int step = 0; step < N_sweeps; step++)
    for (int i = 0; i < N_neurons; i++)
        random_numbers[step][i] = (int) min(
            (double) N_neurons,
            1.0 / (2.0 * BJ) * gsl_ran_exponential(ran, 1.0)
        );
}

void IsingModel::initRandomNumbersFromExp(const vector<vector<double>>& exp_base) {
    for (int step = 0; step < N_sweeps; step++)
    for (int i = 0; i < N_neurons; i++)
        random_numbers[step][i] = (int) min(
            (double) N_neurons,
            1.0 / (2.0 * BJ) * exp_base[step][i]
        );
}

void IsingModel::init(gsl_rng* ran) {
    initConnections();
    initSpins(ran);
    initRandomNumbers(ran);
}
