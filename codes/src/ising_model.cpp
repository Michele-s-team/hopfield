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
      N_sweeps(N_sweeps)
{}

// ising_model.cpp
void IsingModel::setNSweeps(int n) {
    N_sweeps = n;
}

void IsingModel::setBJ(double new_BJ) {
    BJ = new_BJ;
}

int IsingModel::randomNumber(gsl_rng* ran) {
    double val;
    val = min((double)N_neurons, 1.0 / (2.0 * BJ) * gsl_ran_exponential(ran, 1.0));
    return (int)val;

}
/*
void IsingModel::initrandomNumbers(gsl_rng* ran) {
    double val;
    for (int sweep = 0; sweep < N_sweeps; sweep++)
    for (int i = 0; i < N_neurons; i++){
        val = randomNumber(ran);
        random_numbers[sweep][i] = (int)val;
    }
}


void IsingModel::initRandomNumbersFromExp(const vector<vector<double>>& exp_base) {
    for (int sweep = 0; sweep < N_sweeps; sweep++)
    for (int i = 0; i < N_neurons; i++)
        random_numbers[sweep][i] = (int) min((double) N_neurons,1.0 / (2.0 * BJ) * exp_base[sweep][i]);
}
*/
/*
void IsingModel::init(gsl_rng* ran) {
    initSpins(ran);
    initrandomNumbers(ran);
}
*/


void IsingModel::SaveMagnetizations(const string& filename) {
    ofstream file(filename, ios::app);
    vector<double> magnetizations(n_bits);
    GetMagnetizations(magnetizations);
    file << BJ;
    for (int r = 0; r < n_bits; r++)
        file << "," << magnetizations[r];
    file << "\n";
}