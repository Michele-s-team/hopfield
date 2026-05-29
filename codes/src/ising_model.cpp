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

// =====================================================
// CONSTRUCTION
// =====================================================

IsingModel::IsingModel(int N, double betaJ, int N_sweeps)
    : SpinSystem(N), betaJ(betaJ), inv2betaJ(1.0 / (2.0 * betaJ)), N_sweeps(N_sweeps)
{}

// =====================================================
// SETTERS
// =====================================================

void IsingModel::setNSweeps(int n) {
    N_sweeps = n;
}

void IsingModel::setbetaJ(double new_betaJ) {
    betaJ = new_betaJ;
    inv2betaJ = 1.0 / (2.0 * new_betaJ);  
}

// =====================================================
// RANDOM NUMBER GENERATION
// =====================================================

// Draw a Metropolis threshold from an exponential distribution:
//   rng ~ min(N, Exp(1) / (2*betaJ))
int IsingModel::randomNumber(gsl_rng* ran) {
    return (int)min((double)neighbor_count[0], inv2betaJ*gsl_ran_exponential(ran, 1.0));
}
/*
// Pre-generate all random thresholds for the full simulation
void IsingModel::initrandomNumbers(gsl_rng* ran) {
    double val;
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N; i++){
            val = randomNumber(ran);
            random_numbers[sweep][i] = (int)val;
        }
}

// Initialize thresholds from an externally provided exponential base
void IsingModel::initRandomNumbersFromExp(const vector<vector<double>>& exp_base) {
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N; i++)
            random_numbers[sweep][i] = (int) min((double) N, 1.0 / (2.0 * betaJ) * exp_base[sweep][i]);
}
*/

/*
void IsingModel::init(gsl_rng* ran) {
    initSpins(ran);
    initrandomNumbers(ran);
}
*/

// =====================================================
// I/O
// =====================================================

void IsingModel::SaveMagnetizations(int N) {
        vector<double> mags(n_bits);
        GetMagnetizations(mags);
        m_io.SaveMagnetizations(N, betaJ, mags);
    }          // saves the magnetizations of all 64 realizations in a file with the speicified number of sweeps