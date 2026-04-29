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

IsingModel::IsingModel(int L, double BJ, int N_sweeps)
    : SpinSystem(L),
      BJ(BJ),
      N_sweeps(N_sweeps)
{}

// =====================================================
// SETTERS
// =====================================================

void IsingModel::setNSweeps(int n) {
    N_sweeps = n;
}

void IsingModel::setBJ(double new_BJ) {
    BJ = new_BJ;
}

// =====================================================
// RANDOM NUMBER GENERATION
// =====================================================

// Draw a Metropolis threshold from an exponential distribution:
//   rng ~ min(N_spins, Exp(1) / (2*BJ))
int IsingModel::randomNumber(gsl_rng* ran) {
    double val = min((double) neighbor_count[0], 1.0 / (2.0 * BJ) * gsl_ran_exponential(ran, 1.0));
    return (int)val;
}

/*
// Pre-generate all random thresholds for the full simulation
void IsingModel::initrandomNumbers(gsl_rng* ran) {
    double val;
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N_spins; i++){
            val = randomNumber(ran);
            random_numbers[sweep][i] = (int)val;
        }
}

// Initialize thresholds from an externally provided exponential base
void IsingModel::initRandomNumbersFromExp(const vector<vector<double>>& exp_base) {
    for (int sweep = 0; sweep < N_sweeps; sweep++)
        for (int i = 0; i < N_spins; i++)
            random_numbers[sweep][i] = (int) min((double) N_spins, 1.0 / (2.0 * BJ) * exp_base[sweep][i]);
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

// Open one CSV file per realization, writing header if the file is new
void IsingModel::OpenCSVFiles(const string& filename) {
    m_csv_files.resize(n_bits);
    for (int r = 0; r < n_bits; ++r) {
        string file_path = filename + "_r" + to_string(r) + ".csv";

        ifstream test(file_path);
        bool file_exists = test.good();
        test.close();

        m_csv_files[r].open(file_path, ios::app);
        if (m_csv_files[r] && !file_exists)
            m_csv_files[r] << "T,N,m\n";
    }
}

// Flush and close all open CSV files
void IsingModel::CloseCSVFiles() {
    for (auto& f : m_csv_files)
        if (f.is_open()) f.close();
    m_csv_files.clear();
}

// Append current magnetizations to each realization's CSV file
void IsingModel::SaveMagnetizations(int N) {
    vector<double> magnetizations(n_bits);
    GetMagnetizations(magnetizations);
    double T = 1.0 / BJ;

    for (int r = 0; r < n_bits; ++r) {
        if (!m_csv_files[r]) continue;
        m_csv_files[r] << T << "," << N << "," << magnetizations[r] << "\n";
    }
}