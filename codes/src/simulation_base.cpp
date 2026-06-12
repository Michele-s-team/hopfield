//
//  simulation_base.cpp
//  hopfield
//
//  Created by Bastien on 02/06/2026.
//
//
#include "simulation_base.hpp"
#include "main.hpp"


// =====================================================
// CONSTRUCTION
// =====================================================

SimulationBase::SimulationBase(int N,
                                double beta,
                                int N_sweeps)
    : SpinSystem(N),
      m_metropolis(beta, N_sweeps)
{}

// =====================================================
// FILE MANAGEMENT — MAGNETIZATIONS
// =====================================================

void SimulationBase::OpenMagnetizationFiles(const string& folder){
    m_io.OpenMagnetizationFiles(folder, N, m_metropolis.getBeta());
}

void SimulationBase::CloseMagnetizationFiles(){
    m_io.CloseMagnetizationFiles();
}

// =====================================================
// FILE MANAGEMENT — SPIN CONFIGURATIONS
// =====================================================

void SimulationBase::OpenSpinFiles(const string& folder){
    m_io.OpenSpinFiles(folder, N, m_metropolis.getBeta());
}
void SimulationBase::CloseSpinFiles(){
    m_io.CloseSpinFiles();
}

// =====================================================
// GET MAGNETIZATIONS (implémentation par défaut)
// =====================================================

void SimulationBase::GetMagnetizations(vector<double>& magnetizations){
    magnetizations.resize(n_bits);
    for (int r = 0; r < n_bits; ++r){
        double sum = 0.0;
        for (int i = 0; i < N; ++i){
            sum += spins_set[r * N + i];
        }
        magnetizations[r] = sum / N;
    }
}

// =====================================================
// OBSERVABLE SAVING
// =====================================================

void SimulationBase::SaveMagnetizations(int sweep){
    vector<double> mags(n_bits);
    GetMagnetizations(mags);
    m_io.SaveMagnetizations(sweep, mags);
}

void SimulationBase::SaveSpinConfigurations(int sweep){
    m_io.SaveSpinConfigurations(N, sweep, spins_set);
}

void SimulationBase::SavePatterns(const string& folder,
                                   const vector<vector<vector<int>>>& patterns){
    m_io.SavePatterns(folder, N, patterns);
}

// =====================================================
// METROPOLIS
// =====================================================
int SimulationBase::randomNumber(gsl_rng* ran, int max_neighbor_count, int factor){
    return m_metropolis.randomNumber(ran, max_neighbor_count, factor);
}

int SimulationBase::getNSweeps() const {
    return m_metropolis.getNSweeps();
}

void SimulationBase::setNSweeps(int n){
    m_metropolis.setNSweeps(n);
}

void SimulationBase::setBeta(double new_beta){
    m_metropolis.setBeta(new_beta);
}