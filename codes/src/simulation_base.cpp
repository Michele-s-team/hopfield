//
//  simulation_base.cpp
//  hopfield
//
//  Created by Bastien on 02/06/2026.
//

#include "simulation_base.hpp"

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
// FILE MANAGEMENT
// =====================================================

void SimulationBase::OpenCSVFiles(
    const string& folder) {

    m_io.OpenCSVFiles(folder, N);
}

void SimulationBase::CloseCSVFiles() {
    m_io.CloseCSVFiles();
}

// =====================================================
// OBSERVABLE SAVING
// =====================================================

void SimulationBase::SaveMagnetizations(
    int sweep) {

    vector<double> mags(n_bits);

    GetMagnetizations(mags);

    m_io.SaveMagnetizations(
        sweep,
        m_metropolis.getBeta(),
        mags);
}

// =====================================================
// METROPOLIS
// =====================================================

int SimulationBase::randomNumber(gsl_rng* ran, int max_neighbor_count, int factor) {
    return m_metropolis.randomNumber(ran, max_neighbor_count, factor);
}

int SimulationBase::getNSweeps() const {
    return m_metropolis.getNSweeps();  // si Metropolis a cette méthode
}