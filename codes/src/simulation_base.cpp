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

void SimulationBase::SetBaseFolder(const string& folder){
    m_base_folder = folder;
}

// =====================================================
// UTILITIES
// =====================================================

int SimulationBase::num_blocks(int N){
    return (N + BLOCK_MASK) / BITS_PER_BLOCK;
}

uint64_t SimulationBase::PackBlock(const int* data, int start, int end){
    uint64_t packed = 0;
    for (int i = start; i < end; ++i){
        packed <<= 1;
        if (data[i] > 0)
            packed |= 1ULL;
    }
    return packed;
}

// =====================================================
// FILE MANAGEMENT — MAGNETIZATIONS
// =====================================================

void SimulationBase::OpenMagnetizationFiles(const string& subfolder){
    m_io.OpenMagnetizationFiles(m_base_folder + subfolder, N, m_metropolis.getBeta());
}

void SimulationBase::CloseMagnetizationFiles(){
    m_io.CloseMagnetizationFiles();
}

// =====================================================
// FILE MANAGEMENT — SPIN CONFIGURATIONS
// =====================================================

void SimulationBase::OpenSpinFiles(const string& subfolder){
    m_io.OpenSpinFiles(m_base_folder + subfolder, N, m_metropolis.getBeta());
}
void SimulationBase::CloseSpinFiles(){
    m_io.CloseSpinFiles();
}

// =====================================================
// GET OBSERVABLES
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

// Converts the +-1 spins into packed binary blocks, for each iteration
void SimulationBase::GetSpinConfigurations(vector<vector<uint64_t>>& configs){
    const int n_blocks = num_blocks(N);
    configs.assign(n_bits, vector<uint64_t>(n_blocks));

    vector<int> binary(N);
    for (int r = 0; r < n_bits; ++r){
        for (int i = 0; i < N; ++i)
            binary[i] = (spins_set[r * N + i] > 0) ? 1 : 0;

        for (int b = 0; b < n_blocks; ++b){
            int start = b * BITS_PER_BLOCK;
            int end   = min(N, start + BITS_PER_BLOCK);
            configs[r][b] = PackBlock(binary.data(), start, end);
        }
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
    vector<vector<uint64_t>> configs;
    GetSpinConfigurations(configs);
    m_io.SaveSpinConfigurations(sweep, configs);
}

void SimulationBase::SavePatterns(const string& subfolder,
                                   const vector<vector<vector<int>>>& patterns){
    m_io.SavePatterns(m_base_folder + subfolder, N, patterns);
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