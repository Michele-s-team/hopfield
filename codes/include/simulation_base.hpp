//
//  simulation_base.hpp
//  hopfield
//
//  Created by Bastien on 02/06/2026.
//
//

#ifndef SIMULATION_BASE_HPP
#define SIMULATION_BASE_HPP

#include "spin_system.hpp"
#include "metropolis.hpp"
#include "simulation_io.hpp"
#include <string>
#include <vector>

using namespace std;

class SimulationBase : public SpinSystem {
protected:

    Metropolis   m_metropolis;
    SimulationIO m_io;
    string       m_base_folder;

    
    static uint64_t PackBlock(const int* data, int start, int end);

public:

    static int num_blocks(int N);

    void SetBaseFolder(const string& folder);
        
    SimulationBase(int N,
                   double beta,
                   int N_sweeps);
    virtual ~SimulationBase() = default;

    // =========================
    // I/O layer
    // =========================

    void OpenMagnetizationFiles();
    void CloseMagnetizationFiles();
    void SaveMagnetizations(int sweep);

    void OpenSpinFiles();
    void CloseSpinFiles();
    void SaveSpinConfigurations(int sweep);

    void SavePatterns(const vector<int>& patterns, int P);
    
    static string format_beta(double beta);

    // =========================
    // Engine access (IMPORTANT)
    // =========================

    // Metropolis random threshold generator
    int randomNumber(gsl_rng* ran, int max_neighbor_count, int factor = 1);
    // Simulation length (if you want to remove N_sweeps from models)
    int getNSweeps() const;
    double getBeta() const;
    // Setters for Metropolis parameters
    void setNSweeps(int n);
    void setBeta(double new_beta);

    // =========================
    // Model interface
    // =========================

    // Implémentation par défaut pour NoBits, peut être override par Bits
    virtual void GetMagnetizations(vector<double>& magnetizations);
    virtual void GetSpinConfigurations(vector<vector<uint64_t>>& configs);
};
#endif