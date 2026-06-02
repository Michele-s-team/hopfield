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

public:

    SimulationBase(int N,
                   double beta,
                   int N_sweeps);

    virtual ~SimulationBase() = default;

    // =========================
    // I/O layer
    // =========================
    void OpenCSVFiles(const std::string& folder);

    void CloseCSVFiles();

    void SaveMagnetizations(int sweep);

    // =========================
    // Engine access (IMPORTANT)
    // =========================

    // Metropolis random threshold generator
    int randomNumber(gsl_rng* ran, int max_neighbor_count, int factor = 1);

    // Simulation length (if you want to remove N_sweeps from models)
    int getNSweeps() const;
    
    // Setters for Metropolis parameters
    void setNSweeps(int n);
    void setBeta(double new_beta);

    // =========================
    // Model interface
    // =========================

    // Implémentation par défaut pour NoBits, peut être override par Bits
    virtual void GetMagnetizations(vector<double>& magnetizations);
};

#endif