//
//  ising_model.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//
//THIS CLASS SETS UP THE VARIABLES USED FOR THE SIMULATION OF THE THERMALIZATION OF AN ISING HAMILTONIAN ON THE GENERATED NETWORK
#ifndef ising_model_hpp
#define ising_model_hpp

#include "spin_system.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gsl_rng.h"

using namespace std;

class IsingModel : public SpinSystem {
protected:
    double BJ;                          // inverse temperature times coupling: β*J
    int N_sweeps;                       // number of Metropolis sweeps
    vector<vector<int>> random_numbers; // pre-generated random numbers for Metropolis, shape: [N_sweeps][N_neurons]
        
public:
    
    IsingModel(int L, double BJ, int N_sweeps);  // Constructor: allocates all vectors for an L x L lattice with given inverse temperature BJ and number of sweeps
    virtual ~IsingModel() = default;
    void setNSweeps(int); //sets the number of spins
    void setBJ(double);   // Sets a new value of β*J (e.g. when sweeping over temperatures)
    void initRandomNumbers(gsl_rng*);  // Pre-generates exponential random numbers for the Metropolis criterion using the GSL RNG    
    void initRandomNumbersFromExp(const vector<vector<double>>& exp_base); // Pre-generates random numbers from a pre-computed exponential base (useful to keep the same noise across different BJ values)
    void init(gsl_rng*); // Full initialization: spins + random numbers
    //virtual void evolveOneSweep() = 0;  // Performs one Metropolis sweep over all spins (implemented in subclasses)
    virtual void evolve_modular() = 0;  // Performs the full Metropolis simulation using evolveOneSweep() (implemented in subclasses)
    virtual void evolve_monolithic() = 0;  // Performs the full Metropolis simulation in one block (implemented in subclasses)
    void SaveMagnetizations(const string&);
};


#endif