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
#include "simulation_io.hpp"
#include "main.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gsl_rng.h"

using namespace std;

class IsingModel : public SpinSystem {
protected:
    double betaJ;                       // inverse temperature times coupling: β*J
    double inv2betaJ;                   // =1/(2*betaJ)
    int N_sweeps;                       // number of Metropolis sweeps
    SimulationIO m_io;                  // saving files

private:
    vector<ofstream> m_csv_files; 
        
public:
    
    IsingModel(int N, double betaJ, int N_sweeps);   // Constructor: allocates all vectors for an L x L lattice with given inverse temperature betaJ and number of sweeps
    virtual ~IsingModel() = default;
    void setNSweeps(int);                         //sets the number of spins
    void setbetaJ(double);                           // Sets a new value of β*J (e.g. when sweeping over temperatures)
                        // Sets the size of the system
    int randomNumber(gsl_rng*);                   //generates a random number following an exponential distribution
    //void initrandomNumbers(gsl_rng*);           // Pre-generates exponential random numbers for the Metropolis criterion using the GSL RNG    
    //void initRandomNumbersFromExp(const vector<vector<double>>& exp_base); // Pre-generates random numbers from a pre-computed exponential base (useful to keep the same noise across different betaJ values)
    //void init(gsl_rng*);                        // Full initialization: spins + random numbers
   void OpenCSVFiles(const std::string& folder) {
        m_io.OpenCSVFiles(folder, L);
    }
    void CloseCSVFiles() { m_io.CloseCSVFiles(); }
    void SaveMagnetizations(int N);
};


#endif