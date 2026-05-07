//
//  spin_system.hpp
//  hopfield
//
//  Created by Bastien on 22/04/2026.
//

//THIS CLASS SETS UP THE NETWORK ON WHICH THE SIMULATION IS PERFORMED AND HANDLES VECTORS OF SPINS AND EXTRACTS OBSERVABLES
#ifndef spin_system_hpp
#define spin_system_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gsl_rng.h"

using namespace std;

class SpinSystem {
protected:
    int L;                              // linear size of the lattice (L x L)
    int N_spins;                        // total number of spins (L*L)
    vector<vector<int>> neighbors;      // neighbors[i] = liste of neigbhors of each spins
    vector<int> neighbor_count;         // number of neighbors for each spin
    vector<int> spins_set;              // spin configurations (±1), shape: [n_bits*N_spins]: [configurations 1 for all spins, configurations 2 for all spins ...]

public:
   
    SpinSystem(int L);                             // Constructor: allocates all vectors for an L x L lattice
    virtual ~SpinSystem() = default;
    void setSize(int); 
    void initNetwork2D_PBC();                  // Builds the periodic square lattice connectivity (4 neighbors per spin)
    void initNetwork2D_OBC();                  // Builds the square lattice connectivity with Open Boundary Conditions
    void initNetwork_random(gsl_rng*, double); // builds a random connectivity matrix with no self-connections
    int randomBinary(gsl_rng*);                      // initializes randomly a spin +-1
    void initSpins(gsl_rng*);                      // Initializes all spins randomly to ±1 using the GSL RNG
    void initSpinsFromConfig(vector<int>&);        // Initializes spins from a given configuration (deep copy)
    vector<int> getSpinsConfig();                  // Returns a copy of the full spin configuration
    virtual void GetMagnetizations(vector<double>&);       // Computes the magnetization m = (1/N) * sum_i s_i for each realisation
    double GetAverageMagnetization();              // Returns the average magnetization over all realisations    
    //void SaveMagnetizations(const string& filename); // Appends the magnetizations of all realisations to a CSV file
    void SaveSpins(const string& filename);        // Saves the spins configuration in n_bits different files
};

#endif