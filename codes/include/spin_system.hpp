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
    int N;                              // total number of spins
    vector<vector<int>> neighbors;      // neighbors[i] = liste of neighbors of each spins
    vector<vector<int>> non_neighbors;  // non neighbors of i
    vector<int> degrees;         // number of neighbors for each spin
    vector<int> spins_set;              // spin configurations (±1), shape: [N * n_bits]: [configurations 1 for all spins, configurations 2 for all spins ...]

public:
   
    SpinSystem(int L);                             // Constructor: allocates all vectors for an L x L lattice
    virtual ~SpinSystem() = default;
    void setSize(int); 
    void initNetwork_2D_PBC();                  // Builds the periodic square lattice connectivity (4 neighbors per spin)
    void initNetwork_2D_OBC();                  // Builds the square lattice connectivity with Open Boundary Conditions
    void initNetwork_FullyConnected();          // Builds a fully connected network
    void initNetwork_random(gsl_rng*, double); // builds a random connectivity matrix with no self-connections
    void computeNonNeighbors();
    
    int randomSpin(gsl_rng*);                      // initializes randomly a spin +-1
    int randomBit(gsl_rng*);                       // initializes randomly a bit 0/1
    void initSpins(gsl_rng*);                      // Initializes all spins randomly to ±1 using the GSL RNG
    void initSpinsFromConfig(vector<int>&);        // Initializes couplings from a given configuration (deep copy)
    vector<int> getSpinsConfig();                  // Returns a copy of the full couplings configuration
    virtual void GetMagnetizations(vector<double>&);       // Computes the magnetization m = (1/N) * sum_i s_i for each realisation
    double GetAverageMagnetization();              // Returns the average magnetization over all realisations    
    //void SaveMagnetizations(const string& filename); // Appends the magnetizations of all realisations to a CSV file
    void SaveSpins(const string& filename);        // Saves the spins configuration in n_bits different files
};

#endif