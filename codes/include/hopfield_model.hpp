//
//  hopfield_model.hpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//
// THIS CLASS SETS UP THE VARIABLES USED FOR THE SIMULATION OF THE THERMALIZATION 
// OF A HOPFIELD MODEL ON THE GENERATED NETWORK
#ifndef hopfield_model_hpp
#define hopfield_model_hpp

#include "simulation_base.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gsl_rng.h"

using namespace std;

class HopfieldModel : public SimulationBase {
protected:
    int P;                                    // number of patterns
    vector<int> patterns;                  // index : i * P * n_bits + mu * n_bits + r

    vector<vector<int>> couplings;         // couplings[i][k * n_bits + r]
    vector<int> overlaps;                  // overlaps[mu * n_bits + r]
        
public:
    
    // Constructor: allocates all vectors for system with given 
    // inverse temperature beta, number of sweeps, and number of patterns P
    HopfieldModel(int N, double beta, int N_sweeps, int P);
    
    virtual ~HopfieldModel() = default;
    
    // Patterns initialization
    void initPatterns(gsl_rng* ran);
    void initPatternsFromConfig(vector<int> config);
    vector<int> getPatterns();
    
    // Couplings initialization (from patterns)
    void initCouplings();
    vector<vector<int>> getCouplingsConfig();

    void compute_overlaps();

    int neighbor_index(int , int) const;
};

#endif