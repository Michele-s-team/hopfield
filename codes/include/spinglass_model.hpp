//
//  spinglass_model.hpp
//  hopfield
//
//  Created by Bastien on 7/05/2026.
//
// THIS CLASS SETS UP THE VARIABLES USED FOR THE SIMULATION OF THE THERMALIZATION 
// OF A SPIN GLASS HAMILTONIAN ON THE GENERATED NETWORK
#ifndef spinglass_model_hpp
#define spinglass_model_hpp

#include "simulation_base.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gsl_rng.h"

using namespace std;

class SpinGlassModel : public SimulationBase {
protected:
    vector<vector<vector<int>>> couplings;  // couplings[r][i][j] for each replica
        
public:
    
    // Constructor: allocates all vectors for system with given 
    // inverse temperature beta and number of sweeps
    SpinGlassModel(int N,
                   double beta,
                   int N_sweeps);
    
    virtual ~SpinGlassModel() = default;
    
    // Couplings initialization
    void initCouplings(gsl_rng* ran);
    void initCouplingsFromConfig(vector<vector<vector<int>>> config);
    vector<vector<vector<int>>> getCouplingsConfig();

    int neighbor_index(int , int) const;
};

#endif