//
//  ising_model.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//
//THIS CLASS SETS UP THE VARIABLES USED FOR THE SIMULATION OF THE THERMALIZATION OF AN ISING HAMILTONIAN ON THE GENERATED NETWORK
#ifndef ising_model_hpp
#define ising_model_hpp

#include "simulation_base.hpp"
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <vector> 
#include "gsl_rng.h"

#include <vector>
#include "gsl_rng.h"

using namespace std;

class IsingModel : public SimulationBase {

public:

    IsingModel(int N,
               double betaJ,
               int N_sweeps);

    virtual ~IsingModel() = default;
};

#endif