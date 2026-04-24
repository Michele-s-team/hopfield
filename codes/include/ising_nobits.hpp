//
//  ising_nobits.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//
// THIS CLASS IMPLEMENTS THE CLASSIC SIMULATION OF THE ISING HAMILTONIAN ON THE GENERATED NETWORK

#ifndef isingnobits_hpp
#define isingnobits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "ising_model.hpp"

using namespace std;

class IsingNoBits : public IsingModel {

public:
    using IsingModel::IsingModel;

    //void evolveOneSweep(int, gsl_rng*);   // single Monte Carlo sweep (classic spin implementation)
    //void evolve_modular(gsl_rng*) override;    // modular version of classic evolution loop
    void evolve_monolithic(gsl_rng*) override; // reference full-loop implementation (non-modular)

private:
    double DeltaE(int neuron, int realization); // local energy variation for neuron flip decision
};


#endif
