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
    void evolve(gsl_rng*); // reference full-loop implementation (non-modular)
    void evolveIndependentRNG(gsl_rng*); // reference full-loop implementation (non-modular)
    void evolve_save(gsl_rng* ran, double freq, const string& filename);  // run simulation and save the magnetization at the frequency freq (between 0 and 1) in filename
    
    void runSweepsSharedRNG(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq);

private:

    int DeltaE(int spin, int realization); // local energy variation for spin flip decision
    void DeltaE_all(int spin, vector<int>& delta_E);
 
};


#endif
