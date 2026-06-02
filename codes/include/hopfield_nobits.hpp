//
//  hopfield_nobits.hpp
//  hopfield
//
//  Created by Bastien on 12/05/2026.
//
// THIS CLASS IMPLEMENTS THE CLASSIC SIMULATION OF THE ISING HAMILTONIAN ON THE GENERATED NETWORK

#ifndef isingnobits_hpp
#define isingnobits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "hopfield_model.hpp"

class HopfieldNoBits : public HopfieldModel {

public:
    using HopfieldModel::HopfieldModel;

    void evolveSharedRNG(gsl_rng*);
    void evolveIndependentRNG(gsl_rng*);
    void evolveSharedRNG_save(gsl_rng*, double freq, const std::string& filename);

private:

    double DeltaE(int spin, int realization);

    void runSweepsSharedRNG(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq);
};

#endif