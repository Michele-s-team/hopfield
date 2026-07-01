//
//  hopfield_nobits.hpp
//  hopfield
//
//  Created by Bastien on 12/05/2026.
//
// THIS CLASS IMPLEMENTS THE CLASSIC SIMULATION OF THE HOPFIELD HAMILTONIAN ON THE GENERATED NETWORK

#ifndef hopfieldnobits_hpp
#define hopfieldnobits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "hopfield_model.hpp"

class HopfieldNoBits : public HopfieldModel {

public:
    using HopfieldModel::HopfieldModel;

    void evolve(gsl_rng*);
    void evolveIndependentRNG(gsl_rng*);
    void evolve_save(gsl_rng*, double freq, const std::string& filename);

    void runSweepsSharedRNG(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq);

private:

    int DeltaE(int spin, int realization);
    void DeltaE_all(int spin, vector<int>& delta_E);
};

#endif