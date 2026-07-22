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

    void runSweepsSharedRNG(gsl_rng* ran, bool save, double freq);
    void runSweepsSharedRNG_neighbors(gsl_rng* ran, bool save, double freq);
    void runSweepsSharedRNG_overlaps(gsl_rng* ran, bool save, double freq);
    void runSweepsSharedRNG_overlaps_non_neighbors(gsl_rng* ran, bool save, double freq);

    void runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG_neighbors(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG_overlaps(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG_overlaps_non_neighbors(gsl_rng* ran, bool save, double freq);

private:

    int DeltaE(int spin, int realization);
    int DeltaE_pure_overlaps(int spin, int realization);
    int DeltaE_overlaps_non_neighbors(int spin, int r);
   
    void DeltaE_neighbors_all(int spin, vector<int>& delta_E);
    void DeltaE_pure_overlaps_all(int spin, vector<int>& delta_E);
    void DeltaE_overlaps_non_neighbors_all(int spin, vector<int>& delta_E);
};

#endif