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

    // Automatic choice of the formulation (alpha + 0.5 <= rho => overlaps, else neighbors)
    void runSweepsSharedRNG(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq);

    // Explicit formulations (e.g. to compare each against its bitwise equivalent).
    // _overlaps computes the overlaps itself, then picks pure overlaps (complete graph)
    // or overlaps + non-neighbor couplings (non-complete graph).
    void runSweepsSharedRNG_overlaps(gsl_rng* ran, bool save, double freq);
    void runSweepsSharedRNG_neighbors(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG_overlaps(gsl_rng* ran, bool save, double freq);
    void runSweepsIndependentRNG_neighbors(gsl_rng* ran, bool save, double freq);

private:

    enum class SweepMode { NEIGHBORS, OVERLAPS, OVERLAPS_NON_NEIGHBORS };

    void DeltaE_all(int spin, SweepMode mode, vector<int>& delta_E);
    void FlipSpin(int spin, int r, SweepMode mode);

    void runSweepsSharedRNG_core(gsl_rng* ran, bool save, double freq, SweepMode mode);
    void runSweepsIndependentRNG_core(gsl_rng* ran, bool save, double freq, SweepMode mode);
};

#endif