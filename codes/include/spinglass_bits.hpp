//
//  spinglass_bits.hpp
//  hopfield
//
//  Created by Bastien on 7/05/2026.
//
// THIS CLASS IMPLEMENTS THE BITWISE SIMULATION OF THE SPINGLASS HAMILTONIAN ON THE GENERATED NETWORK
#ifndef spinglass_bits_hpp
#define spinglass_bits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "spinglass_model.hpp"
#include "unsigned_int.hpp"
using namespace std;


class SpinGlassBits : public SpinGlassModel {

    vector<Bits> Bits_Spins_Set;                 // bitwise spins across all realizations
    vector<UnsignedInt> Neighbor_Count;          // degree encoded for vectorized ops
    vector<vector<Bits>> Couplings;              // couplings encoded for vectorized ops

public:
    using SpinGlassModel::SpinGlassModel;

    void evolve(gsl_rng* ran);                                               // reference full-loop implementation
    void evolve_save(gsl_rng* ran, double freq, const string& filename);     // run simulation and save the magnetization at the frequency freq (between 0 and 1)
    void GetMagnetizations(vector<double>& magnetizations) override;         // get the n_bits magnetizations using the Bits formalism

private:
    void toCanonical();                                            // convert Bits → ±1 spin representation
    void fromCanonical();                                          // convert ±1 spins → Bits representation
    void runSweeps(gsl_rng* ran, bool save, double freq);
};

#endif