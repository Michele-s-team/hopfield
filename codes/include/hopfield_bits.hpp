//
//  hopfield_bits.hpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//
// THIS CLASS IMPLEMENTS THE BITWISE SIMULATION OF THE HOPFIELD HAMILTONIAN ON THE GENERATED NETWORK
#ifndef hopfield_bits_hpp
#define hopfield_bits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "hopfield_model.hpp"
#include "unsigned_int.hpp"
using namespace std;


class HopfieldBits : public HopfieldModel {

    vector<Bits> Bits_Spins_Set;            // bitwise spins across all realizations
    vector<UnsignedInt> Neighbor_Count;     // degree encoded for vectorized ops
    vector<UnsignedInt> Two_P_times_Neighbor_Count; // P*degree encoded for vectorized ops
    vector<vector<Bits>> Patterns;          // patterns encoded for vectorized ops
    vector<vector<Bits>> Couplings;         // couplings encoded for vectorized ops

public:
    using HopfieldModel::HopfieldModel;

    //void initRandomNumbers(gsl_rng*);                                      // initialize RNG-based thresholds
    //void initRandomNumbersFromExp(const vector<vector<double>>& exp_base); // initialize from external distribution
    //void initEvolveContext();                                               // sync canonical ↔ bitwise + RNG prep
    void evolve(gsl_rng*);                                               // reference full-loop implementation
    void evolve_save(gsl_rng*, double freq, const string& filename);    // run simulation and save the magnetization at the frequency freq (between 0 and 1)
    void GetMagnetizations(vector<double>&) override;    // get the n_bits magnetizations using the UnsignedInt formalism

private:

    //void convertRandomNumbers();                                 // convert double RNG values → UnsignedInt bitwise format
    void toCanonical();                                            // convert Bits → ±1 spin representation
    void fromCanonical();                                          // convert ±1 spins → Bits representation
    void runSweeps(gsl_rng*, bool save, double);
};

#endif
