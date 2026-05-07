//
//  spinglass_bits.hpp
//  hopfield
//
//  Created by Bastien on 7/05/2026.
//
// THIS CLASS IMPLEMENTS THE BITWISE SIMULATION OF THE ISING HAMILTONIAN ON THE GENERATED NETWORK
#ifndef spinglass_bits_hpp
#define spinglass_bits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "spinglass_model.hpp"
#include "unsigned_int.hpp"
using namespace std;


class SpinglassBits : public SpinglassModel {

    vector<Bits> Bits_Spins_Set;                 // bitwise spins across all realizations
    vector<UnsignedInt> Neighbor_Count;     // degree encoded for vectorized ops
    vector<vector<Bits>> Couplings;     // couplings encoded for vectorized ops

public:
    using SpinglassModel::SpinglassModel;

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
    void tryFlip(int, int, Bits&, Bits&, UnsignedInt&, BitSet&);   // test a spin and flips it 
    void runSweeps(gsl_rng*, bool save, double);
};

#endif
