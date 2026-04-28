//
//  ising_bits.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//
// THIS CLASS IMPLEMENTS THE BITWISE SIMULATION OF THE ISING HAMILTONIAN ON THE GENERATED NETWORK
#ifndef ising_bits_hpp
#define ising_bits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "ising_model.hpp"
#include "unsigned_int.hpp"
using namespace std;


class IsingBits : public IsingModel {

    vector<Bits> spins_Set;                 // bitwise spins across all realizations
    vector<UnsignedInt> Neighbor_Count;     // degree encoded for vectorized ops

public:
    using IsingModel::IsingModel;

    //void initRandomNumbers(gsl_rng*);                                      // initialize RNG-based thresholds
    //void initRandomNumbersFromExp(const vector<vector<double>>& exp_base); // initialize from external distribution
    //void initEvolveContext();                                               // sync canonical ↔ bitwise + RNG prep
    void evolve(gsl_rng*);                      // reference full-loop implementation
    void evolve_save(gsl_rng*, double freq);    // run simulation and save the magnetization at the frequency freq (between 0 and 1)
    void GetMagnetizations(vector<double>&);    // get the n_bits magnetizations using the UnsignedInt formalism

private:

    //void convertRandomNumbers();                                 // convert double RNG values → UnsignedInt bitwise format
    void toCanonical();                                            // convert Bits → ±1 spin representation
    void fromCanonical();                                          // convert ±1 spins → Bits representation
    void tryFlip(int, int, Bits&, Bits&, UnsignedInt&, BitSet&);   // test a spin and flips it 
    void runSweeps(gsl_rng*, bool save, double);
};

#endif
