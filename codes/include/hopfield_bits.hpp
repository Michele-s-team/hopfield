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
    vector<UnsignedInt> Degrees;     // degree encoded for vectorized ops
    vector<UnsignedInt> P_times_Degrees; // P*degree encoded for vectorized ops
    vector<Bits> Patterns;                     // patterns encoded for vectorized ops, index i * P + mu
    vector<vector<UnsignedInt>> Couplings;         // couplings encoded for vectorized ops
    vector<vector<UnsignedInt>> Couplings_nonNeighbors;
    vector<UnsignedInt> Shifted_Overlaps;        // Overlaps + N stored in an UnsignedInt list of length P


public:
    using HopfieldModel::HopfieldModel;

    //void initRandomNumbers(gsl_rng*);                                         // initialize RNG-based thresholds
    //void initRandomNumbersFromExp(const vector<vector<double>>& exp_base);    // initialize from external distribution

    void GetMagnetizations(vector<double>&) override;                           // get the n_bits magnetizations using the UnsignedInt formalism
    void GetSpinConfigurations(vector<vector<uint64_t>>& configs) override;     // get the n_bits configurations using the Bits formalism

    void initSpinsBits(gsl_rng* ran);
    void initPatternsBits(gsl_rng* ran);
    void initCouplingsBits();
    void initCouplings_nonNeighbors_Bits();

    void initSpinMetadataBits();

    void initSpinsFromConfigBits(const vector<Bits>& Config);
    void initPatternsFromConfigBits(const vector<Bits>& Config);
    vector<Bits> getPatternsBits();
    vector<vector<UnsignedInt>> getCouplingsConfigBits();
    vector<int> getPatternsBitsToCanonical();

    vector<Bits> corruptPattern(const vector<Bits>& Patterns, int mu, double flip_fraction, gsl_rng* ran) const; // Corrupt a single pattern by flipping each bit independently with
                                                                                                                      // probability flip_fraction.


    void compute_shifted_overlaps();
    bool check_shifted_overlaps_consistency(int sweep, int step);

    //void convertRandomNumbers();                                 // convert double RNG values → UnsignedInt bitwise format
    void toCanonical();                                            // convert Bits → ±1 spin representation
    void fromCanonical();                                          // convert ±1 spins → Bits representation
    
    void runSweeps(gsl_rng* ran, bool save, double freq, int burn_in);
    void runSweeps_neighbors(gsl_rng* ran, bool save, double freq, int burn_in);
    void runSweeps_overlaps(gsl_rng* ran, bool save, double freq, int burn_in);
    void runSweeps_pure_overlaps(gsl_rng* ran, bool save, double freq, int burn_in);
    
    void runSweeps_neighbors_DEBUG(gsl_rng* ran, bool save, double freq, int burn_in);
    void runSweeps_overlaps_DEBUG(gsl_rng* ran, bool save, double freq, int burn_in);
};

#endif
