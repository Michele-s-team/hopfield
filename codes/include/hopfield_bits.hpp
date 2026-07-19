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
    vector<UnsignedInt> P_times_Neighbor_Count; // P*degree encoded for vectorized ops
    vector<vector<Bits>> Patterns;          // patterns encoded for vectorized ops
    vector<vector<UnsignedInt>> Couplings;         // couplings encoded for vectorized ops
    vector<UnsignedInt> Shifted_Overlaps;        // Overlaps + N stored in an UnsignedInt list of length P


public:
    using HopfieldModel::HopfieldModel;

    //void initRandomNumbers(gsl_rng*);                                         // initialize RNG-based thresholds
    //void initRandomNumbersFromExp(const vector<vector<double>>& exp_base);    // initialize from external distribution
    //void initEvolveContext();                                                 // sync canonical ↔ bitwise + RNG prep
    void evolve(gsl_rng*);                              // run simulation and save only initial and final configurations
    void evolve_save(gsl_rng*, double freq);            // run simulation and save the configurations at the frequency freq (between 0 and 1)

    void evolve_save_bits(gsl_rng* ran, double freq);   
    void evolve_bits(gsl_rng* ran);
    
    void GetMagnetizations(vector<double>&) override;                           // get the n_bits magnetizations using the UnsignedInt formalism
    void GetSpinConfigurations(vector<vector<uint64_t>>& configs) override;     // get the n_bits configurations using the Bits formalism

    void initSpinsBits(gsl_rng* ran);
    void initPatternsBits(gsl_rng* ran);
    void initCouplingsBits();

    void initSpinMetadataBits();

    void initSpinsFromConfigBits(const vector<Bits>& Config);
    void initPatternsFromConfigBits(const vector<vector<Bits>>& Config);
    vector<vector<Bits>> getPatternsBits();
    vector<vector<UnsignedInt>> getCouplingsConfigBits();
    vector<vector<vector<int>>> getPatternsBitsToCanonical();

    vector<Bits> corruptPattern(const vector<Bits>& pattern, double flip_fraction, gsl_rng* ran) const; // Corrupt a single pattern by flipping each bit independently with
                                                                                                                      // probability flip_fraction.


    void compute_shifted_overalps();
    bool check_shifted_overlaps_consistency(int sweep, int step);

    //void convertRandomNumbers();                                 // convert double RNG values → UnsignedInt bitwise format
    void toCanonical();                                            // convert Bits → ±1 spin representation
    void fromCanonical();                                          // convert ±1 spins → Bits representation
    void runSweeps_DEBUG(gsl_rng* ran, bool save, double freq, int shift);
    void runSweeps(gsl_rng* ran, bool save, double freq, int shift);
    void runSweeps_overlaps(gsl_rng* ran, bool save, double freq, int shift);
    void runSweeps_overlaps_old(gsl_rng* ran, bool save, double freq, int shift);
    void runSweeps_old(gsl_rng*, bool save, double);
};

#endif
