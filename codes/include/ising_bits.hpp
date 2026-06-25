//
//  ising_bits.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//
// THIS CLASS IMPLEMENTS THE BITWISE SIMULATION OF THE ISING HAMILTONIAN ON THE GENERATED NETWORK

#ifndef ising_bits_hpp
#define ising_bits_hpp

#include "ising_model.hpp"
#include "bits.hpp"
#include "unsigned_int.hpp"
#include <vector>

class IsingBits : public IsingModel {
private:
    std::vector<Bits> Bits_Spins_Set;
    std::vector<UnsignedInt> Neighbor_Count;
    
    void fromCanonical();
    void toCanonical();
    void runSweeps(gsl_rng* ran, bool save, double freq);

public:

    IsingBits(int N, double betaJ, int N_sweeps) 
        : IsingModel(N, betaJ, N_sweeps) {}
    
    virtual ~IsingBits() = default;
    
    void GetMagnetizations(std::vector<double>& magnetizations) override;    // get the n_bits magnetizations using the Bits formalism
    void GetSpinConfigurations(vector<vector<uint64_t>>& configs) override;  // get the n_bits configurations using the Bits formalism

    void initSpinsBits(gsl_rng* ran);

    void evolve(gsl_rng* ran);
    void evolve_save(gsl_rng* ran, double freq, const std::string& folder);

    void evolve_bits(gsl_rng* ran, const string& filename);
    void evolve_save_bits(gsl_rng* ran, double freq, const string& filename);
};

#endif