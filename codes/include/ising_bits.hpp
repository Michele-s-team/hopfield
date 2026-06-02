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
    
    void GetMagnetizations(std::vector<double>& magnetizations) override;
    void evolve(gsl_rng* ran);
    void evolve_save(gsl_rng* ran, double freq, const std::string& folder);
};

#endif