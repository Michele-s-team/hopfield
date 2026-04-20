//
//  ising_bits.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#ifndef ising_bits_hpp
#define ising_bits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "ising_model.hpp"
#include "unsigned_int.hpp"
using namespace std;

//this class describes an Ising lattice thermalization using Bitwise implementation
class IsingBits : public IsingModel {
    vector<Bits>        Neurons_Set;
    vector<UnsignedInt> Neighbor_Count;
    vector<vector<UnsignedInt>> Random_Numbers;   // version bitwise
public:
    using IsingModel::IsingModel;
    void evolve() override;            // boucle bitwise
private:
    void toCanonical();    // Bits → neurons_set (±1) à la fin
    void fromCanonical();  // neurons_set → Bits au début
};


#endif
