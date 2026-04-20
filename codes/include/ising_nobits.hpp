//
//  ising_nobits.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#ifndef isingnobits_hpp
#define isingnobits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "ising_model.hpp"
using namespace std;

//this class describes an Ising lattice thermalization using classical implementation
class IsingNoBits : public IsingModel {
public:
    using IsingModel::IsingModel;
    void evolve() override;            // boucle classique
private:
    double DeltaE(int neuron, int realization);
};


#endif
