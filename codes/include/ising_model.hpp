//
//  ising_model.hpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#ifndef ising_model_hpp
#define ising_model_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gsl_rng.h"

using namespace std;

//this class describes an Ising lattice thermalization using either Bitwise or classical implementation
class IsingModel {
protected:
    int L, N_neurons, N_sweeps;
    double BJ;
    vector<vector<int>> connections; //matrix of the connections in the lattice.
    vector<int>         neighbor_count;
    vector<vector<int>> random_numbers;
    vector<vector<int>> neurons_set;   // (spins ±1)

public:
    IsingModel(int L, double BJ, int N_steps);
    void setBJ(double);
    void initConnections();
    void initSpins(gsl_rng*);
    void initSpinsFromConfig(const vector<vector<int>>&);
    void initRandomNumbersFromExp(const vector<vector<double>>& exp_base);
    void initRandomNumbers(gsl_rng*);
    void init(gsl_rng*);           // initializes connection, neurons and random_numbers 
    vector<vector<int>> getSpinsConfig() const;
    const vector<vector<int>>& getState() const;
    virtual void evolve() = 0;
    virtual ~IsingModel() = default;
    vector<double> GetMagnetizations();
    void SaveMagnetizations(const string& filename);
    double GetAverageMagnetization();
};


#endif
