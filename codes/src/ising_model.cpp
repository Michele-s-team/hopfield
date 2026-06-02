//
//  ising_model.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//
#include "ising_model.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"

IsingModel::IsingModel(int N, double betaJ,int N_sweeps)
    : SimulationBase(N, betaJ, N_sweeps)
{}