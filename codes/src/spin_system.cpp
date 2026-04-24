//
//  spin_system.cpp
//  hopfield
//
//  Created by Bastien on 22/04/2026.
//

#include "spin_system.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"


SpinSystem::SpinSystem(int L)
    : L(L),
      N_neurons(L * L),
      neighbors(L * L),            
      neighbor_count(L * L, 0),
      neurons_set(n_bits * N_neurons, 0)
{}

void SpinSystem::initConnections2D() {
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
            int i = x + L * y;
            neighbors[i] = {
                (x+1)%L + L*y,
                (x-1+L)%L + L*y,
                x + L*((y+1)%L),
                x + L*((y-1+L)%L)
            };
            neighbor_count[i] = 4;
        }
}

void SpinSystem::initConnections_random(gsl_rng* ran, double p) {
    neighbors.assign(N_neurons, vector<int>());
    fill(neighbor_count.begin(), neighbor_count.end(), 0);

    for (int i = 0; i < N_neurons; i++)
        for (int j = 0; j < N_neurons; j++)
            if (i != j && gsl_rng_uniform(ran) < p) {
                neighbors[i].push_back(j);
                neighbor_count[i]++;
            }
}


int SpinSystem::randomSpin(gsl_rng* ran){
    return 2 * gsl_rng_uniform_int(ran, 2) - 1;
}

void SpinSystem::initSpins(gsl_rng* ran) {
    for (int i = 0; i < n_bits*N_neurons; i++)
        neurons_set[i] = randomSpin(ran);
}

void SpinSystem::initSpinsFromConfig(vector<int>& initial_set) {
    neurons_set = initial_set;
}

vector<int> SpinSystem::getSpinsConfig() {
    return neurons_set;
}

//magnetizations must have size n_bits
void SpinSystem::GetMagnetizations(vector<double>& magnetizations) {
    for (int r = 0; r < n_bits; r++) {
        double sum = 0;
        for (int i = 0; i < N_neurons; i++) sum += neurons_set[r*N_neurons+i];
        magnetizations[r] = sum / N_neurons;
    }
}

double SpinSystem::GetAverageMagnetization() {
    vector<double> magnetizations(n_bits);
    GetMagnetizations(magnetizations);
    double sum = 0;
    for (double m : magnetizations) sum += m;
    return sum / n_bits;
}

void SpinSystem::SaveMagnetizations(const string& filename) {
    // BJ n'est pas connu ici — à surcharger dans IsingModel si besoin
    ofstream file(filename, ios::app);
    vector<double> magnetizations(n_bits);
    GetMagnetizations(magnetizations);
    
    for (int r = 0; r < n_bits; r++)
        file << "," << magnetizations[r];
    file << "\n";
}