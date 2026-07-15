//
//  spin_system.cpp
//  hopfield
//
//  Created by Bastien on 22/04/2026.
//
#include "spin_system.hpp"

#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"
#include <math.h>

// =====================================================
// CONSTRUCTION
// =====================================================

SpinSystem::SpinSystem(int N)
    : N(N),
      neighbors(N),
      neighbor_count(N, 0),
      spins_set(n_bits * N, 0)
{}


void SpinSystem::setSize(int new_N) {
    N = new_N;
    
    neighbors.assign(N, vector<int>());
    neighbor_count.assign(N, 0);
    spins_set.assign(n_bits * N, 0);
}

// =====================================================
// NETWORK TOPOLOGY
// =====================================================

// 2D square lattice with periodic boundary conditions
void SpinSystem::initNetwork2D_PBC() {
    int L = sqrt(N);
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

// 2D square lattice with open boundary conditions (edge spins have fewer neighbors)
void SpinSystem::initNetwork2D_OBC() {
    int L = sqrt(N);
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
            int i = x + L * y;
            neighbors[i].clear();
            if (x + 1 < L) neighbors[i].push_back((x+1) + L*y);
            if (x - 1 >= 0) neighbors[i].push_back((x-1) + L*y);
            if (y + 1 < L) neighbors[i].push_back(x + L*(y+1));
            if (y - 1 >= 0) neighbors[i].push_back(x + L*(y-1));
            neighbor_count[i] = neighbors[i].size();
        }
}

// Fully connected network: every site is connected to every other site
void SpinSystem::initNetworkFullyConnected(){
    for (int i = 0; i < N; ++i) {
        neighbors[i].clear();
        neighbors[i].reserve(N - 1);
        for (int j = 0; j < N; ++j) {
            if (j != i)
                neighbors[i].push_back(j);
        }
        neighbor_count[i] = N - 1;
    }
}

// Erdos-Renyi random graph: each directed edge (i,j) included with probability p
void SpinSystem::initNetwork_random(gsl_rng* ran, double p) {
    neighbors.assign(N, vector<int>());
    fill(neighbor_count.begin(), neighbor_count.end(), 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j && gsl_rng_uniform(ran) < p) {
                neighbors[i].push_back(j);
                neighbor_count[i]++;
            }
}

// =====================================================
// SPIN INITIALIZATION
// =====================================================

// Draw a random spin value in {-1, +1}
int SpinSystem::randomSpin(gsl_rng* ran){
    return 2 * gsl_rng_uniform_int(ran, 2) - 1;
}

// Draw a random binary value in {0, 1}
int SpinSystem::randomBit(gsl_rng* ran){
    return gsl_rng_uniform_int(ran, 2);
}

// Initialize all spins randomly across all realizations
void SpinSystem::initSpins(gsl_rng* ran) {
    for (int i = 0; i < n_bits * N; i++)
        spins_set[i] = randomSpin(ran);
}

// Set spins from an externally provided configuration
void SpinSystem::initSpinsFromConfig(vector<int>& initial_set) {
    spins_set = initial_set;
}

// =====================================================
// OBSERVABLES
// =====================================================

vector<int> SpinSystem::getSpinsConfig() {
    return spins_set;
}

// Compute magnetization m = (1/N) Σ σ_i for each realization //might need to be moved to Ising nobits
void SpinSystem::GetMagnetizations(vector<double>& magnetizations) {
    for (int r = 0; r < n_bits; r++) {
        double sum = 0;
        for (int i = 0; i < N; i++) sum += spins_set[r*N+i];
        magnetizations[r] = sum / N;
    }
}

// Average magnetization over all realizations
double SpinSystem::GetAverageMagnetization() {
    vector<double> magnetizations(n_bits);
    GetMagnetizations(magnetizations);
    double sum = 0;
    for (double m : magnetizations) sum += m;
    return sum / n_bits;
}