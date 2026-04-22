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
      connections(L * L, vector<int>(L * L, 0)),
      neighbor_count(L * L, 0),
      neurons_set(n_bits, vector<int>(L * L, 0))
{}

void SpinSystem::initConnections() {
    for (int y = 0; y < L; y++)
    for (int x = 0; x < L; x++) {
        int i = x + L * y;
        int nb[4] = {
            (x+1)%L + L*y,   (x-1+L)%L + L*y,
            x + L*((y+1)%L), x + L*((y-1+L)%L)
        };
        for (int k = 0; k < 4; k++){
            connections[i][nb[k]] = 1;
            neighbor_count[i]+=1;
        }
    }
}

// Initialize a random connectivity matrix with no self-connections.
// p: probability that a directed edge (i->j) exists, in [0,1]
void SpinSystem::initConnections_random(gsl_rng* ran, double p) {
    fill(neighbor_count.begin(), neighbor_count.end(), 0);

    for (int i = 0; i < N_neurons; i++) {
        for (int j = 0; j < N_neurons; j++) {
            if (i != j) {
                int connection = (gsl_rng_uniform(ran) < p) ? 1 : 0;
                connections[i][j] = connection;
                neighbor_count[i] += connection;
            } else {
                connections[i][j] = 0;  // no self-connections
            }
        }
    }
}

void SpinSystem::initSpins(gsl_rng* ran) {
    for (int r = 0; r < n_bits; r++)
    for (int i = 0; i < N_neurons; i++)
        neurons_set[r][i] = 2 * gsl_rng_uniform_int(ran, 2) - 1;
}

void SpinSystem::initSpinsFromConfig(const vector<vector<int>>& initial_set) {
    if (initial_set.size() != (size_t)n_bits) {
        cerr << "Erreur: nombre de réalisations incorrect !" << endl;
        return;
    }
    if (n_bits > 0 && initial_set[0].size() != (size_t)N_neurons) {
        cerr << "Erreur: taille des réseaux de neurones incorrecte !" << endl;
        return;
    }
    neurons_set = initial_set;
}

vector<vector<int>> SpinSystem::getSpinsConfig() const {
    return neurons_set;
}
//magnetizations must have size n_bits
void SpinSystem::GetMagnetizations(std::vector<double>& magnetizations) {
    for (int r = 0; r < n_bits; r++) {
        double sum = 0;
        for (int i = 0; i < N_neurons; i++)
            sum += neurons_set[r][i];
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