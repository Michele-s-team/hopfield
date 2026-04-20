//
//  isingmodel.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_model.hpp"
#include "gsl_rng.h"
#include "gsl_randist.h"

#include "gsl_math.h"
#include <vector>//
//  ising_bits.cpp
//  hopfield
//
//  Created by Bastien on 20/04/2026.
//

#include "ising_bits.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"

static constexpr int N_NEIGHBORS = 4;



// ──────────────────────────────────────────────
// Constructorr
// ──────────────────────────────────────────────
IsingModel::IsingModel(int L, double BJ, int N_sweeps)
    : L(L),
      N_neurons(L * L),
      N_sweeps(N_sweeps),
      BJ(BJ),
      connections(L * L, vector<int>(L * L, 0)),
      neighbor_count(L * L, 0),
      random_numbers(N_sweeps, vector<int>(L * L, 0)),
      neurons_set(n_bits, vector<int>(L * L, 0))  // ← ici
{}

void IsingModel::setBJ(double new_BJ) {
    BJ = new_BJ;
}

// ──────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────
void IsingModel::initConnections() {
    for (int y = 0; y < L; y++)
    for (int x = 0; x < L; x++) {
        int i = x + L * y;
        int nb[4] = {
            (x+1)%L + L*y,  (x-1+L)%L + L*y,
            x + L*((y+1)%L), x + L*((y-1+L)%L)
        };
        for (int k = 0; k < 4; k++) connections[i][nb[k]] = 1;
        neighbor_count[i] = N_NEIGHBORS;
    }
}

void IsingModel::initSpins(gsl_rng* ran) {
    for (int r = 0; r < n_bits; r++)
    for (int i = 0; i < N_neurons; i++)
        neurons_set[r][i] = 2 * gsl_rng_uniform_int(ran, 2) - 1;
}

void IsingModel::initSpinsFromConfig(const vector<vector<int>>& initial_set) {
    if (initial_set.size() != (size_t)n_bits) {
        cerr << "Erreur: nombre de réalisations incorrect !" << endl;
        return;
    }
    if (n_bits > 0 && initial_set[0].size() != (size_t)N_neurons) {
        cerr << "Erreur: la taille des réseaux de neurones ne correspond pas !" << endl;
        return;
    }

    // Copie profonde du vecteur de vecteurs
    neurons_set = initial_set;
}


// dans isingmodel.cpp
void IsingModel::initRandomNumbersFromExp(const vector<vector<double>>& exp_base) {
    for (int step = 0; step < N_sweeps; step++)
        for (int i = 0; i < N_neurons; i++)
            random_numbers[step][i] = (int) min(
                (double) N_neurons,
                1.0 / (2.0 * BJ) * exp_base[step][i]
            );
}

void IsingModel::initRandomNumbers(gsl_rng* ran) {
    for (int step = 0; step < N_sweeps; step++)
    for (int i = 0; i < N_neurons; i++)
        random_numbers[step][i] = (int) min(
            (double) N_neurons,
            1.0 / (2.0 * BJ) * gsl_ran_exponential(ran, 1.0)
        );
}

vector<vector<int>> IsingModel::getSpinsConfig() const {
    return neurons_set;
}



void IsingModel::init(gsl_rng* ran) {
    initConnections();
    initSpins(ran);
    initRandomNumbers(ran);
}

// ──────────────────────────────────────────────
// Getter
// ──────────────────────────────────────────────
const vector<vector<int>>& IsingModel::getState() const {
    return neurons_set;
}

vector<double> IsingModel::GetMagnetizations() {
    vector<double> magnetizations(n_bits);
    for (int r = 0; r < n_bits; r++) {
        double sum = 0;
        for (int i = 0; i < N_neurons; i++)
            sum += neurons_set[r][i];
        magnetizations[r] = sum / N_neurons;
    }
    return magnetizations;
}

void IsingModel::SaveMagnetizations(const string& filename) {
    vector<double> magnetizations = GetMagnetizations();
    ofstream file(filename, ios::app);
    
    file << BJ;
    for (int r = 0; r < n_bits; r++)
        file << "," << magnetizations[r];
    file << "\n";
}

double IsingModel::GetAverageMagnetization() {
    vector<double> magnetizations = GetMagnetizations();
    double sum = 0;
    for (double m : magnetizations)
        sum += m;
    return sum / n_bits;
}