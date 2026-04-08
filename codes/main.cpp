#include <iostream>
#include <iomanip> 
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <strstream>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <list>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
//this has to be uncommented to run on mesopsl
#include <stdint.h>
#include <chrono>

#include "gsl_rng.h"
#include "gsl_math.h"
#include "gsl_sf_log.h"
#include "gsl_randist.h"

#include "main.hpp"
#include "int.hpp"
#include "system_bits.hpp"
#include "system_nobits.hpp"

//all entries of BitSet_one are equal to 1
BitSet BitSet_one;
Bits Bits_one, Bits_zero;


#include <iostream>
#include <vector>
#include "gsl_rng.h"
#include "gsl_randist.h"

using namespace std;

#include <iostream>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


using namespace std;

const int col_width = 3;   
const int sys_label_width = 12; // largeur fixée pour "System XX ; " + "before: "
const int prefix_width = 12;

const int N_neurons = 32;
const int N_real=1;
const int J = 1;
const int Beta = 1;
const int N_steps = 100;


void init_neurons_set(vector<vector<int>>& neurons_set, gsl_rng* ran) {

    for (size_t sys = 0; sys < neurons_set.size(); sys++) {
        for (int i = 0; i < N_neurons; i++) {
            neurons_set[sys][i] = 2 * gsl_rng_uniform_int(ran, 2) - 1;
        }
    }

}

void init_connections_set(std::vector<std::vector<int>>& connections_set, gsl_rng* ran) {

    for (size_t sys = 0; sys < connections_set.size(); sys++) {
        for (int i = 0; i < N_neurons; i++) {
            for (int j = 0; j < N_neurons; j++) {
                if (i != j) {
                    connections_set[sys][i * N_neurons + j] = gsl_rng_uniform_int(ran, 2);
                }
            }
        }
    }

}



// DeltaE for a single neuron
double DeltaE(int neuron, const vector<int>& connection, const vector<int>& neurons) {
    int sum = 0;
    for (size_t j = 0; j < neurons.size(); j++) {
        if (connection[j] == 1) sum += neurons[j];
    }
    return neurons[neuron] * sum;
}

// Evolution of multiple neuron systems with pre-generated random numbers
void evolve_systems(vector<vector<int>>& neurons_set,
                    const vector<vector<int>>& connections_set,
                    const vector<double>& random_numbers) {

    for (int step = 0; step < N_steps; step++) {
        double rho = random_numbers[step]; // same rho for all systems at this step

        for (size_t sys = 0; sys < n_bits; sys++) {
            vector<int>& neurons = neurons_set[sys];
            const vector<int>& connections = connections_set[sys];

            for (int i = 0; i < N_neurons; i++) {
                double dE = DeltaE(i, connections, neurons);
                if (rho >= dE) neurons[i] = -neurons[i];
            }
        }
    }
}

void print_neurons(const vector<vector<int>>& neurons_set,
                   const vector<vector<int>>& neurons_set_before,
                   int N_neurons, int prefix_width, int col_width) {

    for (size_t sys = 0; sys < neurons_set.size(); sys++) {
        // Affichage before
        ostringstream oss_before;
        oss_before << "System " << right << setw(3) << sys+1 << " ; before: ";
        cout << left << setw(prefix_width) << oss_before.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_before[sys][i] << " ";
        cout << "\n";

        // Affichage after
        ostringstream oss_after;
        oss_after << "          " << " ;  after: ";
        cout << left << setw(prefix_width) << oss_after.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set[sys][i] << " ";
        cout << "\n\n\n";
    }
}

int main() {

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);

    // Initialization of the neurons
    vector<vector<int>> neurons_set(n_bits, vector<int>(N_neurons));
    init_neurons_set(neurons_set, ran);

    vector<vector<int>> connections_set(n_bits, vector<int>(N_neurons * N_neurons, 0));
    init_connections_set(connections_set, ran);


    vector<UnsignedInt> Neurons_Set;
    Neurons_Set.reserve(N_neurons * N_real);


    for (int i = 0; i < N_neurons; i++) {
        UnsignedInt Neuron_tmp(1);
        for (int r = 0; r < n_bits; r++) {
            int bit = (neurons_set[r][i] + 1) / 2;
            Neuron_tmp.Set(r, bit);
        }
        Neurons_Set.push_back(Neuron_tmp);

        // Vérification
        cout << "Neuron " << i << " spins: ";
        for (int r = n_bits-1; r >= 0; r--) cout << (neurons_set[r][i] + 1) / 2;  // ordre inverse
        cout << "\nNeuron " << i << " bits: ";
        Neuron_tmp.Print("");
        cout << "\n";
    }

    // Generation of a random sequence number
    vector<double> random_numbers(N_steps);
    for (int t = 0; t < N_steps; t++) {
        random_numbers[t] = gsl_ran_exponential(ran, 1.0 / (2.0 * Beta * J));
    }

    vector<vector<int>> neurons_set_before = neurons_set;

    evolve_systems(neurons_set, connections_set, random_numbers);

    print_neurons(neurons_set, neurons_set_before, N_neurons, prefix_width, col_width);
    gsl_rng_free(ran);
    return 0;
}