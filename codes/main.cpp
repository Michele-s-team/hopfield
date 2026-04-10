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

using namespace std;

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



#include <iostream>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

const int col_width = 3;   
const int sys_label_width = 12; // largeur fixée pour "System XX ; " + "before: "
const int prefix_width = 12;

const int N_neurons = 32;
const int J = 1;
const int Beta = 1;
const int N_steps = 100;

//CLASSIC IMPLEMENTATION
// connections_set: Matrix of size N_neurons*N_neurons connections are the same across all 64 realisations, encodes the connections on the networks (1 if there is a connection, 0 else), built at random
// neighbor_counts: List of size N_neurons, encodes the number of neighbors of each Neuron (identical across all realisations). No need for list of the netowrk is regular.
// neurons_set: Matrix of n_bits row and N_neurons columns, encodes the initial value of each neuron for each realisation


//BITSET IMPLEMENTATION
// connexions_set: same as for the classic case
// neighbor_counts: same as for the classic case
// Neurons_Set: list of BitSets of size N_neurons. Each BitSet encodes the value of each neuron for all initial conditions. BitSet of the list are technically bits (line of bits) as Neurons only take value -1/1 --> 0/1 in bitwise arithmetic


void init_neurons_set(vector<vector<int>>& neurons_set, gsl_rng* ran) {

    for (size_t r = 0; r < neurons_set.size(); r++) {
        for (int i = 0; i < N_neurons; i++) {
            neurons_set[r][i] = 2 * gsl_rng_uniform_int(ran, 2) - 1;
        }
    }

}

void init_connections_set(std::vector<std::vector<int>>& connections,
                          std::vector<int>& neighbor_counts,
                          gsl_rng* ran) {

    // Initialize neighbor counts to zero
    std::fill(neighbor_counts.begin(), neighbor_counts.end(), 0);

    for (int i = 0; i < N_neurons; i++) {
        for (int j = 0; j < N_neurons; j++) {

            if (i != j) {
                int connection = gsl_rng_uniform_int(ran, 2);

                // Direct 2D access
                connections[i][j] = connection;

                // Count outgoing neighbors of neuron i
                neighbor_counts[i] += connection;
            } else {
                // No self-connections
                connections[i][j] = 0;
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
                    const vector<vector<int>>& connections,
                    const vector<double>& random_numbers) {

    for (int step = 0; step < N_steps; step++) {
        double rho = random_numbers[step];

        for (size_t r = 0; r < n_bits; r++) {
            vector<int>& neurons = neurons_set[r];

            for (int i = 0; i < N_neurons; i++) {
                double dE = DeltaE(i, connections[i], neurons);
                if (rho >= dE) neurons[i] = -neurons[i];
            }
        }
    }
}

void evolve_systems_bits(vector<UnsignedInt>& Neurons_Set,
                    const vector<vector<int>>& connections,
                    const vector<int>& neighbor_count,
                    const vector<double>& random_numbers) {
    BitSet sum(1000); //max value of the sum that can be stored
    unsigned long long int somme=858;
    sum.SetAll(somme);
    BitSet and_ij(1);
    and_ij.SetAll(0);
    cout << "\nsum ";
    sum.Print("Initial value of the sum 858");
    cout << "\nand ";
    and_ij.Print("");
    cout << "\n";
    cout.flush();

    for (int step = 0; step < N_steps; step++) {
        cout<<"step: "<<step<<endl;
        cout.flush();
        for (int i = 0; i < N_neurons; i++) {
            cout<< "Neuron i: "<< i+1<<endl;
            if (random_numbers[step] >= neighbor_count[i]) {
                Neurons_Set[i].ComplementTo();
            } else {
                sum.SetAll(0);
                and_ij.SetAll(0);

                for (int j = 0; j < N_neurons; j++) {
                    cout<< "Neuron j: "<< j+1<<endl;
                    // Vérification
                    cout << "\nsum ";
                    sum.Print("");
                    cout << "\nand ";
                    and_ij.Print("");
                    cout << "\n";
                    if (i != j && connections[i][j]) {  
                        Neurons_Set[i].And(&Neurons_Set[j][0], &and_ij); //&Neurons_Set[j][0] extract the only Bits that constitutes the BitSet
                        sum += &and_ij;
                    }
                }
                Bits mask = sum < random_numbers[step]; //calculations yields the same comparison, even though the random variable changed
                cout<<"test mask creation"<<endl;
                Neurons_Set[i] ^= &mask;  // flips only where mask==1, work only if Neurons_Set[i] is actually a bits 
                cout<<"test mask application"<<endl;
            }
        }
    }
}


void print_neurons(const vector<vector<int>>& neurons_set_before,
                   const vector<vector<int>>& neurons_set,  
                   const vector<vector<int>>& neurons_set_bits,
                   int N_neurons, int prefix_width, int col_width) {

    for (size_t r = 0; r < neurons_set.size(); r++) {
        // Affichage before
        ostringstream oss_before;
        oss_before << "Realization" << right << setw(3) << r+1 << " ; before: ";
        cout << left << setw(prefix_width) << oss_before.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_before[r][i] << " ";
        cout << "\n";

        // Affichage after classic
        ostringstream oss_after;
        oss_after << "          " << " ;  after: ";
        cout << left << setw(prefix_width) << oss_after.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set[r][i] << " ";
        cout << "\n";

        // Affichage after bits
        ostringstream oss_after_bits;
        oss_after_bits << "          " << " ;   bits: ";
        cout << left << setw(prefix_width) << oss_after_bits.str();
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_set_bits[r][i] << " ";
        cout << "\n\n\n";
    }
}

int main() {

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);

    // Initialization of the neurons
    vector<vector<int>> neurons_set(n_bits, vector<int>(N_neurons));
    init_neurons_set(neurons_set, ran);

    vector<vector<int>> connections(N_neurons, vector<int>(N_neurons, 0));
    vector<int> neighbor_count(N_neurons, 0);  

    init_connections_set(connections, neighbor_count, ran);

    vector<UnsignedInt> Neurons_Set;
    Neurons_Set.reserve(N_neurons);


    for (int i = 0; i < N_neurons; i++) {
        UnsignedInt Neuron_tmp(1);
        for (int r = 0; r < n_bits; r++) {
            int bit = (neurons_set[r][i] + 1) / 2;
            Neuron_tmp.Set(r, bit);
        }
        Neurons_Set.push_back(Neuron_tmp);
        /*
        // Vérification
        cout << "Neuron " << i << " spins: ";
        for (int r = n_bits-1; r >= 0; r--) cout << (neurons_set[r][i] + 1) / 2;  // ordre inverse
        cout << "\nNeuron " << i << " bits: ";
        Neuron_tmp.Print("");
        cout << "\n";
        */
    }


    // Generation of a random sequence number
    vector<double> random_numbers(N_steps);
    for (int t = 0; t < N_steps; t++) {
        random_numbers[t] = gsl_ran_exponential(ran, 1.0 / (2.0 * Beta * J));
    }
    
    vector<vector<int>> neurons_set_before = neurons_set;

    //evolve_systems(neurons_set, connections, random_numbers);

    evolve_systems_bits(Neurons_Set, connections, neighbor_count, random_numbers);
    
    vector<vector<int>> neurons_set_after_bits(n_bits, vector<int>(N_neurons)); 

    for (int r=0; r<n_bits;r++){
        for (int i = 0; i < N_neurons; i++) {
            neurons_set_after_bits[r][i]=-1+2*Neurons_Set[i].Get(r);
        }
    }

    print_neurons(neurons_set_before, neurons_set, neurons_set_after_bits, N_neurons, prefix_width, col_width);
   /*
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

    print_neurons(neurons_set_before, neurons_set, neurons_set, N_neurons, prefix_width, col_width);
    */

    gsl_rng_free(ran);
    return 0;
}