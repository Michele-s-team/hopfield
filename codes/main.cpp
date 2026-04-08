#include <iostream>
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


const int N_sets = 50000;
const long long MAX_VALUE = pow(2,30);
const int N_neurons=10;
const int N_steps;
const int J=1;

double DeltaE(int neuron, vector<int> connection, vector<int> neurons){
    int sum=0

    for (int j=0; j<neurons.size();j++){
        if (connection[j]==1){sum+=neurons[j];}
    }
    return neurons[neuron]*sum
}

void actualization(vector<vector<int>> connections){
    for (int i=0; i<N_neurons;i++){
        DeltaE=DeltaE(i,connection[i], neurons)
        


    }

}

int main() {

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 12345);


    vector<int> Neurons(N_neurons); // ton vecteur de neurones
    for(int i = 0; i < N_neurons; i++) Neurons[i] = 2*gsl_rng_uniform_int(ran, 2)-1;

    // Exemple de matrice de connexions (adjacency matrix)
    vector<vector<int>> connections(N_neurons, vector<int>(N_neurons, 0));

    for(int i = 0; i < N_neurons; i++) {
        for(int j = 0; j < N_neurons; j++) {
            if (j!=i){connections[i][j] = gsl_rng_uniform_int(ran, 2);}
        }
    }

    vector<int> neighbors(N_neurons, 0);

    for(int i = 0; i < N_neurons; i++) {
        int count = 0;
        for(int j = 0; j < N_neurons; j++) {
            count += connections[i][j]; // additionne tous les 1 de la ligne i
        }
        neighbors[i] = count;
    }

        // Affichage
    for(int i = 0; i < N_neurons; i++){
        cout << "Neuron " << i << ": ";
        for(int j = 0; j < N_neurons; j++){
            cout << connections[i][j] << " ";
        }
        cout << endl;
    }
}