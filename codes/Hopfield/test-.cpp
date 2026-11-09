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
#include <stdint.h>
#include <chrono>
#include <sys/stat.h> // for mkdir

using namespace std;

#include "gsl_rng.h"
#include "gsl_math.h"
#include "gsl_sf_log.h"
#include "gsl_randist.h"

#include "main.hpp"
#include "int.hpp"

#include "ising_model.hpp"
#include "ising_nobits.hpp"
#include "ising_bits.hpp"

#include "spinglass_model.hpp"
#include "spinglass_nobits.hpp"
#include "spinglass_bits.hpp"

#include "hopfield_model.hpp"
#include "hopfield_nobits.hpp"
#include "hopfield_bits.hpp"

//g++ min_working_example_hopfield.cpp src/*.cpp -llapack -lgsl -lgslcblas -lm -O3 -flto -Wno-deprecated -Iinclude -I/usr/include/gsl -DHAVE_INLINE -o main.o


BitSet BitSet_one; // really strange that we need to define this for the operator -= of BitSet 

// =============================================================================
// Alpha sweep — HopfieldBits only
//
// For each value of alpha between 0 and 0.20 (step 0.01), a new base
// directory "results/alpha_XX/" is created, the network and patterns are
// initialized, and the model is evolved and saved into that directory.
// =============================================================================

// ──────────────────────────────────────────────────────────────────────────────
// make_dir
//
void make_dir(const string& path) {
    mkdir(path.c_str(), 0755);
}

// Create a directory if it does not already exist (mkdir fails silently
// with errno = EEXIST if the directory is already present).
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────

int main() {
    UnsignedInt X(100), Y(100);
    // charge X = 37 et Y = 12 dans tous les réplicas (SetAll)
    X.SetAll(37); Y.SetAll(12);
    Bits bw;
    X.SubtractTo(&Y, &bw);       // attendu : 25 partout, bw = 0

    X.PrintBase10(cout);
    X.SetAll(5);  Y.SetAll(9);
    X.SubtractTo(&Y, &bw);       // attendu : bw = tous 1 (5 < 9)
    X.PrintBase10(cout);
    return 0;


}