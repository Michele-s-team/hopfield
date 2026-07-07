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

using namespace std;

//g++ Hopfield/alpha_sweep_multiN.cpp src/*.cpp -llapack -lgsl -lgslcblas -lm -O3 -flto -Wno-deprecated -Iinclude -I/usr/include/gsl -DHAVE_INLINE -o main.o


BitSet BitSet_one; // really strange that we need to define this for the operator -= of BitSet 

// =============================================================================
// alpha_sweep_multiL.cpp
//
// Simulates the Hopfield network (bitwise Metropolis, HopfieldBits) on a
// fully-connected topology of N = L*L neurons, across a range of network
// sizes L and a range of loading ratios alpha = P/N.
//
// For each size N:
//   - the alpha grid is built with several segments of different step
//     sizes, with finer resolution near the critical capacity
//     alpha_c ≈ 0.138 (AGS result), exactly as the temperature grid in
//     phase_diagram.cpp is refined near Tc.
//   - for each alpha: P = round(alpha * N) patterns are drawn, the network
//     is initialized from one of the patterns, and N_sweeps Metropolis
//     sweeps are run (first half discarded for equilibration, second half
//     saved).
//   - N_sweeps is adapted locally to the alpha-grid step dalpha, so that
//     points close to alpha_c (where the grid is finer) get more sweeps.
//
// Output: ../results/Hopfield/phase_transition/init_from_pattern/
//         alpha_{alpha}/N{N}/beta{beta}/...
// =============================================================================

void make_dir(const string& path) {
    mkdir(path.c_str(), 0755);
}

int main() {

    // ── Fixed simulation parameters ──────────────────────────────────────────
    const double beta       = 5;          // inverse temperature
    int          N_sweeps    = 1 << 14;    // default / base number of sweeps

    // ── Lattice sizes (N = L*L, fully-connected Hopfield) ────────────────────
    vector<int> N_vals = {100, 200, 400, 600, 1000, 1500, 2000, 4000, 8000, 10000};

    // ── Alpha grid, refined around alpha_c ≈ 0.138 (AGS critical capacity) ──
    const double alpha_c   = 0.138;
    const double alpha_min = 0.01;
    const double alpha_max = 0.28;
    const int    n_alpha   = 27;           // same resolution as n_T in phase_diagram.cpp

    const double alpha_pow = 2.5;  // exposant : 2→ resserré modérément, 3→ très resserré autour de alpha_c

    vector<double> alphas;

    for (int i = 0; i < n_alpha; ++i) {
        double u = (double)i / (n_alpha - 1);   // ∈ [0, 1]
        double s = 2*u - 1;                     // ∈ [-1, 1], s=0 en alpha_c
        double a;
        if (s >= 0)
            a = alpha_c + (alpha_max - alpha_c) * pow(s, alpha_pow);
        else
            a = alpha_c + (alpha_min - alpha_c) * pow(-s, alpha_pow);

        a = round(a * 10000.0) / 10000.0;   // arrondi à 0.001

        alphas.push_back(a);
    }

    for (double a : alphas) cout << a << "  ";
    cout << " \n";

    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, 123);

    clock_t start_all = clock();

    for (int N : N_vals) {

        clock_t start_N = clock();

        cout << "\n##############################################\n";
        cout << "N = " << N << "\n";
        cout << "##############################################\n";

        for (int i = 0; i < (int)alphas.size(); ++i) {

            double alpha = alphas[i];
            int P = max(1, static_cast<int>(round(alpha * N))); //avoids P=0 error

            // ── Adaptive N_sweeps ──────────────────────────────────────────
            // Compute local step size dalpha, exactly as dT is computed from
            // the temperature grid in phase_diagram.cpp.

            /*
            double dalpha;
            if (i == 0)
                dalpha = abs(alphas[1] - alphas[0]);
            else if (i == (int)alphas.size() - 1)
                dalpha = abs(alphas[i] - alphas[i-1]);
            else
                dalpha = abs(alphas[i+1] - alphas[i-1]) / 2.0;

            const double dalpha_ref = (alpha_max - alpha_min) / (n_alpha - 1);  // pas uniforme de référence
            const int    exp_base   = 12;
            const int    exp_max    = 16;  // plafond à 2^16

            int exp_sweeps = exp_base + (int)round(log2(dalpha_ref / dalpha));
            exp_sweeps     = max(exp_base, min(exp_max, exp_sweeps));
            N_sweeps       = 1 << exp_sweeps;

            */

            // Build folder name

            ostringstream alpha_str;
            alpha_str << fixed << setprecision(4) << alpha;

            string root_dir   = "../results/Hopfield/phase_transition_multiN_init_from_pattern";
            string N_dir      = root_dir  + "/beta" + SimulationBase::format_beta(beta)+ "/N" + to_string(N);
            string alpha_dir  = N_dir     + "/alpha_" + alpha_str.str();
            string base_folder = alpha_dir;
            make_dir(root_dir);
            make_dir(N_dir);
            make_dir(alpha_dir);
            make_dir(base_folder);

            // ── Model initialization ────────────────────────────────────────
            HopfieldBits bits(N, beta, N_sweeps, P);
            bits.SetBaseFolder(base_folder.c_str());

            cout << "\n==============================================\n";
            cout << "Hopfield Model - N = " << N 
                 << "  alpha = " << alpha << " (P = " << P << ")"
                 << "  (" << i+1 << "/" << alphas.size() << ")\n";
            //cout << "  dalpha:        " << dalpha << "\n";
            cout << "  Temperature:   " << 1.0/beta << "\n";
            cout << "  Sweeps:        " << N_sweeps << "\n";
            cout << "  Base folder:   " << base_folder << "\n";
            cout << "==============================================\n\n";

            bits.initNetworkFullyConnected();
            cout << "Network initialized (Fully Connected)\n";

            // ── Evolution: HopfieldBits ────────────────────────────────────
            gsl_rng_set(ran, 42);  // fixed seed for the dynamics
            clock_t start_bits = clock();
            bits.initPatternsBits(ran);
            auto Patterns = bits.getPatternsBits();
            int mu = gsl_rng_uniform_int(ran, P);
            auto corrupted = bits.corruptPattern(Patterns[mu], 5.0/100.0, ran);
            bits.initSpinsFromConfigBits(corrupted);
            cout << "Bitwise initialization done" << endl;

            vector<vector<vector<int>>> patterns = bits.getPatternsBitsToCanonical();
            bits.SavePatterns(patterns);
            cout << "Patterns saved" << endl;

            bits.OpenSpinFiles();
            cout << "evolve_save_bits called" << endl;

            // First half: equilibration
            bits.runSweeps(ran, /*save=*/false, 0, /*shift=*/0);
            cout << " first half of the simulation done" << endl;

            // Second half: save
            bits.setNSweeps(N_sweeps);
            bits.runSweeps(ran, /*save=*/true, 1, /*shift=*/N_sweeps);
            bits.SaveSpinConfigurations(bits.getNSweeps());
            bits.CloseSpinFiles();

            cout << "evolve_save_bits terminated" << endl;
            clock_t end_bits = clock();
            double clock_bits = double(end_bits - start_bits) / CLOCKS_PER_SEC;
            cout << "\nHopfieldBits done for N = " << N << ", alpha = " << alpha
                 << ". Time: " << clock_bits << " s\n";
        }

        clock_t end_N = clock();
        double clock_N = double(end_N - start_N) / CLOCKS_PER_SEC;
        cout << "\nN = " << N << " done, " << clock_N << " s\n";
    }

    clock_t end_all = clock();
    double clock_all = double(end_all - start_all) / CLOCKS_PER_SEC;

    cout << "\nSimulation Complete. \n"
         << "Total Time: " << clock_all << " s\n";

    gsl_rng_free(ran);

    return 0;
}