//
//  hopfield_model.hpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//
//THIS CLASS SETS UP THE VARIABLES USED FOR THE SIMULATION OF THE THERMALIZATION OF A SPIN GLASS HAMILTONIAN ON THE GENERATED NETWORK
#ifndef hopfield_model_hpp
#define hopfield_model_hpp

#include "spin_system.hpp"
#include "simulation_io.hpp"
#include "main.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "gsl_rng.h"

using namespace std;

class HopfieldModel : public SpinSystem {
protected:
    double beta;                       // inverse temperature
    double inv2beta;                   // =1/(2*beta)
    int N_sweeps;                    // number of Metropolis sweeps
    int P;                           // number of patterns
    vector<vector<vector<int>>> patterns;
    vector<vector<vector<int>>> couplings; // the couplings are intialized with the patterns
    SimulationIO m_io;                  // saving files
    int neighbor_index(int, int) const;
        
public:
    
    HopfieldModel(int N, double beta, int N_sweeps, int P);   // Constructor: allocates all vectors for an L x L lattice with given inverse temperature betaJ and number of sweeps
    virtual ~HopfieldModel() = default;
    void initCouplings();
    void initPatterns(gsl_rng*);
    void initPatternsFromConfig(vector<vector<vector<int>>>);
    void setNSweeps(int);                         //sets the number of spins
    void setbeta(double);                           // Sets a new value of β*J (e.g. when sweeping over temperatures)
                        // Sets the size of the system
    int randomNumber(gsl_rng*);                   //generates a random number following an exponential distribution
    //void initrandomNumbers(gsl_rng*);           // Pre-generates exponential random numbers for the Metropolis criterion using the GSL RNG    
    //void initRandomNumbersFromExp(const vector<vector<double>>& exp_base); // Pre-generates random numbers from a pre-computed exponential base (useful to keep the same noise across different betaJ values)
    //void init(gsl_rng*);           
    vector<vector<vector<int>>> getPatternsConfig(); 
    vector<vector<vector<int>>> getCouplingsConfig(); 
    void OpenCSVFiles(const string& folder) {
        m_io.OpenCSVFiles(folder, N);
    }
    void CloseCSVFiles() { m_io.CloseCSVFiles(); }
    void SaveMagnetizations(int N) {
        vector<double> mags(n_bits);
        GetMagnetizations(mags);
        m_io.SaveMagnetizations(N, beta, mags);
    }          // saves the magnetizations of all 64 realizations in a file with the speicified number of sweeps
};


#endif