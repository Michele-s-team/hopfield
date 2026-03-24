//
//  system_nobits.hpp
//  gillespie
//
//  Created by Michele on 07/02/2024.
//

#ifndef system_nobits_hpp
#define system_nobits_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "unsigned_int.hpp"


using namespace std;


//a reaction-diffusion system without bits framework
class SystemNoBits{
    
public:
    
    unsigned long long int seed, /*the total number of molecules in the system*/N,  /*N*(N-1)/2*/N_N_m_1_2, /*the left-hand side of Eq. (10b) in gillespie2007stochastic*/L, /*the random number used to draw the reaction, which represents r2 in Eq. (10b) in gillespie2007stochastic*/  R, t;
    vector<unsigned int> /*the vector containing the numbers of molecules:  x[0] -> A, x[1] -> L, x[2] -> D*/x, x_t, /*the reaction rates: a[j](x) = c[j] * f(x), where c[j] is independent of x*//*c,*/ /*the quantities a_j in Gillespie algorithm, a[0] corresponds to the resction A+D -> 2D, a[1] to the reaction A+L -> 2L, a[2] to the reaction L+D->2A*/a;
    unsigned int  /*this is a_0 in gillespie2007stochastic*/z, rhs;
    //the random number used to draw the reaction time, equal to r1 in Eq. (10a) in gillespie2007stochastic
    double r1, tau;
    string out_directory;
    stringstream outfile_name;
    ofstream outfile;
    bool compare_old, compare_new, changer;
    //a temporary variable to store the carry of operations
    gsl_rng* ran;
    
    
    SystemNoBits(void);
    SystemNoBits(const unsigned long long int, const unsigned int, const string&);
    void iterate(void);

};

#endif
