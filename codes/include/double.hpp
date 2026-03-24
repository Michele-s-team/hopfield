//
//  double.hpp
//  gillespie
//
//  Created by Michele on 10/02/2024.
//

#ifndef double_hpp
#define double_hpp

#include <iostream>

using namespace std;

#include "bits.hpp"
#include "bitset.hpp"
#include "unsigned_int.hpp"

//this class represents a double (i.e. a collection of n_bits doubles) represented in base 2
class Double{
    
private:
    
    //the fraction: b has size 52
    BitSet b;
    //the exponent: e.b has size 11
    UnsignedInt e;
    //the sign
    Bits s;
    
public:
    
    Double(void);
    void Clear(void);
    void Normalize(void);
    void Replace(Double*, Bits*);
    void Swap(Double*, Bits&, Bits*);
    void SetAll_IEEE754(double);
    void SetAll(bool, unsigned long long int, double);
    void Set(unsigned int, bool, unsigned long long int, double);
    void SetAll(bool, unsigned long long int, BitSet&);
    void SetRandom(gsl_rng*);
    void SetRandom(unsigned int);
    void Print(string);
    void PrintBase10_IEEE754(void);
    void PrintBase10(string);
    void GetBase10(vector<double>&);

    void AddTo(Double*);
    void operator += (Double*), operator *= (Double&);
    Bits operator <(Double&);

};

#endif
