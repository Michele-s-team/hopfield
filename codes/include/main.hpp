//
//  main.h
//  gillespie
//
//  Created by Michele on 07/02/2024.
//

#ifndef main_hpp
#define main_hpp

#include <iostream>

#include "bits.hpp"
#include "bitset.hpp"
#include "unsigned_int.hpp"

using namespace std;



#define n_bits 64
#define n_bits_mantissa 52
#define n_bits_exponent 11
#define n_bits_sign 1
//the maximum value for c[j]
#define my_M 8
#define cout_precision 8
const unsigned long long int ullong_1 = 1;
const unsigned long long int ullong_0 = 0;


//fill all the entries of BitSet_one and of Bits_one with 1, same for 0 with Bits_zero
extern BitSet BitSet_one;
extern Bits Bits_one, Bits_zero;


#endif
