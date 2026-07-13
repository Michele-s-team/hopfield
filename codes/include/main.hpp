//
//  main.hpp
//  hopfield
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

static constexpr int BITS_PER_BLOCK = 64;
static constexpr int BLOCK_MASK = BITS_PER_BLOCK - 1; // 63

inline Bits Bits_one(~0ULL);
inline Bits Bits_zero(0ULL);
inline UnsignedInt UnsignedInt_one;
extern BitSet BitSet_one;


#endif
