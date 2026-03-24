//
//  int.cpp
//  gillespie
//
//  Created by Michele on 18/02/2024.
//

#include "int.hpp"

#include "gsl_math.h"


//default constructor
inline Int::Int(void): BitSet(){}

//constructor that resizes *this in order to host N and sets all the n_bits entries of s equal to the sign of N
inline Int::Int(long long int N): BitSet(fabs(N)){}

inline void Int::SetAll(long long int N){
    
    //set the magnitude of *this
    BitSet::SetAll(((unsigned long long int)(fabs(N))));
    //set the sign of *this
    s.SetAll((bool)((-1+GSL_SIGN(N))/2));
    
}
