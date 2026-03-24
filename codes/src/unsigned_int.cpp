//
//  int.cpp
//  gillespie
//
//  Created by Michele on 07/02/2024.
//

#include "unsigned_int.hpp"

#include "fraction.hpp"
#include "lib.hpp"
#include "main.hpp"

#include <vector>

#include "gsl_math.h"

//default constructor
//inline 
UnsignedInt::UnsignedInt(void) : BitSet(){}

//constructor taht resizes *this in ordert to host N
//inline 
UnsignedInt::UnsignedInt(unsigned long long int N) : BitSet(N){}


//if this->GetSize() == replacer->GetSize(), replace bit-by-bit all bs of *this with the respective bs of *replacer, and leave *this unchanged otherwise
inline void UnsignedInt::Replace(UnsignedInt* replacer, Bits* check){
    
    if((this->GetSize()) == (replacer->GetSize())){
        
        for(unsigned int s=0; s<(this->GetSize()); s++){
            b[s].Replace((replacer->b.data()) + s, check);
        }
        
    }
    
}


//set the s-th bit of *this equal to i. This requires b to be sized properly
//inline 
void UnsignedInt::Set(unsigned int s, unsigned long long int i){
    
    unsigned int p;
    Bits n(i);
    
    for(p=0; p<b.size(); p++){
        b[p].Set(s, ((bool)(n.Get(p))));
    }
    
}


//inline 
//print *this in base 10 in comma-separated-value format
void UnsignedInt::PrintBase10(ostream& output_stream){
    
    unsigned int  p;
    vector<unsigned long long int> v;
    
    for(p=0, GetBase10(v); p<n_bits; p++){
        output_stream << v[n_bits-1-p] << ",";
    }

    
}


inline void UnsignedInt::GetBase10(vector<unsigned long long int>& v){
    
    unsigned int p;
    for(p=0, v.resize(n_bits); p<n_bits; p++){
        
        v[p] = Get(p);
        
    }
    
}


inline void UnsignedInt::operator = (BitSet m){
    
    b = m.b;
    
}
