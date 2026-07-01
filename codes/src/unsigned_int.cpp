//
//  int.cpp
//  hopfield
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

//constructor that resizes *this in order to host N
//inline 
UnsignedInt::UnsignedInt(unsigned long long int N) : BitSet(N){}


//if this->GetSize() == replacer->GetSize(), replace bit-by-bit all bs of *this with the respective bs of *replacer, and leave *this unchanged otherwise
void UnsignedInt::Replace(UnsignedInt* replacer, Bits* check){
    
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
    output_stream << "\n";
    output_stream << endl;
    
}


void UnsignedInt::GetBase10(vector<unsigned long long int>& v){
    
    unsigned int p;
    for(p=0, v.resize(n_bits); p<n_bits; p++){
        
        v[p] = Get(p);
        
    }
    
}


void UnsignedInt::operator = (BitSet m){
    
    b = m.b;
    
}

// Initializes the UnsignedInt from a vector of 64-bit unsigned integers.
// Each element of the vector corresponds to one column (system),
// and is decomposed bit by bit into the rows of b:
// b[p][s] = p-th bit of vec[s], for p = 0..n_bits-1, s = 0..n_cols-1
// Remaining rows above n_bits are zeroed out.
void UnsignedInt::SetFromVector(const vector<unsigned long long>* vec) {
    if (!vec) return; // safety check

    size_t n_cols = vec->size();

    if (GetSize() < n_bits) Resize(n_bits);

    for (size_t p = 0; p < n_bits; p++) {
        for (size_t s = 0; s < n_cols; s++) {
            b[p].Set(s, ((*vec)[s] >> p) & 1ULL);
        }
    }

    for (size_t p = n_bits; p < GetSize(); p++) {
        b[p].SetAll(false);
    }
}


void UnsignedInt::AddScalar(unsigned long long int val) {
    UnsignedInt tmp(val);
    tmp.SetAll(val);
    (*this) += &tmp;
}

// Increment by 1 the parallel systems (lanes) selected by *mask, leaving all other
// lanes unchanged. Ripple-carry adder across the n_bits rows of b, from LSB (p=0)
// to MSB. carry is seeded with *mask: for lanes where mask==0, carry reste nul à
// chaque étape et le bit correspondant n'est jamais modifié.
void UnsignedInt::Increment(Bits* mask){
    
    Bits carry(*mask);
    
    for(unsigned int p=0; p<n_bits; p++){
        
        Bits new_carry = b[p] & carry;   // carry-out du bit p (b[p]==1 et carry==1)
        b[p] = b[p] ^ carry;             // bit somme (inchangé si carry==0)
        carry = new_carry;
        
    }
    
}


// Decrement by 1 the parallel systems (lanes) selected by *mask, leaving all other
// lanes unchanged. Ripple-borrow subtractor across the n_bits rows of b, from LSB
// to MSB. borrow est initialisé à *mask.
void UnsignedInt::Decrement(Bits* mask){
    
    Bits borrow(*mask);
    
    for(unsigned int p=0; p<n_bits; p++){
        
        Bits new_borrow = (~b[p]) & borrow;  // borrow-out du bit p (b[p]==0 et borrow==1)
        b[p] = b[p] ^ borrow;                // bit différence (inchangé si borrow==0)
        borrow = new_borrow;
        
    }
    
}

// Adds *val to *this, only on the lanes selected by *mask
void UnsignedInt::AddMasked(UnsignedInt* val, Bits* mask){
    Bits carry; carry.SetAll(false);
    for(unsigned int p=0; p<n_bits; p++){
        Bits add_bit = val->b[p] & (*mask);
        Bits sum      = b[p] ^ add_bit ^ carry;
        Bits new_carry = (b[p] & add_bit) | (b[p] & carry) | (add_bit & carry);
        b[p]  = sum;
        carry = new_carry;
    }
}

// Subtracts *val from *this, only on the lanes selected by *mask
void UnsignedInt::SubtractMasked(UnsignedInt* val, Bits* mask){
    Bits borrow; borrow.SetAll(false);
    for(unsigned int p=0; p<n_bits; p++){
        Bits sub_bit = val->b[p] & (*mask);
        Bits diff       = b[p] ^ sub_bit ^ borrow;
        Bits new_borrow = ((~b[p]) & sub_bit) | ((~b[p]) & borrow) | (sub_bit & borrow);
        b[p]   = diff;
        borrow = new_borrow;
    }
}