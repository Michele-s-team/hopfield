//
//  bits.cpp
//  gillespie
//
//  Created by Michele on 07/02/2024.
//

#include "bits.hpp"
#include "main.hpp"

#include <vector>

Bits::Bits(){}

Bits::Bits(unsigned long long int n_in){
    
    n  = n_in;
    
}


//class setter
//inline 
void Bits::Set(const unsigned long long int& m){
    
    n = m;
    
}


//class setter
//inline 
void Bits::Set(const Bits& m){
    
    Set(m.n);
    
}


//class getter
//inline 
unsigned long long int Bits::Get(void){
    
    return n;
    
}


//set the s-th bit of *this equal to bit
void Bits::Set(unsigned int s, bool bit){
    
    unsigned int p;
    unsigned long long int result;
    
    result = 0;
    for(p=0; p<n_bits; p++){
        
        if(p!=s){
            
            result |= (((n >> p) & ullong_1) << p);
            
        }else{
            
            if(bit){
                result |= (ullong_1 << s);
            }
            
        }
        
    }
    
    n = result;
    
}


//set all n_bits entries of *this equal to bit
//inline 
void Bits::SetAll(bool bit){
    
    if(bit){(*this) = Bits_one;}
    else{(*this) = Bits_zero;}
    
}


//return the s-th bit ofinput
//inline 
bool Bits::Get(unsigned int s){
    
    return ((n >> s) & ullong_1);
    
}


//convert (bit-by_bit) *this in base 10 (so either 0 or 1) and write the result in the entries of v
//inline 
void Bits::GetBase10(vector<unsigned long long int>& v){
    
    unsigned int p;
    
    for(p=0, v.resize(n_bits); p<n_bits; p++){
        
        v[p] = Get(p);
        
    }
    
}


//set *this randomly with seed seed
inline void Bits::SetRandom(unsigned int seed){
    
    gsl_rng* ran;
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);

    SetRandom(ran);
    
    gsl_rng_free(ran);

}


//set *this randomly with random number generator ran
inline void Bits::SetRandom(gsl_rng* ran){
    
    unsigned int s;
    
    for(s=0; s<n_bits; s++){
        Set(s, (bool)(gsl_rng_uniform_int(ran, 2)));
    }
    
}


//return the one's complement of *this 
inline Bits Bits::Complement(void){
    
    return(~(*this));
    
}


//compute the one's complement of *this and write it into *this
inline void Bits::ComplementTo(void){
    
    (*this) = (~(*this));
    
}


//replace bit-by-bit *this with replacer if check = true, and leave *this unchanged if check = false
//inline 
void Bits::Replace(Bits* replacer,  Bits* check){
    
    (*this) =  (((*this) & (~(*check))) | ((*replacer) & (*check)));
    
}


//swap bit-by-bit the pair {*this, *a} if *check = true and write the result in {*this, *a}, and leave *this and *a unchanged if *check = false, where *work is a temporary variable needed to store stuff. This method requires *a and *work to be allocated
inline void Bits::Swap(Bits* a, Bits& check, Bits* work_space){

    //store temporarily *this in *work
    (*work_space) = (*this);
    
    //swap *this with *a according to *check and write the result in *this
    (*this) = ((*this) & (~check)) | ((*a) & check);
    //swap *a with *this according to *check and write the result in *a
    (*a) = ((*a) & (~check)) | ((*work_space) & check);
    
}


//set all bits of *this to 0
//inline 
void Bits::Clear(void){
    
    n = 0;
    
}


//print *this as a sequence of n_bits zeros and ones
//inline 
void Bits::Print(string title){
    
    cout << title << "{ ";
    for(unsigned int s=0; s<n_bits; s++){
        cout << this->Get(n_bits-1-s);
    }
    cout << " }\n";
    
}


//print *this as an integer to the output stream output_stream
//inline 
void Bits::Print(ostream& output_stream){
    
    output_stream << n;

}


////inline
//void Bits::operator = (const Bits& m){
//    
//    n = (m.n);
//    
//}


//return (this->n) | (m.n) (bitwise OR)
//  inline
Bits Bits::operator | (const Bits& m){
    
    return Bits((n | (m.n)));
    
}


//return (this->n) & (m.n) (bitwise AND)
//inline 
Bits Bits::operator & (const Bits& m){
    
    return Bits((n & (m.n)));
    
}


//return (this->n) ^ (m.n) (bitwise xOR)
//inline
Bits Bits::operator ^ (const Bits& m){
    
    return Bits((n ^ (m.n)));
    
}


//apply a bit-by-bit negation to *this and return the result
//inline 
Bits Bits::operator ~ (void){
    
    return Bits((~n));

}


//return (bit-by-bit) true if *this == m, false otherwise
inline Bits Bits::operator == (Bits& m){
    
    return(Bits(~(n ^ (m.n))));
    
}


//inline 
void Bits::operator &= (const Bits& m){
    
    n &= (m.n);
    
}


//inline 
void Bits::operator &= (Bits* m){
    
    n &= (m->n);
    
}


inline void Bits::operator ^= (Bits* m){
    
    n ^= (m->n);
    
}
