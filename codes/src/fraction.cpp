#include "fraction.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"

inline Fraction::Fraction(void) : BitSet(){}


//construct a Fraction with size bits
inline Fraction::Fraction(unsigned int size) : BitSet(){

    b.resize(size);
    
}


//obtain the mantissa from x (in the IEEE 754 format) and store it bit-by-bit in *this, by setting all n_bits entries of  b[s] equal
inline void Fraction::SetAllFromDoubleMantissa(double x, vector<bool>* work_space){
        
    unsigned int p;
    
    GetMantissaFromDouble(work_space, x);
    
    for(p=0; p<GetSize()-1; p++){
        //the [p+1] is to account of the offset of one entry between the IEEE 754 convention and the convention used here to write Fraction
        b[p].SetAll((*work_space)[p+1]);
    }
    //in the IEEE 754 format the first entry of the mantissa is 1 by default -> I write it in b
    b.back().SetAll(true);
   
}

//get  bit-by-bit the value in base 10 of *this and write it in *v
inline void Fraction::GetBase10(vector<double>* v){
    
    unsigned int p, s;
    
    v->resize(n_bits);
    
    for(s=0; s<n_bits; s++){
        
        for(p=0, (*v)[s]=0.0; p<GetSize(); p++){
            (*v)[s] += gsl_pow_int(2.0, p-GetSize()) * b[p].Get(s);
        }
        
    }
    
}

inline void Fraction::PrintBase10(string title){
    
    unsigned int p;
    vector<double> v;
    
    cout << title << ": {";
    for(GetBase10(&v), p=0; p<n_bits; p++){
        
        cout << v[n_bits-1-p] << " ";
        
    }
    
    cout << "}" << endl;
    
    
}



//compute the floor of *this x *multiplicant and store the result in *result. This requires result->GetSize() = multiplicand->GetSize() and work_space->GetSize() = (this->GetSize()) + (multiplicand->GetSize())
inline void Fraction::FloorMultiply(UnsignedInt* multiplicand, UnsignedInt* result, UnsignedInt* work_space){
    
    Multiply(multiplicand, work_space);
  
    //copy the last multiplicand->GetSize() bits of work_space into *result : this will yield the floor of the multiplication above
    copy(work_space->b.begin()+GetSize(), work_space->b.end(), result->b.begin());
     
}
