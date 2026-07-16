//
//  unsigned_int.hpp
//  hopfield
//
//  Created by Michele on 07/02/2024.
//


#ifndef unsigned_int_hpp
#define unsigned_int_hpp

#include <iostream>

#include "bitset.hpp"

using namespace std;



//an int represented in base 2 in b
class UnsignedInt : public BitSet{
    
private:
    
    //here b[i] is the i-th bit of the representation of *this in base 2
    
public:
    
    UnsignedInt(void);
    UnsignedInt(unsigned long long int);
    
    void PrintBase10(ostream&);
    void GetBase10(vector<unsigned long long int>&);
    void Replace(UnsignedInt*, Bits*);
    void Set(unsigned int, unsigned long long int);
    void SetFromVector(const vector<unsigned long long>*);


    void AddScalar(unsigned long long int val, UnsignedInt& tmp);
    void ComplementTo(unsigned int);

    UnsignedInt PositionOfFirstSignificantBit(void);
    void Increment(Bits* mask);
    void Decrement(Bits* mask);

    void AddMasked(UnsignedInt* val, Bits* mask);
    void SubtractMasked(UnsignedInt* val, Bits* mask);
    void MultiplyByConstant(unsigned long long int constant, UnsignedInt* result);

    void AddTo(UnsignedInt*, Bits*), AddTo(const Bits*, Bits*), SubstractTo(UnsignedInt*, Bits*), SubstractTo(Bits*, Bits*);
    void Multiply(UnsignedInt*, UnsignedInt*), MultiplyByTwoTo(void), DivideByTwoTo(void);
    void operator += (UnsignedInt*), operator -= (UnsignedInt*), operator *= (UnsignedInt*), operator += (Bits*);
    Bits operator < (const UnsignedInt&), operator <= (UnsignedInt&); 
    UnsignedInt operator + (UnsignedInt*), operator - (UnsignedInt*);
    void Add(UnsignedInt*, UnsignedInt*, Bits*), Substract(UnsignedInt*, UnsignedInt*, Bits*);    
    void AddAnd(UnsignedInt*, Bits*);
    void IncrementMaskedFast(UnsignedInt& counter, Bits mask);
};

#endif
