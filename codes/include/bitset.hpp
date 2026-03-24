//
//  bitset.hpp
//  gillespie
//
//  Created by Michele on 12/02/2024.
//

#ifndef bitset_hpp
#define bitset_hpp

#include <iostream>

#include "bits.hpp"

#include <vector>

using namespace std;

class UnsignedInt;



class BitSet{
    
private:
    //b[i] is the i-th bit of the BitSet
    vector<Bits> b;
    
public:
    
    BitSet();
    BitSet(unsigned long long int);
    
    void Clear(void);
    void Swap(BitSet*, Bits&, Bits*);
    void Normalize(void);
    void Normalize(unsigned int);
    void Resize(unsigned long long int);
    unsigned int GetSize(void);
    void SetRandom(gsl_rng*);
    void SetRandom(unsigned int);
    void SetAll(unsigned long long int);
    void SetAllToSize(unsigned long long int);
    void SetAll(Bits&);
    void SetAllFromDoubleMantissa(double, vector<bool>*);
    void Set(BitSet*);
    void SetFromDoubleMantissa(unsigned int, double);
    void ComplementTo(void);
    void ComplementTo(unsigned int);
    void ResizeAndSetAll(unsigned long long int);
    UnsignedInt PositionOfFirstSignificantBit(void);
    void RemoveFirstSignificantBit(void);
    unsigned long long int Get(unsigned int);
    void Print(string);
    void Print(ostream&);

    //bitwise sum and substraction
    void AddTo(BitSet*, Bits*), AddTo(Bits*, Bits*), SubstractTo(BitSet*, Bits*), SubstractTo(Bits*, Bits*), Multiply(UnsignedInt*, UnsignedInt*), MultiplyByTwoTo(void), DivideByTwoTo(void), AndTo(Bits*, unsigned int, unsigned int), And(Bits*, BitSet*);
    BitSet operator + (BitSet*), operator - (BitSet*), operator << (Bits*), Add(BitSet*, Bits*), Substract(BitSet*, Bits*);
    void operator += (BitSet*), operator -= (BitSet*), operator *= (BitSet*);
    Bits& operator [] (const unsigned int&);
    Bits operator == (BitSet&), operator < (const BitSet&), operator <= (BitSet&);
    void operator >>= (UnsignedInt*), operator <<= (UnsignedInt*), operator >>= (Bits*), operator <<= (Bits*), operator &= (Bits*);

    friend class UnsignedInt;
    friend class Double;
    friend class Fraction;
    friend class SystemBits;
    
    
};

#endif
