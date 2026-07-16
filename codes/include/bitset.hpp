//
//  bitset.hpp
//  hopfield
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



class BitSet {

private:

    // b[i] is the i-th bit of the BitSet.
    vector<Bits> b;

public:
    unsigned int bits(unsigned long long int);
    BitSet();
    BitSet(unsigned long long int);

    void Clear(void);
    void Swap(BitSet*, Bits&, Bits*);
    void Normalize(void), Normalize(unsigned int);
    void Resize(unsigned long long int);
    unsigned int GetSize(void) const;

    void SetRandom(gsl_rng*), SetRandom(unsigned int);
    void SetAll(unsigned long long int), SetAllToSize(unsigned long long int), SetAll(Bits&);
    void SetAllFromDoubleMantissa(double, vector<bool>*);
    void Set(BitSet*);
    void SetFromDoubleMantissa(unsigned int, double, vector<bool>&);
    void ComplementTo(void);

    void ResizeAndSetAll(unsigned long long int);
    UnsignedInt PositionOfFirstSignificantBit(void);
    void RemoveFirstSignificantBit(void);
    unsigned long long int Get(unsigned int);

    void Print(string), Print(ostream&);

    // Bitwise operators.
    
    BitSet operator<<(Bits* m);
    Bits& operator[](const unsigned int&);
    void AndTo(Bits*, unsigned int, unsigned int), And(Bits*, BitSet*);
    Bits operator==(BitSet&);

    void operator^=(Bits*),operator>>=(UnsignedInt*), operator<<=(UnsignedInt*);
    BitSet& operator=(const BitSet&);
    void CopyValues(const BitSet&);
    void operator>>=(Bits*), operator<<=(Bits*), operator&=(Bits*);

    friend class UnsignedInt;
};

#endif
