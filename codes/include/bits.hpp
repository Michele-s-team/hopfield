//
//  bits.h
//  hopfield
//
//  Created by Michele on 07/02/2024.
//

#ifndef bits_hpp
#define bits_hpp
#include <iostream>
#include <vector>
#include "gsl_rng.h"
using namespace std;

class Bits{
private:
    unsigned long long int n;
public:
    Bits();
    Bits(unsigned long long int);
    void Set(const unsigned long long int&);
    void Set(const Bits&);
    bool Get(unsigned int);
    void Set(unsigned int, bool);
    void SetAll(bool);
    unsigned long long int Get();
    void GetBase10(vector<unsigned long long int>&);
    void SetRandom(unsigned int);
    void SetRandom(gsl_rng*);
    Bits Complement(void);
    void ComplementTo(void);
    void Replace(Bits*, Bits*);
    void Swap(Bits*, Bits&, Bits*);
    void Clear(void);
    void Print(string);
    void Print(ostream&);
    friend class UnsignedInt;

    Bits operator &  (const Bits&) const;
    Bits operator |  (const Bits&) const;
    Bits operator ^  (const Bits&) const;
    Bits operator ~  (void)        const;
    Bits operator == (const Bits&) const;
    bool equal       (const Bits&) const;

    void operator &= (const Bits&);
    void operator &= (Bits*);
    void operator ^= (Bits*);
};
#endif
