//
//  int.hpp
//  gillespie
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
    void RemoveFirstSignificantBit(void);
    void Replace(UnsignedInt*, Bits*);
    void Set(unsigned int, unsigned long long int);

    void operator = (BitSet);
    
    friend class Double;
    
};

#endif
