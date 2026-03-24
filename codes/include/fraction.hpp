
#ifndef fraction_hpp
#define fraction_hpp

#include <iostream>

#include <vector>
#include "bitset.hpp"
#include "unsigned_int.hpp"

using namespace std;


//a real number 0 <= 0 < 1  represented in base 2 in b: x = sum_{i=0}^{b.size()-1} 2^{i-b.size()} b[i]
class Fraction : public BitSet{
    
private:
    
    
public:
    
    Fraction(void);
    Fraction(unsigned int);
    void SetAllFromDoubleMantissa(double, vector<bool>*);
    void FloorMultiply(UnsignedInt*, UnsignedInt*, UnsignedInt*);
    void GetBase10(vector<double>*);
    void PrintBase10(string);
    
//    friend class ;
    
};

#endif
