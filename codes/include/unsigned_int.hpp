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
    void SubtractShifted(UnsignedInt* val, unsigned int shift, Bits* borrow);
    void SubtractShifted(Bits* val, unsigned int shift, Bits* borrow);
    void MultiplyByConstant(unsigned long long int constant, UnsignedInt* result);

    void AddTo(UnsignedInt*, Bits*), AddTo(const Bits*, Bits*), SubtractTo(UnsignedInt*, Bits*), SubtractTo(Bits*, Bits*);
    void Multiply(UnsignedInt*, UnsignedInt*), MultiplyByTwoTo(void), MultiplyByPowerOfTwo(unsigned int), DivideByTwoTo(void);
    void operator += (UnsignedInt*), operator -= (UnsignedInt*), operator *= (UnsignedInt*), operator += (Bits*);
    Bits operator < (const UnsignedInt&), operator <= (UnsignedInt&); 
    UnsignedInt operator + (UnsignedInt*), operator - (UnsignedInt*);
    void Add(UnsignedInt*, UnsignedInt*, Bits*), Subtract(UnsignedInt*, UnsignedInt*, Bits*);    
    void AddAnd(UnsignedInt*, Bits*);
    void IncrementMaskedFast(UnsignedInt& counter, Bits mask);

    // ============================================================
    // Tree 1: reduce an array of Bits (single-bit-per-lane terms)
    // into one UnsignedInt. Used for sum_c.
    // ============================================================
    template <int MaxDeg, int MaxWidth>
    static UnsignedInt CSAReduceBits(const Bits (&terms)[MaxDeg], int deg,
                                  unsigned long long bound,
                                  Bits (&buckets)[MaxWidth][MaxDeg],  
                                  Bits (&resultBit)[MaxWidth]) {

        int count[MaxWidth] = {0};
        for (int i = 0; i < deg; ++i)
            buckets[0][count[0]++] = terms[i];
        
        for (int w = 0; w < MaxWidth; ++w)
            resultBit[w] = Bits(0);

        for (int w = 0; w < MaxWidth; ++w) {
            while (count[w] >= 3) {
                int i = 0, out = 0;
                for (; i + 3 <= count[w]; i += 3) {
                    Bits a = buckets[w][i], b = buckets[w][i+1], c = buckets[w][i+2];
                    Bits ab    = a ^ b;
                    Bits sum   = ab ^ c;
                    Bits carry = (a & b) | (ab & c);
                    buckets[w][out++] = sum;          // feed sum back into this level
                    if (w + 1 < MaxWidth)
                        buckets[w+1][count[w+1]++] = carry;
                }
                for (; i < count[w]; ++i)
                    buckets[w][out++] = buckets[w][i]; // carry leftovers down
                count[w] = out;
            }
            if (count[w] == 1) {
                resultBit[w] = buckets[w][0];
            } else if (count[w] == 2) {
                Bits x = buckets[w][0], y = buckets[w][1];
                resultBit[w] = x ^ y;
                if (w + 1 < MaxWidth)
                    buckets[w+1][count[w+1]++] = x & y;
            }
        }

        // Construction avec la vraie sémantique : bound = valeur max possible (ex: max_deg)
        UnsignedInt result(bound);
        result.SetAll(0);
        for (int w = 0; w < MaxWidth && w < (int)result.GetSize(); ++w)
            result[w] = resultBit[w];
        return result;
    }
    // ============================================================
    // Tree 2: reduce an array of UnsignedInt into one UnsignedInt.
    // Used for sum_cg.
    // ============================================================
    template <int MaxDeg>
    static UnsignedInt& CSAReduceUnsignedInt(const UnsignedInt (&operands)[MaxDeg], int deg,
                                            UnsignedInt (&buf)[MaxDeg],
                                            UnsignedInt (&scratchSum)[MaxDeg],
                                            UnsignedInt (&scratchCarry)[MaxDeg]) {
        // Réinitialise juste ce qui sera utilisé, pas de reconstruction
        for (int i = 0; i < deg; ++i) buf[i] = operands[i];   // copie légère si operator= ne réalloue pas

        int n = deg;
        while (n > 2) {
            int w = 0, i = 0;
            int slot = 0;

            for (; i + 3 <= n; i += 3) {
                UnsignedInt& sum   = scratchSum[slot];
                UnsignedInt& carry = scratchCarry[slot];
                slot++;

                CSAdd(buf[i], buf[i + 1], buf[i + 2], &sum, &carry);

                buf[w++] = sum;
                buf[w++] = carry;
            }
            for (; i < n; ++i) buf[w++] = std::move(buf[i]);
            n = w;
        }

        if (n == 1) return buf[0];

        Bits carry(0);
        buf[0].AddTo(&buf[1], &carry);
        return buf[0];
    }

private:
    static void CSAdd(const UnsignedInt& a, const UnsignedInt& b, const UnsignedInt& c,
                   UnsignedInt* sum, UnsignedInt* carry);

};

#endif
