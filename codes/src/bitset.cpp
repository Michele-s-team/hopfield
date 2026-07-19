//
//  bitset.cpp
//  hopfield
//
//  Created by Michele on 12/02/2024.
//

#include "bitset.hpp"
#include "gsl_math.h"
#include "main.hpp"
#include <vector>
#include <stdint.h>


unsigned long long int two_pow(unsigned long long int i){
    return 1ULL << i;
}

//return the number of bits necessary to write n in base 2
unsigned int BitSet::bits(unsigned long long int n){
    if (n == 0) return 1;
    return 64 - __builtin_clzll(n);  // count leading zeros, disponible sur GCC/Clang
}


//write the mantissa of x into *result by resizing *result properly. This method requires that result->size() = n_bits_mantissa
void GetMantissaFromDouble(vector<bool>* result, double x){
    
    uint64_t *pointer = (uint64_t*)&x;
    uint64_t byte;
    unsigned int p;
    
    
    byte = pointer[0];
    
    for(p=0; (p < 64) && (p < result->size());  p++){
        //run through the part of x which is the mantissa
        
        (*result)[p] = ((bool)(byte & 1));
        
        byte >>= 1;
        
    }
    
}


// ============================================================================
// Constructors & Lifecycle Management
// ============================================================================

// Default constructor: Initializes an empty BitSet.
BitSet::BitSet() {}

// Sized constructor: Initializes the BitSet with enough capacity to hold N bits.
BitSet::BitSet(unsigned long long int N) {
    b.resize(bits(N));
}

// ============================================================================
// Core Management & Structural Modifiers
// ============================================================================

// Resets all elements in the BitSet container to zero.
void BitSet::Clear() {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        b[s].Set(0);
    }
}


// Swaps bits with another BitSet condition-by-condition using a workspace buffer.
void BitSet::Swap(BitSet* a, Bits& check, Bits* work_space) {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        b[s].Swap(&((a->b)[s]), check, work_space);
    }
}

// Removes trailing elements that only contain zeros, keeping at least one element.
void BitSet::Normalize() {
    int p;
    // Changed p>=0 to p>0 to avoid deleting the only remaining Bits element
    for (p = GetSize() - 1; p > 0; p--) {
        if (b[p].Get() == 0) {
            b.pop_back();
        } else {
            break;
        }
    }
}

// Removes trailing zero elements, ensuring the final size does not drop below 'n'.
void BitSet::Normalize(unsigned int n) {
    int p;
    for (p = GetSize() - 1; p >= 0; p--) {
        if ((b[p].Get() == 0) && (GetSize() >= n)) {
            b.pop_back();
        } else {
            break;
        }
    }
}

// Resizes the internal vector container to the specified size.
void BitSet::Resize(unsigned long long int size) {
    b.resize(size);
}

// Returns the current number of elements inside the BitSet container.
unsigned int BitSet::GetSize() const {
    return static_cast<unsigned int>(b.size());
}

// ============================================================================
// Randomization, Initializations & Value Mapping
// ============================================================================

// Fills all internal bits randomly using an active GSL random number generator instance.
void BitSet::SetRandom(gsl_rng* ran) {
    unsigned int s, p;
    const unsigned int size = static_cast<unsigned int>(b.size());
    for (s = 0; s < size; s++) {
        for (p = 0; p < n_bits; p++) {
            b[s].Set(p, static_cast<bool>(gsl_rng_uniform_int(ran, 2)));
        }
    }
}
// Allocates a temporary GSL random engine using a seed to randomize the BitSet.
void BitSet::SetRandom(unsigned int seed) {
    gsl_rng* ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);

    SetRandom(ran);
    
    gsl_rng_free(ran);
}

// Maps an unsigned integer's individual bits across the elements of this BitSet.
void BitSet::SetAll(unsigned long long int i) {
    Bits m(i);
    unsigned int n= bits(m.Get());

    if (GetSize() < n) {
        std::cerr << "BitSet too small\n";
        abort();
    }

    const unsigned int size = GetSize();  // cache separately from 'n'
    for (unsigned int s = 0; s < n; s++) {
        b[s].SetAll(m.Get(s));
    }
    for (unsigned int s = n; s < size; s++) {
        b[s].SetAll(false);
    }
}

// Distributes the bits of 'i' sequentially across the vector up to current size.
void BitSet::SetAllToSize(unsigned long long int i) {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        b[s].SetAll((i >> s) & ullong_1);
    }
}

// Broadcasts and copies a single Bits element into every slot of this BitSet.
void BitSet::SetAll(Bits& m) {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        b[s] = m;
    }
}

// Extracts a double's mantissa bytes and saves them across all elements.
void BitSet::SetAllFromDoubleMantissa(double x, vector<bool>* work_space) {
    GetMantissaFromDouble(work_space, x);
    const unsigned int size = GetSize();
    for (unsigned int p = 0; p < size; p++) {
        b[p].SetAll((*work_space)[p]);
    }
}

// Sets a single specific bit column index using an extracted double mantissa.
void BitSet::SetFromDoubleMantissa(unsigned int s, double x, vector<bool>& v) {
    GetMantissaFromDouble(&v, x);
    const unsigned int size = GetSize();
    for (unsigned int p = 0; p < size; p++) {
        b[p].Set(s, v[p]);
    }
    v.clear();
}

// Copies another BitSet's entries into this one, padding the remainder with zeros.
void BitSet::Set(BitSet* m) {
    unsigned int s;
    const unsigned int m_size = m->GetSize();
    const unsigned int this_size = GetSize();
    for (s = 0; s < m_size; s++) {
        b[s] = (m->b)[s];
    }
    for (; s < this_size; s++) {
        b[s].SetAll(false);
    }
}

// Inverts every bit inside the entire BitSet (performs a bitwise NOT/one-complement).
void BitSet::ComplementTo() {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        b[s].ComplementTo();
    }
}

// Automatically scales the BitSet capacity to fit 'i' before writing its value.
void BitSet::ResizeAndSetAll(unsigned long long int i) {
    Resize(bits(i));
    SetAll(i);
}

// Locates the first active significant bit starting from the end and flips it to 0.
void BitSet::RemoveFirstSignificantBit() {
    int s;
    Bits check_old, check_new;
    
    for (s = GetSize() - 1, check_old.SetAll(0); s >= 0; s--) {
        check_new.Set(check_old | (*this)[s]);
        b[s].Set((*this)[s] & check_old & check_new);
        check_old = check_new;
    }
}

// Reconstructs and returns an unsigned long long representation from a bit profile row.
unsigned long long int BitSet::Get(unsigned int p) {
    unsigned int s;
    unsigned long long int result;
    const unsigned int size = GetSize();
    
    for (result = 0, s = 0; s < size; s++)
        result |= (static_cast<unsigned long long int>(b[s].Get(p)) << s);
    
    return result;
}

// ============================================================================
// Display & Stream Output Formatting
// ============================================================================

// Prints the entire BitSet to standard console output decorated with a custom title.
void BitSet::Print(string title) {
    cout << title << endl;
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        cout << "[" << s << "] = ";
        if (s < 10) { cout << " "; }
        b[s].Print("");
    }
    cout << endl;
}

// Streams raw BitSet content tab-separated directly into an active output stream.
void BitSet::Print(ostream& output_stream) {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        b[s].Print(output_stream);
        output_stream << "\t";
    }
}

// ============================================================================
// Bitwise Operator Overloads & Surcharges
// ============================================================================

// Evaluates a copy shifted left by the pattern provided in 'm' without modifying source.
BitSet BitSet::operator<<(Bits* m) {
    BitSet t = (*this);
    t <<= m;
    return t;
}

// Direct access bracket operator returning a reference to a specific element index.
Bits& BitSet::operator[](const unsigned int& i) {
    return b[i];
}

const Bits& BitSet::operator[](const unsigned int& i) const {
    return b[i];
}

// Applies a bitwise AND with 'm' across a constrained segment from start to end-1.
void BitSet::AndTo(Bits* m, unsigned int start, unsigned int end) {
    for (unsigned int s = start; s < end; s++) {
        b[s] &= m;
    }
}

// Runs a bitwise AND across all elements, putting output values into a target BitSet.
void BitSet::And(Bits* m, BitSet* result) {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        (result->b)[s] = (b[s] & (*m));
    }
}

// Compares equality against another BitSet item-by-item, matching sizes first.
Bits BitSet::operator==(BitSet& m) {
    unsigned int p;
    Bits result;
    
    if (GetSize() == m.GetSize()) {
        const unsigned int size = GetSize();
        for (p = 0, result.SetAll(true); p < size; p++) {
            result &= (b[p] == ((m.b)[p]));
        }
    } else {
        result.SetAll(false);
    }
    
    return result;
}

// Applies an in-place bitwise XOR across all storage entries using mask 'm'.
void BitSet::operator^=(Bits* m) {
    const unsigned int size = GetSize();
    for (unsigned int s = 0; s < size; s++) {
        b[s] ^= m;
    }
}

// Standard copy assignment operator replicating container data arrays.
BitSet& BitSet::operator=(const BitSet& m) {
    b = m.b;
    return *this;
}

void BitSet::CopyValues(const BitSet& m){
    unsigned int n = min(GetSize(), m.GetSize());

    for(unsigned int i = 0; i < n; i++)
        b[i] = m.b[i];

    for(unsigned int i = n; i < GetSize(); i++)
        b[i].SetAll(false);
}

void BitSet::CopyValues(const Bits& m){
    b[0] = m;
}
// Shifts entries down to the right dynamically by a factor scaling with binary powers.
void BitSet::operator>>=(UnsignedInt* e) {
    const unsigned int e_size = e->GetSize();       // cache #1: e is not modified during the loop
    const int size = static_cast<int>(GetSize());   // cache #2: this is not resized during the loop
    unsigned int n;
    int m;
    Bits zero;
    zero.SetAll(false);
    
    for (n = 0; n < e_size; n++) {
        int shift_dist = 1 << n;   // 2^n via integer shift instead of float pow
        
        // First chunk: shift valid elements to the right
        for (m = 0; m < size - shift_dist; m++) {
            b[m].Replace(&(b[m + shift_dist]), &((e->b)[n]));
        }
        
        // Second chunk: fill remaining overflow spaces with zeros
        for (m = max(size - shift_dist, 0); m < size; m++) {
            b[m].Replace(&zero, &((e->b)[n]));
        }
    }
}

// Shifts entries up to the left dynamically by a factor scaling with binary powers.
void BitSet::operator<<=(UnsignedInt* e) {
    const unsigned int e_size = e->GetSize();       // cache #1: e is not modified during the loop
    const int size = static_cast<int>(GetSize());   // cache #2: this is not resized during the loop
    unsigned int n;
    int m;
    Bits zero;
    zero.SetAll(false);
    
    for (n = 0; n < e_size; n++) {
        int shift_dist = 1 << n;   // 2^n via integer shift instead of float pow, see note below
        
        // First chunk: shift valid elements to the left
        for (m = size - 1; m >= shift_dist; m--) {
            b[m].Replace(&(b[m - shift_dist]), &((e->b)[n]));
        }
        
        // Second chunk: fill remaining overflow spaces with zeros
        for (m = min(size - 1, shift_dist - 1); m >= 0; m--) {
            b[m].Replace(&zero, &((e->b)[n]));
        }
    }
}

// Shifts elements down towards the right by one position, dropping the first element.
void BitSet::operator>>=(Bits* l) {
    const int size = static_cast<int>(GetSize());
    int m;
    for (m = 0; m < size - 1; m++) {
        b[m].Replace(b.data() + (m + 1), l);
    }
    b.back().Replace(&Bits_zero, l);
}

// Shifts elements up towards the left by one position, clearing out the bottom slot.
void BitSet::operator<<=(Bits* l) {
    const int size = static_cast<int>(GetSize());
    int m;
    for (m = size - 1; m > 0; m--) {
        b[m].Replace(b.data() + (m - 1), l);
    }
    b.front().Replace(&Bits_zero, l);
}

// Performs a complete destructive bitwise AND across all entries against condition 'm'.
void BitSet::operator&=(Bits* m) {
    AndTo(m, 0, GetSize());
}