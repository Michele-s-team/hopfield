//
//  unsigned_int.cpp
//  hopfield
//
//  Created by Michele on 07/02/2024.
//

#include "unsigned_int.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include <vector>

// ============================================================================
// Constructors & Lifecycle Management
// ============================================================================

// Default constructor initializing an empty UnsignedInt instance.
//default constructor
//inline 
UnsignedInt::UnsignedInt(void) : BitSet(){}

// Sized constructor allocating layout spaces to host up to N bits.
//constructor that resizes *this in order to host N
//inline 
UnsignedInt::UnsignedInt(unsigned long long int N) : BitSet(N){}

// ============================================================================
// Input / Output & Representations
// ============================================================================

// Streams row track profiles formatted into standard base 10 values.
//inline 
//print *this in base 10 in comma-separated-value format
void UnsignedInt::PrintBase10(ostream& output_stream){
    
    unsigned int  p;
    vector<unsigned long long int> v;
    
    for(p=0, GetBase10(v); p<n_bits; p++){
        output_stream << v[n_bits-1-p] << ",";
    }
    output_stream << "\n";
    output_stream << endl;
    
}

// Translates active bit alignments into primitive 10-base integer profiles.
void UnsignedInt::GetBase10(vector<unsigned long long int>& v){
    
    unsigned int p;
    for(p=0, v.resize(n_bits); p<n_bits; p++){
        
        v[p] = Get(p);
        
    }
    
}

// ============================================================================
// Setters, Data Conversions & Overwrites
// ============================================================================

// Replaces entries bit-by-bit from matching structures depending on condition flags.
//if this->GetSize() == replacer->GetSize(), replace bit-by-bit all bs of *this with the respective bs of *replacer, and leave *this unchanged otherwise
void UnsignedInt::Replace(UnsignedInt* replacer, Bits* check){
    const unsigned int size = this->GetSize();
    
    if(size == (replacer->GetSize())){
        
        for(unsigned int s=0; s<size; s++){
            b[s].Replace((replacer->b.data()) + s, check);
        }
        
    }
    
}

// Overwrites specific bit positions inside the target structural alignment columns.
//set the s-th bit of *this equal to i. This requires b to be sized properly
//inline 
void UnsignedInt::Set(unsigned int s, unsigned long long int i){
    
    unsigned int p;
    const unsigned int size = b.size();
    Bits n(i);
    
    for(p=0; p<size; p++){
        b[p].Set(s, ((bool)(n.Get(p))));
    }
    
}

// Decomposes primitive arrays into corresponding system column profiles bit-by-bit.
// Initializes the UnsignedInt from a vector of 64-bit unsigned integers.
// Each element of the vector corresponds to one column (system),
// and is decomposed bit by bit into the rows of b:
// b[p][s] = p-th bit of vec[s], for p = 0..n_bits-1, s = 0..n_cols-1
// Remaining rows above n_bits are zeroed out.
void UnsignedInt::SetFromVector(const vector<unsigned long long>* vec) {
    if (!vec) return; // safety check

    const unsigned int size = GetSize();

    size_t n_cols = vec->size();

    if (size < n_bits) Resize(n_bits);

    for (size_t p = 0; p < n_bits; p++) {
        for (size_t s = 0; s < n_cols; s++) {
            b[p].Set(s, ((*vec)[s] >> p) & 1ULL);
        }
    }

    for (size_t p = n_bits; p < size ; p++) {
        b[p].SetAll(false);
    }
}

// ============================================================================
// Scalar Operations & Bitwise Modifications
// ============================================================================

// Broadcasts an absolute scalar constant sum operation uniformly over the object tracks.
void UnsignedInt::AddScalar(unsigned long long int val) {
    UnsignedInt tmp(val);
    tmp.SetAll(val);
    (*this) += &tmp;
}

// Computes the one-complement logic inversion mapping out padding allocations.
//write the one-complement of *this with respect to a size 'size' of the binary representation and write it into *this
void UnsignedInt::ComplementTo(const unsigned int size){
    
    unsigned int s;
    const unsigned int n = GetSize();
    
    
    //set the first bits common to *this
    for(s=0; s<n; s++){
        b[s] = (b[s]).Complement();
    }
    
    Resize(size);

    //set the remaining bits equal to one 
    for(; s<n; s++){
        b[s] = Bits_one;
    }
    
}

// ============================================================================
// Masked Operations (Conditional Bit-by-Bit Arithmetic)
// ============================================================================

// Adds a unit step value over parallel system lanes chosen by mask flags.
// Increment by 1 the parallel systems (lanes) selected by *mask, leaving all other
// lanes unchanged. Ripple-carry adder across the n_bits rows of b, from LSB (p=0)
// to MSB. carry is seeded with *mask: for lanes where mask==0, carry reste nul à
// chaque étape et le bit correspondant n'est jamais modifié.
void UnsignedInt::Increment(Bits* mask){
    
    Bits carry(*mask);
    
    for(unsigned int p=0; p<n_bits; p++){
        
        Bits new_carry = b[p] & carry;   // carry-out du bit p (b[p]==1 et carry==1)
        b[p] = b[p] ^ carry;             // bit somme (inchangé si carry==0)
        carry = new_carry;
        
    }
    
}

// Subtracts a unit step value over parallel system lanes chosen by mask flags.
// Decrement by 1 the parallel systems (lanes) selected by *mask, leaving all other
// lanes unchanged. Ripple-borrow subtractor across the n_bits rows of b, from LSB
// to MSB. borrow est initialisé à *mask.
void UnsignedInt::Decrement(Bits* mask){
    
    Bits borrow(*mask);
    
    for(unsigned int p=0; p<n_bits; p++){
        
        Bits new_borrow = (~b[p]) & borrow;  // borrow-out du bit p (b[p]==0 et borrow==1)
        b[p] = b[p] ^ borrow;                // bit différence (inchangé si borrow==0)
        borrow = new_borrow;
        
    }
    
}

// Combines parallel inputs together restricting updates to checked active lanes.
// Adds *val to *this, only on the lanes selected by *mask
void UnsignedInt::AddMasked(UnsignedInt* val, Bits* mask){
    Bits carry; carry.SetAll(false);
    for(unsigned int p=0; p<n_bits; p++){
        Bits add_bit = val->b[p] & (*mask);
        Bits sum      = b[p] ^ add_bit ^ carry;
        Bits new_carry = (b[p] & add_bit) | (b[p] & carry) | (add_bit & carry);
        b[p]  = sum;
        carry = new_carry;
    }
}

// Evaluates differential additions strictly constrained inside lane groupings.
// Subtracts *val from *this, only on the lanes selected by *mask
void UnsignedInt::SubtractMasked(UnsignedInt* val, Bits* mask){
    Bits borrow; borrow.SetAll(false);
    for(unsigned int p=0; p<n_bits; p++){
        Bits sub_bit = val->b[p] & (*mask);
        Bits diff       = b[p] ^ sub_bit ^ borrow;
        Bits new_borrow = ((~b[p]) & sub_bit) | ((~b[p]) & borrow) | (sub_bit & borrow);
        b[p]   = diff;
        borrow = new_borrow;
    }
}

// Multiplies content scales skipping inactive bits dynamically to accelerate math.
/*
 * Multiplies *this by a scalar constant (same value across all
 * n_bits lanes) and writes the result to *result.
 *
 * Numerical equivalent of Multiply(&multiplicand, result) where multiplicand
 * would be an UnsignedInt broadcasting “constant” across all lanes
 * (as via MultiplyByInteger), BUT:
 *   - no temporary UnsignedInt is created for the constant
 *   - the zero bits of ‘constant’ are skipped entirely: no AND operation,
 *     no unnecessary carry propagation on these rows (mathematically,
 *     when multiplicand[s] is all zeros, the corresponding iteration
 *     changes anything: u=0, carry remains at 0, so skipping it is strictly
 *     equivalent)
 *   - non-zero bits are treated as a simple addition shifted by
 *     *this (no need for an AND operation since the mask is ‘all ones’)
 *
 * CONTRACT (identical to Multiply):
 * - `this` has a size `n = GetSize()`
 * - `result` MUST already be allocated with a size >= `n + bits(constant)`
 * - `result` is NOT resized here; no bounds checking
 */
void UnsignedInt::MultiplyByConstant(unsigned long long int constant, UnsignedInt* result){
    const unsigned int n_size = GetSize();
    const unsigned int result_size = result->GetSize();

    result->SetAll(Bits_zero); // une seule fois

    for (unsigned int s = 0; constant != 0; ++s, constant >>= 1)
    {
        if (!(constant & 1ULL)) continue; // bit nul : rien à faire, on saute tout le rang

        Bits carry; carry.Clear();

        for (unsigned int p = 0; p < n_size; ++p)
        {
            unsigned int idx = p + s;
            if (idx >= result_size) continue; // ou assert(false)

            Bits t;
            t.Set(((result->b)[idx]) ^ b[p] ^ carry);

            carry.Set((b[p] & (((result->b)[idx]) | carry)) |
                      (((result->b)[idx]) & carry));

            ((result->b)[idx]).Set(t);
        }

        unsigned int idx = s + n_size;
        if (idx < result_size)
            ((result->b)[idx]).Set(carry);
    }
}

// ============================================================================
// Standard Arithmetic (Buffer Modifications / Low-Level Subroutines)
// ============================================================================

// Adds entries element-by-element cascading carry outputs across structural boundaries.
//same as UnsignedInt::operator +=  but the last bit is not pushed back into b, but written into *carry. This method requires this->GetSize() to be >= addend->GetSize()
//inline 
void UnsignedInt::AddTo(UnsignedInt* addend, Bits* carry){
    
    Bits t;
    unsigned int p;
    unsigned int size_addend= addend->GetSize();
    const unsigned int size= GetSize();

    
    for(p=0, carry->Clear(); p<size_addend; p++){
        //run over  bits of addend
        
        t.Set(((b[p]).Get()) ^ (((addend->b)[p]).Get()) ^ (carry->Get()));
        carry->Set(((((addend->b)[p]).Get()) & (((b[p]).Get()) | (carry->Get()))) | (((b[p]).Get()) & (carry->Get())));
        (b[p]).Set(t);
        
    }
    for(p=size_addend; p<size && carry->Get()!=0; p++){
        t.Set((b[p]) ^ (*carry));
        carry->Set((b[p]) & (*carry));
        (b[p]).Set(t);   
    }
    
}

// Combines a standalone bit entry incrementing data with low-level carry ripples.
//add bit-by-bit addend (which here is either 1 or 0) to *this and store the result in *this and the carry in *carry. This method requires this->GetSize() to be > 1
//inline 
void UnsignedInt::AddTo(Bits* addend, Bits* carry){
    Bits t;
    unsigned int p;
    const unsigned int size = GetSize();

    carry->Set((*addend) & (b[0]));
    (b[0]).Set((b[0]) ^ (*addend));

    for(p=1; p<size && carry->Get()!=0; p++){
        t.Set((b[p]) ^ (*carry));
        carry->Set((b[p]) & (*carry));
        (b[p]).Set(t);
    }
}

// Subtracts container components evaluating borrow patterns progressively across rows.
void UnsignedInt::SubstractTo(UnsignedInt* subtrahend, Bits* borrow){

    unsigned int p;
    unsigned int size_subtrahend= subtrahend->GetSize();
    const unsigned int size= GetSize();
    Bits a, s, t;

    borrow->Clear();

    for(p = 0; p < size_subtrahend; ++p){
        a = b[p];
        s = (*subtrahend)[p];

        // result bit
        t.Set(a ^ s ^ (*borrow));

        // borrow_out = (~a & (s | borrow_in)) | (s & borrow_in)
        borrow->Set((~a & (s | (*borrow))) | (s & (*borrow)));

        b[p].Set(t);
    }

    for(; p < size; ++p){
        a = b[p];

        t.Set(a ^ (*borrow));

        // subtraction of borrow from a
        borrow->Set((~a) & (*borrow));

        b[p].Set(t);
    }
}

// Deducts a isolated track profile tracking borrow transitions downwards.
void UnsignedInt::SubstractTo(Bits* subtrahend, Bits* borrow){

    unsigned int p;
    const unsigned int size= GetSize();
    Bits a, t;

    a = b[0];

    t.Set(a ^ (*subtrahend));

    borrow->Set((~a) & (*subtrahend));

    b[0].Set(t);

    for(p = 1; p <size; ++p)
    {
        a = b[p];

        t.Set(a ^ (*borrow));

        borrow->Set((~a) & (*borrow));

        b[p].Set(t);
    }
}

// Executes matrix multiplication mapping product layers to a target buffer layout.
/*
 * CONTRACT:
 * - multiplicand has size m = multiplicand->GetSize()
 * - this has size n = GetSize()
 * - result MUST already be preallocated with size >= m + n
 *
 * This function does NOT resize result and assumes sufficient capacity.
 * No bounds checking is performed for performance reasons.
 *
 * WARNING:
 * Accesses result->b[p + s] assume valid indexing; caller is responsible
 * for guaranteeing correct allocation size.
 */
void UnsignedInt::Multiply(UnsignedInt* multiplicand, UnsignedInt* result){
    const unsigned int m_size = multiplicand->GetSize();
    const unsigned int result_size = result->GetSize();
    const unsigned int n_size = GetSize();

    Bits carry, t, u;

    // IMPORTANT:
    // result must already be sized to at least (m_size + n_size)
    result->SetAll(Bits_zero);  // done once only

    for (unsigned int s = 0; s < m_size; ++s)
    {
        carry.Clear();

        for (unsigned int p = 0; p < n_size; ++p)
        {
            u.Set(((*multiplicand)[s]) & (b[p]));

            unsigned int idx = p + s;
            if (idx >= result_size)
                continue; // or assert(false)

            t.Set(((result->b)[idx]) ^ u ^ carry);

            carry.Set((u & (((result->b)[idx]) | carry)) |
                      (((result->b)[idx]) & carry));

            ((result->b)[idx]).Set(t);
        }

        unsigned int idx = s + n_size;
        if (idx < result_size)
            ((result->b)[idx]).Set(carry);
    }
}

// Multiplies structures using temporary layouts acting as broadcast variables.
void UnsignedInt::MultiplyByInteger(unsigned long long int n, UnsignedInt* result) {
    UnsignedInt multiplicand(n);
    multiplicand.SetAll(n);  
    Multiply(&multiplicand, result);
}

// Multiplies the underlying values by shifting layouts leftwards.
//this method  multiplies *this by 2 and writes the result in *this
//inline
void UnsignedInt::MultiplyByTwoTo(void){

    if (b[b.size()-1].Get()!=0) b.push_back(Bits_zero);  //add a line of zero to create space for the shift if the last line is not free

    //to multiply by two, I shift all entries to the left by one place
    (*this) <<= (&Bits_one);
    //if (b.size() > 1) Normalize();   
    
}

// Divides the underlying values by shifting layouts rightwards.
//this method requires *this to be even, it divides *this by 2 and writes the result in *this
//inline
void UnsignedInt::DivideByTwoTo(void){
    
    //to divide by two, I shift all entries to the right by one place
    (*this) >>= (&Bits_one);
    
    
}

// ============================================================================
// Compound Assignment Operators
// ============================================================================

// Processes arithmetic assignments appending carry tracks onto block sequences.
// add addend to *this, and store the result in *this.
// This method requires this->GetSize() to be >= addend->GetSize()
void UnsignedInt::operator += (UnsignedInt* addend){
    Bits carry, t;
    AddTo(addend, &carry);
    // add the carry bit from the addition as a new entry in b
    // ******** THIS MAY BE TIME CONSUMING ********
    if(carry.Get()!=0){
        cout << "new size exceeds the max allocated size, issue (UnsignedInt addition)" << endl;
        b.push_back(carry);
    }
    // Only normalize if b has more than one entry: if b has exactly one entry,
    // normalizing would delete it when the value is 0, leaving b empty (GetSize()=0),
    //if (b.size() > 1) Normalize();
}

// Deducts items directly transforming internal profiles via two-complement additions.
//substract m to *this and write the result in *this
//DOES NOT WORK (PROBABLY)
void UnsignedInt::operator -= (UnsignedInt* subtrahend) {
    
    UnsignedInt subtrahend_t;
    
    subtrahend_t = (*subtrahend);
    
//    cout << "this:";
//    this->Print();
//
//    cout << "subtrahend:";
//    subtrahend.Print();
    
    //GIVEN THAT *THIS HAS BEEN RESIZED WITH ONE ADDITIONAL ENTRY AND THAT I WANT TO COMPUTE THE COMPLEMENT WITH RESPECT TO THE ACTUAL SIZE OF THIS (WITHOUT THE ADDITIONAL ENTRY) HERE I CALL  ComplementTo with argument (this->GetSize())-1 RATHER THAN WITH ARGUMENT (this->GetSize())
    subtrahend_t.ComplementTo((this->GetSize()));
    
//    cout << "subtrahend complement:";
//    subtrahend.Print();

    
    (*this) += (&subtrahend_t);
    
//    cout << "*this + subtrahend complement:";
//    this->Print();
 
    (*this) += (&UnsignedInt_one);
    
//    cout << "*this + subtrahend complement + 1:";
//    this->Print();

    
//    minuend = (minuend + subtrahend.Complement(minuend.GetSize()) + one);
    
//    cout << "minuend + subtrahend.Complement + 1 ";
//    minuend.Print();
 
    
    this->RemoveFirstSignificantBit();

//    cout << "[*this + subtrahend complement + 1 ]_ removed first significant bit:";
//    this->Print();

    
//    cout << "(minuend + ~subtrahend + 1 ).remove first digit";
//    minuend.Print();
 
    
}

// Expands value states tracking consecutive shifts inside active profiles.
//multiply *this by addend (as if they were two UnsignedInts)  and store the result in *this. This method requires this->GetSize() to be >= addend.GetSize(). once this method is called, *this has size [size of *this before the method is called] + multiplicand.GetSize()
//inline 
void UnsignedInt::operator *= (UnsignedInt* multiplicand){
    
    unsigned int s;
    const unsigned int multiplicant_size = multiplicand->GetSize();
    const unsigned int size = GetSize();
    UnsignedInt result, t;
    

    //THIS MAY SLOW DOWN THE CODE
    //resize *this and result in order to be large enough to host the result
    Resize(size + multiplicant_size);
    for(s=size-multiplicant_size; s<size; s++){
        b[s].SetAll(false);
    }
    result.Resize(size);
    //THIS MAY SLOW DOWN THE CODE
    

    for(s=0, result.SetAll(0); s<multiplicant_size; s++){
        //multiply by the s-th element of multiplicand: at each step of this loop *this is shifted by one unit to the left
        
        //the temporarly variable t is set equal to the original value of *this multiplyed by 2^s
        t = (*this);
        //I perform this & to multiply by the s-th bit of the multiplicand
        t &= &((*multiplicand)[s]);
        
        //add the partial sum to the result
        result += &t;
        
        //shift this
        (*this) <<= &Bits_one;

    }
    
    //during the for loop above, the line result += &t has uselessly increased the size of result -> THIS MAY SLOW DOWN THE CODE -> I resize result to the maximum size it can have after the multiplication 
    result.Resize(size);
    //result now is complete: set *this equal to result
    (*this) = result;
    
}

// Compiles basic bitwise tracking expansions inline directly.
// add addend to *this, and store the result in *this.
// This method requires this->GetSize() to be >= addend->GetSize()
void UnsignedInt::operator += (Bits* addend){
    Bits carry, t;
    AddTo(addend, &carry);
    // add the carry bit from the addition as a new entry in b
    // ******** THIS MAY BE TIME CONSUMING ********

    if(carry.Get()!=0){
        cout << "new size exceeds the max allocated size, issue (Bits addition)" << endl;
        b.push_back(carry);
    }
    
    
    // Only normalize if b has more than one entry: if b has exactly one entry,
    // normalizing would delete it when the value is 0, leaving b empty (GetSize()=0),
    //if (GetSize() > 1) Normalize(); //ISSUES WITH THIS CONDITION --> enters even when only one Bits, has to be checked
}

// ============================================================================
// Logical Comparisons
// ============================================================================

// Checks order from top-most indices to compare absolute value sizes down tracks.
//Compare *this with m and store the result in result. result is 1 if *this < m, and 0 otherwise
Bits UnsignedInt::operator < (const UnsignedInt& m){
    int s;
    Bits result, equal_so_far;
    
    int sizeA = GetSize();
    int sizeB = m.GetSize();
    int sizeMax = std::max(sizeA, sizeB);

    // Partir du bit de poids fort (ligne la plus haute)
    // Si une seule des deux a cette ligne, l'autre vaut 0 implicitement
    auto getA = [&](int i) -> Bits { return (i < sizeA) ? b[i] : Bits(0); };
    auto getB = [&](int i) -> Bits { return (i < sizeB) ? m.b[i] : Bits(0); };

    result      = (~getA(sizeMax-1)) & getB(sizeMax-1);
    equal_so_far = ~(getA(sizeMax-1) ^ getB(sizeMax-1));

    for(s = sizeMax-2; s >= 0; s--){
        result       = result | (equal_so_far & (~getA(s)) & getB(s));
        equal_so_far = equal_so_far & ~(getA(s) ^ getB(s));
    }
    return result;
}

// Inverts lesser-than conditions evaluating boundary values up to equality.
//Compare *this with m and store the result in result. result is 1 if *this <= m, and 0 otherwise
Bits UnsignedInt::operator <= (UnsignedInt& m){
    
    return(~(m < (*this)));
    
}

// ============================================================================
// Standard Arithmetic Evaluation Operators (Returning Copies)
// ============================================================================

// Evaluates combination summaries allocating separate instances based on capacities.
UnsignedInt UnsignedInt::operator+(UnsignedInt* addend) {
    UnsignedInt a;
    if (addend->GetSize() >= this->GetSize())
        a = *addend, a += this;
    else
        a = *this, a += addend;
    return a;
}

// Computes structural deductions producing unlinked standalone instances.
//return *this - m
UnsignedInt UnsignedInt::operator - (UnsignedInt* addend) {
    
    UnsignedInt t;
    
    t = (*this);
    t -= addend;

    return t;

}

// Builds calculated results keeping internal structures immutable.
//return *this + *addend and write the carry in *carry
UnsignedInt UnsignedInt::Add(UnsignedInt* addend, Bits* carry) {
    
    UnsignedInt a;
    
    a = (*this);
    a.AddTo(addend, carry);

    return a;

}

// Outputs differential configurations computing borrow dependencies externally.
//return *this - *subrahend and write the borrow in *borrow
UnsignedInt UnsignedInt::Substract(UnsignedInt* subtrahend, Bits* borrow) {
    
    UnsignedInt t;
    
    t = (*this);
    t.SubstractTo(subtrahend, borrow);

    return t;

}

// Adds the bitwise AND of a and mask to this UnsignedInt without creating temporary objects.
void UnsignedInt::AddAnd(UnsignedInt* a, Bits* mask){
    Bits carry;
    carry.Clear();

    const unsigned int size = a->GetSize();

    for (unsigned int p = 0; p < size; ++p)
    {
        Bits add_bit = a->b[p] & (*mask);

        Bits new_carry = (b[p] & add_bit) |
                         (b[p] & carry) |
                         (add_bit & carry);

        b[p] ^= &add_bit;
        b[p] ^= &carry;

        carry = new_carry;
    }
}


// Increment a bit-sliced counter by mask (per-lane 0/1), early-exiting once
// the carry has died out. Number of iterations needed adapts automatically
// to however many bit-planes 'counter' was sized with (i.e. to deg).
void UnsignedInt::IncrementMaskedFast(UnsignedInt& counter, Bits mask) {
    Bits carry = mask;
    Bits new_carry;
    const unsigned int L = counter.GetSize();       // = bits(deg), auto-derived
    for (unsigned int p = 0; p < L && carry.Get() != 0; ++p) {
        new_carry = counter[p] & carry;
        counter[p] ^= &carry;
        carry = new_carry;
    }
}