//
//  double.cpp
//  gillespie
//
//  Created by Michele on 10/02/2024.
//

#include "double.hpp"

#include <stdint.h>

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"

//default constructor
inline Double::Double(void){
    
    b.Resize(52);
    e.Resize(11);
    
}

//set the fraction to zero, the exponent to 1023 and the sign to +
inline void Double::Clear(){
    
    unsigned int p;
    
    //set the fraction to zero
    for(p=0; p<b.GetSize(); p++){
        (b[p]).Set(0);
    }
    
    //set the exponent to 1023
    for(p=0; p<e.GetSize()-1; p++){
        (e[p]).SetAll(true);
    }
    e.b.back().SetAll(false);
    
    //set the sign to +
    s.SetAll(false);
    
}


//replace bit-by-bit *this with *replacer if *check=true, and leave *this unchanged otherwise
inline void Double::Replace(Double* replacer, Bits* check){
    
    unsigned int p;
    
    //replace the mantisssa
    for(p=0; p<(b.GetSize()); p++){
        b[p].Replace((replacer->b.b.data()) + p, check);
    }
    
    //replace the exponent
    for(p=0; p<(e.GetSize()); p++){
        e[p].Replace((replacer->e.b.data()) + p, check);
    }
    
    //replace the sign
    s.Replace(&(replacer->s), check);
    
}


//swap bit-by-bit the pair {*this, *a} if *check = true and write the result in {*this, *a}, and leave *this and *a unchanged if *check = false, where *work_space is a temporary variable needed to store stuff. This method requires *a and *work_space to be allocated and *this and *a to have the same size
inline void Double::Swap(Double* a, Bits& check, Bits* work_space){
    
    //swap the sign
    s.Swap(&(a->s), check, work_space);
    //swap the exponent
    e.Swap(&(a->e), check, work_space);
    //swap the mantissa
    b.Swap(&(a->b), check, work_space);
    
}

//initialize *this randomly with seed seed
inline void Double::SetRandom(unsigned int seed){
    
    gsl_rng* ran;
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    SetRandom(ran);
    
    gsl_rng_free(ran);
    
}





//initialize *this randomly with the (already initialized) random generator *ran
inline void Double::SetRandom(gsl_rng* ran){
    
    unsigned int i;
    
    for(i=0; i<b.GetSize(); i++){
        b[i].SetRandom(ran);
    }
    for(i=0; i<e.GetSize(); i++){
        e[i].SetRandom(ran);
    }
    s.SetRandom(ran);
    
}


void Double::Print(string title){
    
    cout << title << " :" << endl;
    s.Print("s");
    e.Print("e");
    b.Print("b");
    
    
    
}



//set all entries of *this according to the double x, where x is written in binary according to the IEEE754 convention
inline void Double::SetAll_IEEE754(double x){
    
    uint8_t *bytePointer = (uint8_t *)&x;
    size_t index;
    uint8_t byte;
    int bit;
    unsigned int p;
    
    for(index=0, p=0; index < sizeof(double); index++){
        
        byte = bytePointer[index];
        
        for(bit = 0; bit < 8; bit++, p++){
            
            //            printf("%d", byte & 1);
            
            if(p < b.GetSize()){
                
                b[p].SetAll(((bool)(byte & 1)));
                
            }else{
                
                if(p < b.GetSize()+e.GetSize()){
                    
                    //                    int temp;
                    //                    bool bb;
                    
                    //                    temp = p-(b.size());
                    //                    bb = ((bool)(byte & 1));
                    
                    //                    cout << "\nbefore";
                    //                    e[temp].Print();
                    
                    e[p-(b.GetSize())].SetAll(((bool)(byte & 1)));
                    //                    e[temp].SetAll(true);
                    //                    cout << "\nafter";
                    //                    e[temp].Print();
                    
                }else{
                    
                    s.SetAll(((bool)(byte & 1)));
                    
                    
                }
                
                
            }
            
            byte >>= 1;
        }
    }
    
}


//set all the n_bits entries of *this equal to the double given by (-1)^sign * 2^{exponent - 1023} * (mantissa[0] 2^0 + mantissa[1] 2^{-1} + ... ). Here exponent must be
inline void Double::SetAll(bool sign, unsigned long long int exponent,  BitSet& mantissa){
    
    if((exponent < two_pow(n_bits_exponent+1)) && (mantissa.GetSize() == n_bits_mantissa)){
        
        s.SetAll(sign);
        e.SetAll(exponent);
        b = mantissa;
        
    }else{
        
        cout << "Double::SetAll error: exponent and mantissa do not have the right range/size!!" << endl;
        
    }
    
}


//set all the n_bits entries of s equal to sign, all n_bits entries of e equal to exponent, and all n_bits entries of b from the IEEE754 entries of mantissa
inline void Double::SetAll(bool sign, unsigned long long int exponent,  double mantissa){
    
    if(exponent < two_pow(n_bits_exponent+1)){
        
        //BOTTLENECK
        vector<bool> work_space(n_bits_mantissa);
        //BOTTLENECK

        s.SetAll(sign);
        e.SetAll(exponent);
        b.SetAllFromDoubleMantissa(mantissa, &work_space);
        
        work_space.clear();
        
    }else{
        
        cout << "Double::SetAll error: exponent does not have the right range!!" << endl;
        
    }
    
}


//set the p-th bit entry of s equal to sign, the p-th entry of e equal to exponent, and the p-th entry of b from the IEEE754 entries of mantissa
inline void Double::Set(unsigned int p, bool sign, unsigned long long int exponent,  double x){
    
    if(exponent < two_pow(n_bits_exponent+1)){
        
        s.Set(p, sign);
        e.Set(p, exponent);
        b.SetFromDoubleMantissa(p, x);
        
    }else{
        
        cout << "Double::SetAll error: exponent does not have the right range!!" << endl;
        
    }
    
}



//print *this in base 10 according to the IEEE754 convention
inline void Double::PrintBase10_IEEE754(void){
    
    unsigned int i, p;
    double b_10, e_10/*, A*/;
    
    cout << "Double in base 10: {";
    for(p=0; p<n_bits; p++){
        
        for(b_10=1.0, i=0; i<b.GetSize(); i++){
            b_10 += gsl_pow_int(2.0, -(i+1)) * (b[b.GetSize()-1-i].Get(p));
        }
        for(e_10=-1023.0, i=0; i<e.GetSize(); i++){
            e_10 += two_pow(i) * (e[i].Get(p));
        }
        
        cout << gsl_pow_int(-1.0, s.Get(p)) * gsl_pow_int(2.0, e_10) * b_10 << " ";
        
    }
    
    cout << "}\n";
    
}


//print *this in base 10 according to the my convention, where the mantissa is \sum_{i=0}^{52-1} b_{52-1-i} 2^{-i}
inline void Double::PrintBase10(string title){
    
    unsigned int p;
    vector<double> v;
    
    cout << title << ": {";
    for(GetBase10(v), p=0; p<n_bits; p++){
        
        cout << v[n_bits-1-p] << " ";
        
    }
    
    cout << "}" << endl;
    
}


//convert bit-by-bit *this in base 10 and write the result in v, which is resized
inline void Double::GetBase10(vector<double>& v){
    
    unsigned int i, p;
    double b_10, e_10;
    
    
    for(v.resize(n_bits), p=0; p<n_bits; p++){
        
        for(b_10=0.0, i=0; i<b.GetSize(); i++){
            b_10 += gsl_pow_int(2.0, -i) * (b[b.GetSize()-1-i].Get(p));
        }
        for(e_10=-1023.0, i=0; i<e.GetSize(); i++){
            e_10 += two_pow(i) * (e[i].Get(p));
        }
        
        v[p] = gsl_pow_int(-1.0, s.Get(p)) * gsl_pow_int(2.0, e_10) * b_10;
        
    }
    
}


//sum *this to addend and write the result in *this. For the time being, this method assumes that this->s 0 = all_0 and addend.s = all_0 (*this and x contain all non-negative numbers)
inline void Double::operator += (Double* addend){
    
    Double augend_t, addend_t;
    Bits compare, t, carry;
    UnsignedInt de;
    
    
    //set augend and addend, compare  bit-by-bit  the exponent of augend and the exponent of addend and write the result in compare
    //    augend = (*this);
    addend_t = (*addend);
    compare = (e < (addend_t.e));
    
    
    //swap bit-by-bit (augend.e) and (addend.e) in such a way that (augend.e) >= (addend.e)
    augend_t = (*this);
    this->Replace(addend, &compare);
    addend_t.Replace(&augend_t, &compare);
    
    //now augend.e >= addend.e
    
    
    //    cout << "Augend: " << endl;
    //    augend.Print();
    //    augend.PrintBase10();
    //    cout << "Addend: " << endl;
    //    addend.Print();
    //    addend.PrintBase10();
    
    de = e - (&(addend_t.e));
    
    //    de.PrintBase10("de");
    
    
    //    cout << "Addend.b before shift: "<< endl;
    //    addend.Print();
    
    //shift the mantissa of b by the different between the two exponents in order to cast addend in a form in which is can be easily added to augend
    (addend_t.b) >>= (&de);
    
    //    cout << "Addend.b after shift: "<< endl;
    //    addend.Print();
    
    
    //    cout << "***** Before +=: " << endl;
    //    cout << "Augend:" << endl;
    //    augend.Print();
    //    augend.PrintBase10();
    //    cout << "Addend:" << endl;
    //    addend.Print();
    //    addend.PrintBase10();
    
    //now sum augend.b and addend.b
    
    b += (&(addend_t.b));
    
    //    cout << "***** After += : " << endl;
    //    cout << "Augend:" << endl;
    //    augend.Print();
    //    augend.PrintBase10();
    
    
    
    //    cout << "***** Before augend.e+= carry: " << endl;
    //    cout << "Augend.e:" << endl;
    //    augend.e.Print();
    //    augend.e.PrintBase10();
    
    
    //the operation augend.b += addend.b adds an extra bit to augend.b (the carry) -> this extra bit must be removed and re-incorporated into augend.e
    //incoroprate the extra bit into the exponent
    //    Bits temp;
    //    temp = (augend.b.b.back());
    
    e.AddTo(&(b.b.back()), &carry);
    
    //    cout << "***** After augend.e+= carry: " << endl;
    //    cout << "Augend.e:" << endl;
    //    augend.e.Print();
    
    //the last entry of augend.e must be zero (unless the sum reaches overflow)
    //    augend.e.b.pop_back();
    
    
    //    augend.e.PrintBase10();
    
    
    
    //    cout << "Before << :" << endl;
    //    cout << "Augend.b: " << endl;
    //    augend.b.Print();
    
    //shift the mantissa of the augend if the carry is nonzero, and leave it unchanged otherwise
    t = (~(b.b.back()));
    b <<= (&t);
    b.b.erase(b.b.begin());
    
    
    //    cout << "After << :" << endl;
    //    cout << "Augend.b: " << endl;
    //    augend.b.Print();
    
    Normalize();
    //    (*this) = augend;
    
    
}

//sum *this to addend and write the result in *result, which need to be already allocated. For the time being, this method assumes that this->s 0 = all_0 and x.s = all_0 (*this and x contain all non-negative numbers). THIS METHOD ALTERS THE CONTENT OF *ADDEND
//I wrote the time taken by each line when running the program with  ./main.o -s 0 -S 5
inline void Double::AddTo(Double* addend){
    
    Bits compare, borrow, carry_b, t;
    UnsignedInt de;
    
    
    //set augend and addend, compare  bit-by-bit  the exponent of augend and the exponent of addend and write the result in compare
    //~ 1e-3 s
    compare = (e < (addend->e));
    
    
    //swap bit-by-bit (*this) and (addend) in such a way that (this->e) >= (addend->e)
    //~ 1e-3 s
    Swap(addend, compare, &t);

    //~ 1e-3 s
    de = e.Substract(&(addend->e), &borrow);
    
    //shift the mantissa of addend->b by the different between the two exponents in order to cast *addend in a form in which is can be easily added to *this = augend
    //BOTTLENECK #1: 2.49751000e-01s
    (addend->b) >>= (&de);
    //BOTTLENECK #1:

    
    //now sum augend.b and addend.b
    //~ 4e-3 s
    b.AddTo(&(addend->b), &carry_b);
    
    //the operation augend.b += addend.b adds an extra bit to augend.b (the carry) -> this extra bit must be removed and re-incorporated into augend.e
    //incoroprate the extra bit into the exponent
    //~ 0 s
    e.AddTo(&carry_b, &t);
    
    //the last entry of augend.e must be zero (unless the sum reaches overflow)
    
    /*
     shift the mantissa of the augend if the carry is nonzero, and leave it unchanged otherwise.
     To achieve this, the (bit-by-bit) desired result for b is : 1) If carry_b = true: b = {carry_b, b[51], ..., b[1]}, 2) If carry_b = false : b = {b[51], ..., b[0]}.
    I obtain this result for b with the two folloing lines, which avoid a (very slow) Resize() of b
     */
    //~ 0 s
    b >>= (&carry_b);
    //~ 0 s
    b.b.back().Replace(&carry_b, &carry_b);
  
}


//multiply *this by x and store the result in *this
inline void Double::operator *= (Double& x){
    
    UnsignedInt t(1022);
    
    t.SetAll(1022);
    
    //multiply the sign of *this by the sign of x
    s ^= (&(x.s));
    
    //multiply the exponent parts
    e += (&(x.e));
    e -= (&t);
    //the += above may have increased the size of e -> bring it back to the correct value for the exponent of a Double
    e.Normalize(n_bits_exponent);
    
    
    //multiply the mantissas
    b *= (&(x.b));
    
    b.b.erase(b.b.begin(), b.b.begin() + n_bits_mantissa);
    
}


//return (bit-by-bit) 1 if *this < x, 0 otherwise. This method assumes that this->s = x.s = 0
inline Bits Double::operator < (Double& x){
    
    //normalize the two Double(s) to make sure that the mantisa is of the form 1+....
    Normalize();
    x.Normalize();
    
    return((/*this is to compare the mantissa of *this and the mantissa of x, by treating them as if they were two unsigned ints*/(b < x.b) & (e == x.e)) | (e < x.e));
    
}


//normalize *this by shifting the mantissa in such a way that its first bit is nonzero, and re-incorporrating the shift in e
inline void Double::Normalize(void){
    
    UnsignedInt n;
    
    n = b.PositionOfFirstSignificantBit();
    
    //subtract n from e: this subtraction may have increased the size of e with some additional entries containing all 0s -> Normalize e in order to delete these entries and  bring the size of e back to its standard value n_bits_exponent
    e -= &n;
    e.Normalize(n_bits_exponent);
    
    b <<= &n;
    
}
