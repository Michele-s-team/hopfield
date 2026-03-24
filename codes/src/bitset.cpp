//
//  bitset.cpp
//  gillespie
//
//  Created by Michele on 12/02/2024.
//

#include "bitset.hpp"

#include "gsl_math.h"
#include <vector>

#include "lib.hpp"
#include "main.hpp"


//inline 
BitSet::BitSet(void){
    
    
}


//inline 
BitSet::BitSet(unsigned long long int N){
    
    b.resize(bits(N));
    
}


//set *this to zero
inline void BitSet::Clear(){
    
    for(unsigned int s=0; s<b.size(); s++){
        (b[s]).Set(0);
    }
    
}


//swap bit-by-bit the pair {*this, *a} if *check = true and write the result in {*this, *a}, and leave *this and *a unchanged if *check = false, where *work_space is a temporary variable needed to store stuff. This method requires *a and *work_space to be allocated and *this and *a to have the same size
inline void BitSet::Swap(BitSet* a, Bits& check, Bits* work_space){
    
    for(unsigned int s=0; s<GetSize(); s++){
        
        b[s].Swap(&((a->b)[s]), check, work_space);
        
    }
    
    
}


//inline
void BitSet::Resize(unsigned long long int size){
    
    b.resize(size);
    
}


//inline
unsigned int BitSet::GetSize(void){
    
    return ((unsigned int)(b.size()));
    
}


//initialize *this randomly
inline  void BitSet::SetRandom(unsigned int seed){
    
    gsl_rng* ran;
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);

    SetRandom(ran);
    
    gsl_rng_free(ran);
    
}


inline  void BitSet::SetRandom(gsl_rng* ran){
    
    unsigned int s, p;
    
    for(s=0; s<b.size(); s++){
        for(p=0; p<n_bits; p++){
            b[s].Set(p, (bool)(gsl_rng_uniform_int(ran, 2)));
        }
    }
    
}

//set all n_bits entries of *this to the respective bits of i. This method requires  *this to be properly sized to contain i
//inline 
void BitSet::SetAll(unsigned long long int i){
    
    unsigned int s;
    Bits m(i);
          
    //set the first bits(m.Get()) bits of *this equal to the bits of i
    for(s=0; s<bits(m.Get()); s++){
        (b[s]).SetAll(m.Get(s));
    }
    //set the remaining bits of *this, if any, to false (0)
    for(s=bits(m.Get()); s<GetSize(); s++){
        (b[s]).SetAll(false);
    }
    
}


//inline 
void BitSet::SetAllToSize(unsigned long long int i){
    
    for(unsigned int s=0; s<GetSize(); s++){
        (b[s]).SetAll( ((i >> s) & ullong_1) );
    }
    
}


//set all b[]s equal to m
//inline 
void BitSet::SetAll(Bits& m){
    
    for(unsigned int s=0; s<GetSize(); s++){
        b[s] = m;
    }
    
}


//set the first m->GetSize() entries of *this equal to the respective entries of *m. This method requires this->Getsize() to be >= m->GetSize()
inline void BitSet::Set(BitSet* m){
    
    unsigned int s;
    
    for(s=0; s<m->GetSize(); s++){
        b[s] = (m->b)[s];
    }
    for(; s<GetSize(); s++){
        b[s].SetAll(false);
    }
    
}


//set all n_bits entries of *this equal to the entries stored (in IEEE754 format) in the mantissa of x. This requires b.size() = n_bits_mantissa and work_space->size() = n_bits_mantissa
inline void BitSet::SetAllFromDoubleMantissa(double x, vector<bool>* work_space){
    
    
    GetMantissaFromDouble(work_space, x);
    
    for(unsigned int p=0; p<GetSize(); p++){
        b[p].SetAll((*work_space)[p]);
    }
    
    
}


//set the s-th bit entry of *this equal to the entries stored (in IEEE754 format) in the mantissa of x. This requires b to be properly sized
inline void BitSet::SetFromDoubleMantissa(unsigned int s, double x){
    
    vector<bool> v;
    
    GetMantissaFromDouble(&v, x);
    
    for(unsigned int p=0; p<GetSize(); p++){
        b[p].Set(s, v[p]);
    }
    
    v.clear();
    
}


//reize *this in order to contain all bits of i, and set all n_bits entries of *this to the respective bits of i
//inline 
void BitSet::ResizeAndSetAll(unsigned long long int i){
    
    Resize(bits(i));
    SetAll(i);
    
}


//inline 
void BitSet::Print(string title){
    
    unsigned int s;
    
    cout << title << endl;
    for(s=0; s<b.size(); s++){
        cout << "[" << s << "] = ";
        //extra space to align vertically the output
        if(s < 10){cout << " ";}
        b[s].Print("");
    }
    cout << endl;
    
}

//print *this to the output stream output_stream
//inline 
void BitSet::Print(ostream& output_stream){
        
    for(unsigned int s=0; s<GetSize(); s++){
        (b[s]).Print(output_stream);
        output_stream << "\t";
    }
    
}


//return (bit-by-bit) true if *this == m, and false otherwise. This method requires *this and m to have the same size
inline Bits BitSet::operator == (BitSet& m){
    
    unsigned int p;
    Bits result;
    
    if(GetSize() == m.GetSize()){
        //*this and m have the same size -> check if they are equal
        
        //run through all entries b[p], as soon as there is an entry of *this that is different form the corresponding entry of m, set result to false
        for(p=0, result.SetAll(true); p<GetSize(); p++){
            
            result &= (b[p] == (m.b)[p]);
            
        }
        
    }else{
        //*this and m have different sizes -> set the result equal to false
        
        result.SetAll(false);
        
    }
    
    return result;
    
}

//Confronto *this con m e scrivo in result il risultato. result è uguale a 1 se *this < m e a 0 altrimenti
inline Bits BitSet::operator < (const BitSet& m){
    
    int s;
    Bits result, check;
    
    
    result = (~(b[GetSize() - 1])) & ((m.b)[GetSize() - 1]);
    check = ((b[GetSize() - 1]) ^ ((m.b)[GetSize() - 1]));
    
    for(s=GetSize()-2; s >=0; s--){
        result = ((check & result) | ((~check) & ((~(b[s])) & ((m.b)[s]))) );
        check = (check | ((b[s]) ^ ((m.b)[s])));
    }
    
    return result;
    
}



//inline 
Bits BitSet::operator <= (BitSet& m){
    
    return(~(m < (*this)));
    
}


//shift bit-by-bit to the left the entries of  b[GetSize()-1], b[GetSize()-2] , ... b[0] in *this by *m (thus by either one position or zero positions), replace the remaining entries b[] by all zeros and write the result in *this
inline BitSet BitSet::operator << (Bits* m){
    
    BitSet t = (*this);
    t <<= m;
    return t;
    
}


//return the unsigned long long int written in the p-th bit of *this
//inline 
unsigned long long int BitSet::Get(unsigned int p){
    
    unsigned int s;
    unsigned long long int result;
    
    for(result=0, s=0; s<GetSize(); s++){
        result += two_pow(s) * (b[s].Get(p));
    }
    
    return result;
    
}


//overload of [] operator. IT IS IMPORTANT THAT THIS RETURNS A REFERENCE, NOT AN UnsignedInt: OTHERWISE THE RETURNED OBJECT, WHEN MODIFIED, WILL NOT CHANGE *this
inline Bits& BitSet::operator [] (const unsigned int& i){
    
    return((b[i]));
    
}



//return *this + *addend
inline BitSet BitSet::operator + (BitSet* addend) {
    
    BitSet a;
    
    a = (*this);
    a += addend;

    return a;

}


//return *this - m
inline BitSet BitSet::operator - (BitSet* addend) {
    
    BitSet t;
    
    t = (*this);
    t -= addend;

    return t;

}


//return *this + *addend and write the carry in *carry
inline BitSet BitSet::Add(BitSet* addend, Bits* carry) {
    
    BitSet a;
    
    a = (*this);
    a.AddTo(addend, carry);

    return a;

}



//return *this + *subrahend and write the borrow in *borrow
inline BitSet BitSet::Substract(BitSet* subtrahend, Bits* borrow) {
    
    BitSet t;
    
    t = (*this);
    t.SubstractTo(subtrahend, borrow);

    return t;

}


//add addend to *this,  and store the result in *this. This method requires this->GetSize() to be >= addend.GetSize()
inline void BitSet::operator += (BitSet* addend){
    
    Bits carry, t;
    
    AddTo(addend, &carry);
    
    //    add the last extra bit
    //******** THIS MAY BE TIME CONSUMING ********
    b.push_back(carry);
    
}


//same as BitSet::operator +=  but the last bit is not pushed back into b, but written into *carry. This method requires this->GetSize() to be >= addend->GetSize()
//inline 
void BitSet::AddTo(BitSet* addend, Bits* carry){
    
    Bits t;
    unsigned int p;

    
    for(p=0, carry->Clear();
        p<addend->GetSize();
        p++){
        //run over  bits of addend
        
        t.Set(((b[p]).Get()) ^ (((addend->b)[p]).Get()) ^ (carry->Get()));
        carry->Set(((((addend->b)[p]).Get()) & (((b[p]).Get()) | (carry->Get()))) | (((b[p]).Get()) & (carry->Get())));
        (b[p]).Set(t);
        
    }
    for(p=addend->GetSize(); p<GetSize(); p++){
        //run over the extra bits of augend
        
//        (t.Get()) = (((b[p]).Get()) ^ (carry->Get()));
        t.Set((b[p]) ^ (*carry));
//        (carry->Get()) = (((b[p]).Get()) & (carry->Get()));
        carry->Set((b[p]) & (*carry));
        (b[p]).Set(t);
        
    }
    
}

//remove useless bits on the tail of *this that contain all 0s
inline void BitSet::Normalize(void){
    
    int p;
    
    for(p=GetSize()-1; p>=0; p--){
        
        if((b[p]).Get() == 0){
            b.pop_back();
        }else{
            break;
        }
        
    }
    
}

//remove useless bits on the tail of *this that contain all 0s, as long as this deletion does not make the size of *this < than n
inline void BitSet::Normalize(unsigned int n){
    
    int p;
    
    for(p=GetSize()-1; p>=0; p--){
        
        if(((b[p]).Get() == 0) && (GetSize() >= n)){
            b.pop_back();
        }else{
            break;
        }
        
    }
    
}

//add bit-by-bit addend (which here is either 1 or 0) to *this and store the result in *this and the carry in *carry. This method requires this->GetSize() to be > 1
//inline 
void BitSet::AddTo(Bits* addend, Bits* carry){
        
    Bits t;
    unsigned int p;

    //sum the only bit of addend
    //    (carry->Get()) = ((addend->Get()) & ((b[0]).Get()));
    carry->Set((*addend) & (b[0]));
    
    //    (b[0]).Get() = (((b[0]).Get()) ^ (addend->Get()));
    (b[0]).Set((b[0]) ^ (*addend));
    
    for(p=1; p<GetSize(); p++){
        //run over the extra bits of augend
        
        t.Set((b[p]) ^ (*carry));
        carry->Set((b[p]) & (*carry));
        (b[p]).Set(t);
        
    }
        
    
}




//substract m to *this and write the result in *this
inline void BitSet::operator -= (BitSet* subtrahend) {
    
    
    BitSet subtrahend_t;
    
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
 
    (*this) += (&BitSet_one);
    
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


//do the same as BitSet::operator -= but without resizing operations. This method requires this->GetSize() to be >= addend.GetSize()
//inline 
void BitSet::SubstractTo(BitSet* subtrahend, Bits* borrow) {
    
    unsigned int p;
    Bits t;
    
    for(p=0, borrow->Clear();
        p<subtrahend->GetSize();
        p++){
        //run over  bits of subtrahend
        
        //        (t.Get()) = (((b[p]).Get()) ^ (((subtrahend->b)[p]).Get()) ^ (borrow->Get()));
        t.Set((b[p]) ^ ((subtrahend->b)[p]) ^ (*borrow));
        
        //        (borrow && (! minuend || subtrahend)) || (! minuend && subtrahend)
        borrow->Set(((*borrow) & ((~(b[p])) | ((*subtrahend)[p]))) | ((~(b[p])) & ((*subtrahend)[p])) );
        (b[p]).Set(t);
        
    }

    for(p=subtrahend->GetSize(); p<GetSize(); p++){
        //run over the extra bits of minuend
        
        t.Set((b[p]) ^ (*borrow));
        borrow->Set((~(b[p])) & (*borrow));
        (b[p].Set(t));
        
    }
    
}

//substract bit-by-bit subtrahend (which here is either 1 or 0) to *this and store the result in *this and the borrow in *borrow. This method requires this->GetSize() to be >= addend.GetSize()
//inline 
void BitSet::SubstractTo(Bits* subtrahend, Bits* borrow) {
    
    Bits t;
    unsigned int p;
    
    borrow->Set((~(b[0])) & (*subtrahend));
    (b[0]).Set((b[0]) ^ (*subtrahend));

    for(p=1; p<GetSize(); p++){
        //run over the extra bits of minuend
        
        t.Set((b[p]) ^ (*borrow));
        borrow->Set((~(b[p])) & (*borrow));
        (b[p]).Set(t);
        
    }
    
    
    
}


//write the one-complement of *this with respect to a size 'size' of the binary representation and write it into *this
inline void BitSet::ComplementTo(unsigned int size){
    
    unsigned int s;
    
    
    //set the first bits common to *this
    for(s=0; s<GetSize(); s++){
        b[s] = (b[s]).Complement();
    }
    
    Resize(size);

    //set the remaining bits equal to one 
    for(; s<GetSize(); s++){
        b[s] = Bits_one;
    }
    
}


//write the one-complement of *this  and write it into *this
inline void BitSet::ComplementTo(void){
    
    for(unsigned int s=0; s<GetSize(); s++){
        (b[s]).ComplementTo();
    }
    
}



//set to zero the first bits of *this that is equal to one (starting from the last bit) and leave the others unchanged, write the result in *this
inline void BitSet::RemoveFirstSignificantBit(void){
    
    int s;
    Bits check_old, check_new;
    
//    cout << " before the first significant bit is set to 0:";
//    Print();

    //remove the last significant bit in minuend
    for(s=(this->GetSize())-1, check_old.SetAll(0); s>=0; s--){
        
        check_new.Set(check_old | ((*this)[s]));
        
        (b[s]).Set(((*this)[s]) & check_old & check_new);
        
        check_old = check_new;
        
    }
    
//    cout << " after the first significant bit is set to 0:";
//    Print();
    

}



//shift bit-by-bit to the right the entries of  b[GetSize()-1], b[GetSize()-2] , ... b[0] in *this by *l (thus by either one position or zero positions) and replace the remaining entries b[] by all zeros
//inline 
void BitSet::operator >>= (Bits* l){
    
    int m;
        
    //run through the components of this->b and shift them
    //in this first loop, I run over the first chunk of entries of b: m = 0, ..., b.size() - 2^n-1 and I replace the m-th component of b with the m+2^n-th compoennt if e[n]=true, and do nothing otherwise
    for(m=0; m<((int)GetSize())-1; m++){
        
        b[m].Replace(
                     //the element # m+1 in b
                     (b.data()) + (m+1),
                     l
                     );
        
    }
    
    //I consider the last entry of b: I can no longer replace it with b[m+1] as in the loop above, becuase b[m+1] does not exist in b -> I replace it with zero
    b.back().Replace(
                     //a Bits filled with zeros
                     &Bits_zero,
                     l
                     );
    
}



//shift bit-by-bit to the left the entries of  b[GetSize()-1], b[GetSize()-2] , ... b[0] in *this by *l (thus by either one position or zero positions), replace the remaining entries b[] by all zeros and write the result on *this
//inline 
void BitSet::operator <<= (Bits* l){
    
    int m;
    
    
    //run through the components of this->b and shift them
    //in this first loop, I run over the first chunk of entries of b:  and I replace  if e=true, and do nothing otherwise
    for(m=(this->GetSize())-1; m>0; m--){
        
        b[m].Replace(
                     //the element # m+1 in b
                     (b.data()) + (m-1),
                     l
                     );
        
    }
    
    //I consider the first entry (b[0]) of b: I can no longer replace it with b[-1] as in the loop above, becuase b[-1] does not exist in b -> I replace it with zero
    b.front().Replace(
                     //a Bits filled with zeros
                     &Bits_zero,
                     l
                     );
    
}


//perform (bit-by-bit) an & between  b[s] and *m ,and write the result in b[s] for all s = 0 ... GetSize()
//inline 
void BitSet::operator &= (Bits* m){
      
    AndTo(m, 0, GetSize());
    
}


//perform (bit-by-bit) an & between  b[s] and *m ,and write the result in b[s] for all s = start, ..., end-1
inline void BitSet::AndTo(Bits* m, unsigned int start, unsigned int end){
    
    for(unsigned int s=start; s<end; s++){
        
        b[s] &= m;
        
    }
    
}


//perform (bit-by-bit) an & between  b[s] and *m ,and write the result in (result->b)[s] for all s
//inline 
void BitSet::And(Bits* m, BitSet* result){
    
    for(unsigned int s=0; s<GetSize(); s++){
        
        (result->b)[s] = (b[s] & (*m));
        
    }
    
}


//multiply *this by addend (as if they were two UnsignedInts)  and store the result in *this. This method requires this->GetSize() to be >= addend.GetSize(). once this method is called, *this has size [size of *this before the method is called] + multiplicand.GetSize()
//inline 
void BitSet::operator *= (BitSet* multiplicand){
    
    unsigned int s;
    BitSet result, t;
    

    //THIS MAY SLOW DOWN THE CODE
    //resize *this and result in order to be large enough to host the result
    Resize(GetSize() + (multiplicand->GetSize()));
    for(s=GetSize()-(multiplicand->GetSize()); s<GetSize(); s++){
        b[s].SetAll(false);
    }
    result.Resize(GetSize());
    //THIS MAY SLOW DOWN THE CODE
    

    for(s=0, result.SetAll(0); s<multiplicand->GetSize(); s++){
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
    result.Resize(GetSize());
    //result now is complete: set *this equal to result
    (*this) = result;
    
}




/*multiply *this by *multiplicand and write the result in *result

 
 result->GetSize() <= (this-GetSize()) + (multiplicand->GetSize(), and when I call this method I take result->GetSize() = (this-GetSize()) + (multiplicand->GetSize() to be safe. Thus when this method is called, result->GetSize() must be equal to (this-GetSize()) + (multiplicand->GetSize()

 the times are from  ./main.o -s 0 -S 6
 */
//inline
void BitSet::Multiply(UnsignedInt* multiplicand, UnsignedInt* result){
    
    unsigned int s, p;
    Bits carry, t, u;
        
    
    for(s=0, result->SetAll(Bits_zero); s<multiplicand->GetSize(); s++){
        //multiply by the s-th element of multiplicand: at each step of this loop *this is shifted by s places to the left and added to the result
        
        for(p=0, carry.Clear(); p<GetSize(); p++){
            
            u.Set(((*multiplicand)[s]) & (b[p]));
            t.Set(((result->b)[p+s]) ^ u ^ carry);
            
            carry.Set((u & (((result->b)[p+s]) | carry)) | (((result->b)[p+s]) & carry));
            ((result->b)[p+s]).Set(t);
            
        }
        ((result->b)[p+s]).Set(carry);

    }
    
}


//this method requires *this to be even, it divides *this by 2 and writes the result in *this
//inline
void BitSet::DivideByTwoTo(void){
    
    //to divide by two, I shift all entries to the right by one place
    (*this) >>= (&Bits_one);
    
    
}

//this method  multiplies *this by 2 and writes the result in *this
//inline
void BitSet::MultiplyByTwoTo(void){
    
    //to divide by two, I shift all entries to the right by one place
    (*this) <<= (&Bits_one);
    
    
}



//I am obliged to put this method definition here, because this method needs the full declaration of the UnsignedInt class before it is declared
//shift bit-by-bit the entries of  b[51] b[50] ... in *this by a number of positions to the right (>>) encoded in *e and replace the remaining entries b[] by all zeros
//inline
void BitSet::operator >>=(UnsignedInt* e){
    
    unsigned int n;
    int m;
    Bits zero;
    
    zero.SetAll(false);
    
    
    for(n=0; n<(e->GetSize()); n++){
        //shift by 2^n positions according to e[n]
        
        //run through the components of this->b and shift them
        
        //in this first loop, I run over the first chunk of entries of b: m = 0, ..., b.size() - 2^n-1 and I replace the m-th component of b with the m+2^n-th compoennt if e[n]=true, and do nothing otherwise
        for(m=0; m<GetSize()-gsl_pow_int(2, n); m++){
            
            b[m].Replace(
                         //the element # m+2^n in b
                         &(b[m+gsl_pow_int(2, n)]),
                         //the element #n in e
                         &((e->b)[n])
                         );
            
            
        }
        
        //in this loop, I run over the second chunk of entries of b: m =b.size() - 2^n , ..., b.size(). I can no longer replace b[m] with b[m+2^n] as in the loop above, becuase b[m+2^n] does not exist in b -> I replace it with zero. Note that if 2^n > b.size(), this second loop must cover the entire vector b -> I set as starting value of m  max(((int)GetSize())-((int)gsl_pow_int(2, n)), 0)
        for(m = max(((int)GetSize())-((int)gsl_pow_int(2, n)), 0); (m<((int)GetSize())); m++){
            
            b[m].Replace(
                         //a Bits filled with zeros
                         &zero,
                         //the element #n in e
                         &((e->b)[n])
                         );
            
            
        }
        
    }
    
    
}


//shift bit-by-bit the entries of  b[51] b[50] ... in *this by a number of positions to the left (<<) encoded in *e and replace the remaining entries b[] by all zeros
//inline
void BitSet::operator <<= (UnsignedInt* e){
    
    unsigned int n;
    int m;
    Bits zero;
    
    zero.SetAll(false);
    
    
    for(n=0; n<(e->GetSize()); n++){
        //shift by 2^n positions according to e[n]
        
        //run through the components of this->b and shift them
        
        //in this first loop, I run over the first chunk of entries of b: m = b.size()-1, ..., 2^n and I replace the m-th component of b with the m-2^n-th compoennt if e[n]=true, and do nothing otherwise
        for(m=GetSize()-1; m>=gsl_pow_int(2, n); m--){
            
            b[m].Replace(
                         //the element # m+2^n in b
                         &(b[m-gsl_pow_int(2, n)]),
                         //the element #n in e
                         &((e->b)[n])
                         );
            
            
        }
        
        //in this loop, I run over the second chunk of entries of b: m =2^n-1 , ..., 0. I can no longer replace b[m] with b[m-2^n] as in the loop above, becuase b[m-2^n] does not exist in b -> I replace it with zero. Note that if 2^n > b.size(), this second loop must cover the entire vector b -> I set as starting value of m  max(((int)(this->GetSize()))-((int)gsl_pow_int(2, n)), 0)
        for(m = min(((int)GetSize())-1, (((int)gsl_pow_int(2, n))-1)); m>=0; m--){
            
            b[m].Replace(
                         //a Bits filled with zeros
                         &zero,
                         //the element #n in e
                         &((e->b)[n])
                         );
            
            
        }
        
    }
    
}


//return the position of the first signigificant bit in *this, starting from the last element of  b[]
inline UnsignedInt BitSet::PositionOfFirstSignificantBit(void){
    
    int s;
    Bits check_old, check_new, t, carry;
    //result must be big enough to host an unsigned int equal to this->GetSize()
    UnsignedInt result(GetSize());
    
    check_old.SetAll(0);
    result.SetAll(0);
    
    for(s=GetSize()-1; s>=0; s--){
        
        check_new = check_old | ((*this)[s]);
        t = (~check_new);
        result.AddTo(&t, &carry);
        
        check_old = check_new;
        
    }
    
    return result;
    
    
}
