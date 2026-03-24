//
//  lib.cpp
//  gillespie
//
//  Created by Michele on 13/02/2024.
//

#include "lib.hpp"

#include <vector>
#include <stdint.h>

#include "double.hpp"
#include "fraction.hpp"
#include "main.hpp"
#include "gsl_math.h"


//inline
unsigned long long int two_pow(unsigned long long int i){
    
    return ((unsigned long long int)gsl_pow_int(2.0, ((int)i)));
    
}

//return the number of bits necessary to write n in base 2
unsigned int bits(unsigned long long int n){
    
    unsigned int s;
    
    for(s=0; two_pow(s) <= (unsigned long long int)n; s++){}
    
    return s;
    
}


//print bit-by-bit the double x
inline void bitwise_print_double(const double& x){
    
    uint8_t *bytePointer = (uint8_t*)&x;
    size_t index;
    uint8_t byte;
    int bit;
    unsigned int p;
    vector<bool> result(n_bits);
    
    for(index=0, p=0; index < sizeof(double); index++){
        
        byte = bytePointer[index];
        
        for(bit = 0; bit < 8; bit++, p++){
            
            result[p] = ((bool)(byte & 1));
            byte >>= 1;
            
        }
        
    }
    
    
    cout << "double:";
    for(p=0; p<n_bits; p++){
        
        switch (p) {
            case 0:
                cout << "\n\tSign: ";
                break;
                
            case 1:
                cout << "\n\tExponent: ";
                break;
                
            case 12:
                cout << "\n\tMantissa: ";
                break;
                
            default:
                break;
        }
        
        cout << result[n_bits-1-p];
        
    }
    
    cout << "\n";
    
    
}


//write the mantissa of x into *result by resizing *result properly. This method requires that result->size() = n_bits_mantissa
inline void GetMantissaFromDouble(vector<bool>* result, double x){
    
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




//    Fraction r(n_bits_mantissa);
//    UnsignedInt n(16), result;
//
//    result.Resize(r.GetSize()+n.GetSize());
//    r.SetRandom((unsigned int)0);
//    n.SetRandom((unsigned int)1);
//
//    r.PrintBase10("r");
//    n.PrintBase10("n");
//
//
//
//    r.FloorMultiply(&n, &result);
//
//    result.PrintBase10("result");


//test for Double::operator <
/*
 Double a, b;
 unsigned int s, i;
 gsl_rng* ran;
 Bits smaller;
 vector<double> v_a, v_b;
 bool it_works;
 
 
 ran = gsl_rng_alloc(gsl_rng_gfsr4);
 gsl_rng_set(ran, seed);
 
 
 for(it_works=true, s=0; s<1000; ++s){
 
 for(i=0; i<n_bits; i++){
 a.Set(i, false, 1023 + (128/2 - gsl_rng_uniform_int(ran, 128)), gsl_rng_uniform(ran));
 b.Set(i, false, 1023 + (128/2 - gsl_rng_uniform_int(ran, 128)), gsl_rng_uniform(ran));
 }
 
 a.GetBase10(v_a);
 b.GetBase10(v_b);
 
 cout << "a : " << endl;
 //    a.Print();
 a.PrintBase10();
 cout << "b : " << endl;
 //    b.Print();
 b.PrintBase10();
 smaller  = (a < b);
 cout << "a<b : " << endl;
 smaller.Print();
 
 
 
 for(i=0; i<n_bits; ++i){
 cout << "\t" << (v_a[i] < v_b[i]) << "\t" << smaller.Get(i) << endl;
 if((v_a[i] < v_b[i]) != smaller.Get(i)){it_works = false; break;}
 }
 
 }
 
 
 cout << "Check of the result:" << it_works << endl;
 */




//

//test for == for Unsigned Ints
/*
 UnsignedInt a(10), b(10);
 Bits check;
 a.SetRandom((unsigned int)0);
 b.SetRandom((unsigned int)3);
 cout << "a : " << endl;
 a.PrintBase10();
 cout << "b : " << endl;
 b.PrintBase10();
 
 
 check = (a == b);
 
 int x;
 x=0;
 cout <<  x;
 */


//test for <<= and >>=
/*
 BitSet n(80);
 UnsignedInt shift(12);
 n.SetRandom((unsigned int)0);
 shift.SetRandom((unsigned int)0);
 cout << "shift:" << endl;
 shift.PrintBase10();
 
 cout << "n before >>= :" << endl;
 n.Print();
 
 n >>= &shift;
 
 cout << "n after >>= :" << endl;
 n.Print();
 */


/*
 //test for Double::Normalize
 Double x;
 x.SetRandom((unsigned int)0);
 cout << "x before noramlize:";
 //    x.Print();
 x.PrintBase10();
 x.Normalize();
 cout << "x after noramlize:";
 x.PrintBase10();
 */


//
//test for UnsignedInt::Normalize
//    UnsignedInt x(10), one(1);



/*
 //test for Bits::Swap
 Bits a, b, check, w;
 a.SetRandom((unsigned int)0);
 b.SetRandom((unsigned int)1);
 check.SetRandom((unsigned int)2);
 
 a.Print("a");
 b.Print("b");
 check.Print("c");
 a.Swap(&b, check, &w);
 
 
 a.Print("a");
 b.Print("b");
 */

//test for BitSet::Swap
/*
 BitSet a(4), b(4);
 Bits check, w;
 a.SetRandom((unsigned int)0);
 b.SetRandom((unsigned int)1);
 check.SetRandom((unsigned int)2);
 
 a.Print("a");
 b.Print("b");
 check.Print("      c");
 
 a.Swap(&b, check, &w);
 
 
 a.Print("a");
 b.Print("b");
 
 */


//test for Double::Swap
/*
 Double a, b;
 Bits check, w;
 a.SetRandom((unsigned int)0);
 b.SetRandom((unsigned int)1);
 check.SetRandom((unsigned int)2);
 
 a.PrintBase10("a");
 b.PrintBase10("b");
 check.Print("      c");
 
 a.Swap(&b, check, &w);
 
 
 a.PrintBase10("a");
 b.PrintBase10("b");
 */




//speed test of UnsignedInt::AddTo method, with S samples and seed seed
inline void SpeedTestUnsignedIntAddto(unsigned long long int maximum_value, unsigned long long int S, unsigned long long int seed){
        
    
    cout << " ***************************** Speed test for UnsignedInt::AddTo *****************************" << endl;
    
    clock_t start=0, end=0;
    gsl_rng* ran;
    unsigned int i;
    unsigned long long int s, r=0;
    double x = 0.0;
    Bits carry;
    
    vector<UnsignedInt> A(S), B(S);
    vector<unsigned int> a(n_bits*S), b(n_bits*S);
    
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    
    //****************** calculation without bits ******************
    for(s=0; s<n_bits*S; s++){
        
        r = gsl_rng_uniform_int(ran, maximum_value);
        a[s] = (unsigned int)r;
        if(r!=0){
            b[s] = (unsigned int)(r-1-gsl_rng_uniform_int(ran, r));
        }else{
            b[s] = 0;
        }
        
    }
    
    
    start = clock();
    for(s=0; s<n_bits*S; s++){
        
        //this simulates the drawing of the random number for the Gillespie algorithm
        r = gsl_rng_uniform_int(ran, maximum_value);
        x = gsl_rng_uniform(ran);
        //        (a[s]) += b[s];
        (a[s]) -= b[s];
        
    }
    end = clock();
    cout  << "Time for n_bits*S [random number + operation]s without bits = "  << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << " s" << endl;
    
    
    //****************** calculation with bits ******************
    for(s=0; s<S; s++){
        
        A[s] = UnsignedInt(maximum_value);
        B[s] = UnsignedInt(maximum_value);
        
    }
    
    for(s=0; s<S; s++){
        
        for(i=0; i<n_bits; i++){
            
            r = gsl_rng_uniform_int(ran, maximum_value);
            A[s].Set(i, r);
            if(r!=0){
                B[s].Set(i, r-1-gsl_rng_uniform_int(ran, r));
            }else{
                B[s].Set(i, 0);
            }
            
        }
        
    }
    
    
    start = clock();
    for(s=0; s<S; s++){
        
        //        A[s].PrintBase10("A");
        //        B.PrintBase10("B");
        
        //this simulates the drawing of the random number for the Gillespie algorithm
        r = gsl_rng_uniform_int(ran, maximum_value);
        x = gsl_rng_uniform(ran);
        //        (A[s]) += (&B[s]);
        //        (A[s]).AddTo(&(B[s]), &carry);
    //                (A[s]) -= (&B[s]);
        (A[s]).SubstractTo(&(B[s]), &carry);
        
        //        A[s].PrintBase10("A");
        
    }
    end = clock();
    
    cout << "Time for S [random number + operation]s with bits = " << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << "s" <<  endl << endl;
    
    
    //without this the for loop will not be exectued with -O3
    cout << endl;
    A.back().PrintBase10(cout);
    cout << "dummy print: a = " << a[S-1] << " " << b[S-1] << r << x << endl;
    
}


//test for BitSet::AddTo/SubstractTo
inline void TestUnsignedIntAddTo(unsigned long long int S, unsigned long long int seed){
    
    UnsignedInt a, b;
    vector<unsigned long long int> v_a, v_b, v_a_minus_b;
    unsigned int i, s;
    unsigned long long int r, MAX = 1024;
    gsl_rng* ran;
    Bits borrow;
    bool it_works;
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    a.Resize(10);
    b.Resize(10);
    
    
    for(it_works = true, s=0; s<S; ++s){
        
        
        for(i=0; i<n_bits; i++){
            
            r = gsl_rng_uniform_int(ran, MAX);
            a.Set(i, r);
            if(r!=0){
                b.Set(i, r-1-gsl_rng_uniform_int(ran, r));
            }else{
                b.Set(i, 0);
            }
            
        }
        
        
        //        a.PrintBase10("a");
        //        b.PrintBase10("b");
        
        a.GetBase10(v_a);
        b.GetBase10(v_b);
        
        a.SubstractTo(&b, &borrow);
        
        //            a.PrintBase10("a-b");
        
        a.GetBase10(v_a_minus_b);
        
        for(i=0; i<n_bits; ++i){
            if( v_a[i]-v_b[i] !=  v_a_minus_b[i] ){
                it_works = false;
                break;
            }
            
            cout << "[" << i << "]:\t\t\t" << v_a[i]-v_b[i] << "\t\t\t" << v_a_minus_b[i] << endl;
        }
        //
        
    }
    
    cout << "It works = " << it_works << "." <<  endl;
    
    
}




//test for    UnsignedInt *=
inline void TestUnsignedIntMultiply(unsigned long long int S, unsigned long long int seed){
    
    //the maximum unsigned int that I will draw
    unsigned long long int max = 30;
    UnsignedInt A(max), B(max), C;
    bool it_works;
    vector<unsigned long long int> v_a, v_b, v_a_times_b;
    unsigned int i, s;
    gsl_rng* ran;
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    
    for(it_works = true, s=0; (s<S) & it_works; ++s){
        
        A.Resize(0);
        B.Resize(0);
        
        A.Resize(bits(max));
        B.Resize(bits(max));
        C.Resize(A.GetSize()+B.GetSize());

        A.Clear();
        B.Clear();
        
        
        for(i=0; i<n_bits; i++){
            A.Set(i, gsl_rng_uniform_int(ran, max));
            B.Set(i, gsl_rng_uniform_int(ran, max));
        }
        
        
        //    cout << "----------- Before += -----------" << endl;
        //    cout << "a : " << endl;
        //    a.Print();
        //    a.PrintBase10();
        
        //    cout << "b : " << endl;
        //    b.Print();
        //    b.PrintBase10();
        
        A.GetBase10(v_a);
        B.GetBase10(v_b);
        
//        A *= &B;
        A.Multiply(&B, &C);
        
        //    cout << "----------- After += -----------" << endl;
        //    cout << "a+b: " << endl;
        //    a.Print();
        //    a.PrintBase10();
        
        C.GetBase10(v_a_times_b);
        
        cout << "Check of the result:" << endl;
        for( i=0; i<n_bits; ++i){

            cout << "[" << n_bits-1-i << "]:\t\t\t" << v_a[n_bits-1-i]*v_b[n_bits-1-i] << "\t\t\t" << v_a_times_b[n_bits-1-i] << endl;

            if(v_a[n_bits-1-i]*v_b[n_bits-1-i] != v_a_times_b[n_bits-1-i]){
                it_works = false;
                break;
            }
            
        }
        //
        
    }
    
    cout << "It works  = " << it_works << "." << endl;
    
    
}



inline void SpeedTestUnsignedIntMultiply(unsigned long long int maximum_value, unsigned long long int S, unsigned long long int seed){
    
    
    cout << " ***************************** Speed test for UnsignedInt::Multiply *****************************" << endl;

    
    clock_t start=0, end=0;
    gsl_rng* ran;
    unsigned long long int r=0, s;
    unsigned int i, b, c=0;
    double x = 0.0;
    
    vector<UnsignedInt> A(S);
    UnsignedInt B, C;
    vector<unsigned int> a(n_bits*S);
    
    for(s=0; s<A.size(); ++s){
        A[s].Resize(bits(maximum_value));
    }
    B.Resize(bits(maximum_value));
    C.Resize(A[0].GetSize()+B.GetSize());
    
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    
    
    b = (unsigned int)gsl_rng_uniform_int(ran, maximum_value);
    //    cout << "b: " << b << endl;

    start = clock();
    for(s=0; s<n_bits*S; s++){
        
        //this simulates the drawing of the random number for the Gillespie algorithm
        r = gsl_rng_uniform_int(ran, maximum_value);
        x = gsl_rng_uniform(ran);
        c = (a[s]) * b;
        
    }
    end = clock();
    cout << "Time for n_bits * S [random number + operation]s without bits = "  << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << " s" << endl;
    
    
    //****************** calculation with bits ******************
    for(s=0; s<S; s++){
        
        for(i=0; i<n_bits; i++){
            A[s].Set(i, gsl_rng_uniform_int(ran, maximum_value));
        }
        
    }
    for(i=0; i<n_bits; i++){
        B.Set(i, gsl_rng_uniform_int(ran, maximum_value));
    }

    
    start = clock();
    for(s=0; s<S; s++){
        
        //        A[s].PrintBase10("A");
        //        B.PrintBase10("B");
        
        //        (A[s]) += (&B[s]);
        //        (A[s]).AddTo(&(B[s]), &carry);
        
        //                (A[s]) -= (&B[s]);
        //this simulates the drawing of the random number for the Gillespie algorithm
        r = (unsigned int)gsl_rng_uniform_int(ran, maximum_value);
        x = gsl_rng_uniform(ran);
        (A[s]).Multiply(&B, &C);
        
        //        A[s].PrintBase10("A");
        
    }
    end = clock();
    
    cout << "Time for S [random number + operation]s with bits = "   << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << "s" <<  endl << endl;
    
    
    
    //without this the for loop will not be exectued with -O3
    cout << endl;
    C.PrintBase10(cout);
    cout << "dummy print a = " << a[S-1] << " " << c << " " << r << x << endl;
    
    
    
}


//test for   Fraction::FloorMultiply
inline void TestFractionFloorMultiply(unsigned long long int S, unsigned long long int seed){
    
    //the maximum unsigned int that I will draw
    unsigned long long int max = 1024;
    UnsignedInt B(max), C, W;
    Fraction A;
    bool it_works;
    vector<unsigned long long int> v_A, v_floor_A_times_B;
    vector<double> v_B;
    unsigned int i, s;
    gsl_rng* ran;
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    

    for(it_works = true, s=0; (s<S) & it_works; ++s){
        
        A.Resize(bits(n_bits_mantissa));
        B.Resize(bits(max));
        C.Resize(B.GetSize());
        W.Resize(A.GetSize() + B.GetSize());

        A.Clear();
        B.Clear();
        
        for(i=0; i<n_bits; i++){
            B.Set(i, gsl_rng_uniform_int(ran, max));
        }
        A.SetRandom(ran);
        
            
        A.GetBase10(&v_B);
        B.GetBase10(v_A);
        
        A.FloorMultiply(&B, &C, &W);
        
        C.GetBase10(v_floor_A_times_B);
        
        cout << "Check of the result:" << endl;
        for( i=0; i<n_bits; ++i){

            cout << "[" << n_bits-1-i << "]:\t\t\t" << floor(v_A[n_bits-1-i]*v_B[n_bits-1-i]) << "\t\t\t" << v_floor_A_times_B[n_bits-1-i] << endl;

            if(floor(v_A[n_bits-1-i]*v_B[n_bits-1-i]) != v_floor_A_times_B[n_bits-1-i]){
                it_works = false;
                break;
            }
            
        }
        //
        
    }
    
    cout << "It works  = " << it_works << "." << endl;
    
    
}



inline void SpeedTestFractionFloorMultiply(unsigned int n_bits_factor, unsigned long long int maximum_value_multiplicand, unsigned long long int S, unsigned long long int seed){
    
    
    cout << " ***************************** Speed test for Fraction::FloorMultiply *****************************" << endl;

    
  
    //the maximum unsigned int that I will draw
    unsigned long long int r=0;
    vector<double> a(n_bits*S);
    vector<Fraction> A(S);
    UnsignedInt B(maximum_value_multiplicand), C(maximum_value_multiplicand), W;
    unsigned int b=0, c=0, i, s;
    double x = 0.0;
    gsl_rng* ran;
    clock_t start=0, end=0;

    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    
    //****************** calculation without bits ******************
    b = (unsigned int)gsl_rng_uniform_int(ran, maximum_value_multiplicand);
    for(s=0; s<n_bits*S; s++){
        a[s] = gsl_rng_uniform(ran);
    }
    
    start = clock();
    for(s=0; s<n_bits*S; s++){
        
        //this simulates the drawing of the random number for the Gillespie algorithm
        r = gsl_rng_uniform_int(ran, maximum_value_multiplicand);
        x = gsl_rng_uniform(ran);
        
        c = floor((a[s])*b);
        
    }
    end = clock();
    cout  << "Time for n_bits*S [random number + operation]s without bits = "  << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << " s" << endl;
    

    
    
    //****************** calculation with bits ******************
    for(s=0; s<S; s++){
        A[s].Resize(n_bits_factor);

        A[s].SetRandom(ran);
    }
    for(i=0; i<n_bits; i++){
        B.Set(i, gsl_rng_uniform_int(ran, maximum_value_multiplicand));
    }
    W.Resize(A.front().GetSize() + B.GetSize());
    
    start = clock();
    for(s=0; s<S; ++s){
                
         //this simulates the drawing of the random number for the Gillespie algorithm
         r = gsl_rng_uniform_int(ran, maximum_value_multiplicand);
         x = gsl_rng_uniform(ran);

        A[s].FloorMultiply(&B, &C, &W);
        
    }
    end = clock();
    
    cout << "Time for S [random number + operation]s with bits = " << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << "s" <<  endl << endl;
    
    
    //without this the for loop will not be exectued with -O3
    cout << endl;
    A.back().PrintBase10("dummy print");
    C.PrintBase10(cout);
    cout << "dummy print: a = " << a.back() << " " << b << c << r << x << endl;
 

    
}



inline void SpeedTestDoubleAddTo(unsigned long long int S, unsigned long long int seed){
    
    
    cout << " ***************************** Speed test for Double::AddTo *****************************" << endl;

    
    //speed test  for Double::AddTo
    clock_t start=0, end=0;
    gsl_rng* ran;
    unsigned long long int i, r=0, s;
    double x = 0.0;
    int MAX = 128;
    
    vector<Double> A(S), /*I need to declare B as a vector rather than as a single Double because each AddTo(&B) will alter the content of B and thus lead to potential overflows/unerflows as many AddTo(s) are executed*/B(S);
    Double C;
    vector<double> a(n_bits*S);
    double b;
    
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    
    
    //****************** calculation without bits ******************
    b = gsl_pow_int(2.0, (MAX/2 - (int)gsl_rng_uniform_int(ran, MAX)))*gsl_rng_uniform(ran);
    //    cout << "b: " << b << endl;
    for(s=0; s<n_bits*S; ++s){
        a[s] = gsl_pow_int(2.0, (MAX/2 - (int)gsl_rng_uniform_int(ran, MAX)))*gsl_rng_uniform(ran);
        //                cout << "a[]: " << a[s] << endl;
    }

    start = clock();
    for(s=0; s<n_bits*S; s++){
        
        //this simulates the drawing of the random number for the Gillespie algorithm
        r = gsl_rng_uniform_int(ran, MAX);
        x = gsl_rng_uniform(ran);
        (a[s]) += b;
        
    }
    end = clock();
    cout << "Time for n_bits*S [random number + operation]s without bits = "  << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << " s" << endl;
    
    
    //****************** calculation with bits ******************
    for(i=0; i<n_bits; i++){
        C.Set((unsigned int)i, false, 1023 + (MAX/2 - gsl_rng_uniform_int(ran, MAX)), gsl_rng_uniform(ran));
    }
//    B.PrintBase10("B");
    for(s=0; s<S; ++s){
        
        for(i=0; i<n_bits; i++){
            A[s].Set((unsigned int)i, false, 1023 + (MAX/2 - gsl_rng_uniform_int(ran, MAX)), gsl_rng_uniform(ran));
        }
        B[s] = C;
        
        //        A[s].PrintBase10("A");
    }
    
    
    start = clock();
    for(s=0; s<S; s++){
        
        //        A[s].PrintBase10("A");
        //        B.PrintBase10("B");

        //this simulates the drawing of the random number for the Gillespie algorithm
        r = gsl_rng_uniform_int(ran, MAX);
        x = gsl_rng_uniform(ran);
        //        (A[s]) += (&B);
        (A[s]).AddTo(&(B[s]));
        
        //        A[s].PrintBase10("A");
        
    }
    end = clock();
    
    cout << "Time for n_bits*S [random number + operation]s with bits = "   << std::scientific << ((double)(end - start))/CLOCKS_PER_SEC << "s" << endl;
    
    
    //without this the for loop will not be exectued with -O3
    cout << endl;
    A.back().PrintBase10("dummy print");
    cout << "dummy print a = " << a[S-1] << " " << b << " " << r << x << endl;
    
    
    
}

//test for Double::operator +=
inline void TestDoubleAddTo(unsigned long long int S, unsigned long long int seed){
    
    
    Double A, B;
    double error;
    vector<double> v_a, v_b, v_a_plus_b;
    unsigned int i, s;
    gsl_rng* ran;
    
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    gsl_rng_set(ran, seed);
    
    
    for(error = 0.0, s=0; s<S; ++s){
        
        for(i=0; i<n_bits; i++){
            A.Set(i, false, 1023 + (128/2 - gsl_rng_uniform_int(ran, 128)), gsl_rng_uniform(ran));
            B.Set(i, false, 1023 + (128/2 - gsl_rng_uniform_int(ran, 128)), gsl_rng_uniform(ran));
        }
        
        
        //    cout << "----------- Before += -----------" << endl;
        //    cout << "a : " << endl;
        //    a.Print();
        //    a.PrintBase10();
        
        //    cout << "b : " << endl;
        //    b.Print();
        //    b.PrintBase10();
        
        A.GetBase10(v_a);
        B.GetBase10(v_b);
        
        //        a+= &b;
        A.AddTo(&B);
        
        //    cout << "----------- After += -----------" << endl;
        //    cout << "a+b: " << endl;
        //    a.Print();
        //    a.PrintBase10();
        
        A.GetBase10(v_a_plus_b);
        
        cout << "Check of the result:" << endl;
        for( i=0; i<n_bits; ++i){
            if(fabs(((v_a[n_bits-1-i]+v_b[n_bits-1-i])-v_a_plus_b[n_bits-1-i])/v_a_plus_b[n_bits-1-i]) > error){error = fabs(((v_a[n_bits-1-i]+v_b[n_bits-1-i])-v_a_plus_b[n_bits-1-i])/v_a_plus_b[n_bits-1-i]);}
            
            cout << "[" << n_bits-1-i << "]:\t\t\t" << v_a[n_bits-1-i]+v_b[n_bits-1-i] << "\t\t\t" << v_a_plus_b[n_bits-1-i] << "\t\t\t" << fabs(((v_a[n_bits-1-i]+v_b[n_bits-1-i])-v_a_plus_b[n_bits-1-i])/v_a_plus_b[n_bits-1-i]) << endl;
        }
        //
        
    }
    
    cout << "Maximum relative error = " << error << endl;

    
}


////this is not a method of the Bits class, but an ordinary function which had to be declared in this file because it need to know who the Bits type is
//inline Bits operator ~ (const Bits& m){
//    
//    return Bits((~(m.n)));
//    
//}
