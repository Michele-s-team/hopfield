#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <strstream>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <list>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
//this has to be uncommented to run on mesopsl
#include <stdint.h>

#include "gsl_rng.h"
#include "gsl_math.h"
#include "gsl_sf_log.h"
#include "gsl_randist.h"

/*
 compile on mac
 compile without optimization
*/
// clear; clear;  g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O0 -Wno-deprecated -I/Users/michelecastellana/Documents/office_stuff/work/stages/stage_bastien_dumont_2026/hopfield/codes/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE
 

/*
 compile with optimization
 */
// g++ main.cpp src/*.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated  -I/Users/michelecastellana/Documents/gillespie/include -I/usr/local/include/gsl/ -o main.o -Wall -DHAVE_INLINE

//compile on calcsub
//clear; clear;  g++ main.cpp src/*.cpp  -llapack -lgsl -lgslcblas -lm -O3 -Wno-deprecated -I /usr/include/gsl/ -I./include/ -o main.o -Wall -DHAVE_INLINE
//compile on abacus
//g++ main.cpp src/*.cpp -I ./include/ -I /mnt/beegfs/home/mcastel1/gsl/include/gsl  -I/mnt/beegfs/home/mcastel1/gsl/include/ -L/mnt/beegfs/home/mcastel1/gsl/lib/ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o main.o -DHAVE_INLINE

//run with
//./main.o -N 128 -T 1 -S 5 -s 0 -o /Users/michelecastellana/Desktop
/*
 N is the total perticle number in the Frank model
 S is log_10(total number of iterations)
 s is the seed of the random-number generator
 o is the path where to store the results
 */

#include "main.hpp"
#include "system_bits.hpp"
#include "system_nobits.hpp"

//all entries of BitSet_one are equal to 1


/*
 void UnsignedInt::SetRandom(unsigned long long int seed){
 
 int i;
 gsl_rng* ran;
 
 ran = gsl_rng_alloc(gsl_rng_gfsr4);
 gsl_rng_set(ran, seed);
 
 
 for(i=0; i<b.size(); i++){
 
 
 }
 
 }
 */

//fill all the entries of BitSet_one and of Bits_one with 1, same for 0 with Bits_zero
BitSet BitSet_one;
Bits Bits_one, Bits_zero;


int main(int argc, char * argv[]) {
    
//    int options;
//    unsigned long long int N=0, T=0, S=0, s, t;
//    unsigned int i, seed=0;
//    //the mean of the gain of the btiwise approach vs. the non-bitwise approach, and similarly for the mean of the square of the gain
//    double gain_av, gain_pow_2_av;
//    string output_directory;
//    clock_t start_nobits, start_bits, end_nobits, end_bits;
    
//    cout.precision(cout_precision);
//    BitSet_one.ResizeAndSetAll(1);
//    for(i = 0; i<n_bits; i++){
//        Bits_one.Set(i, true);
//        Bits_zero.Set(i, false);
//    }
    
    
//    while ((options = getopt(argc, argv, ":N:T:S:s:o:")) != -1) {
//        
//        switch (options) {
//                
//            case 'N':
//                N = ((unsigned int)atoi(optarg));
//                break;  
//                
//            case 'T':
//                T = ((unsigned int)gsl_pow_int(10, atoi(optarg)));
//                break;
//                
//            case 'S':
//                S = ((unsigned int)gsl_pow_int(10, atoi(optarg)));
//                break;
//                
//            case 's':
//                seed = ((unsigned int)atoi(optarg));
//                break;
//                
//            case 'o':
//                output_directory.assign(optarg);
//                break;
//                
//                
//        }
//        
//    }
  
    UnsignedInt I(24);
    I.SetRandom((unsigned int)0);
    I.PrintBase10(cout);
    I.MultiplyByTwoTo();
    I.PrintBase10(cout);
    
    
    
    //    TestUnsignedIntMultiply(S, seed);
    //    SpeedTestUnsignedIntMultiply(128, S, seed);
    //    TestUnsignedIntAddTo(S, seed);
    //    TestDoubleAddTo(S, seed);
    //    SpeedTestDoubleAddTo(S, seed);
    //    SpeedTestUnsignedIntAddto(128, S, seed);
    //        TestFractionFloorMultiply(S, seed);
    //        SpeedTestFractionFloorMultiply(32, 128, S, seed);
    
    
    
    /*
    for(gain_av=0.0, gain_pow_2_av=0.0, t=0; t<T; t++){
        
        cout << "* t = " << t;
         
        SystemBits frank_bits(N, (unsigned int)(seed+t), output_directory);
        SystemNoBits frank_nobits(N, (unsigned int)(seed+t), output_directory);
        
        
        // speed test without bits  *********************
        start_nobits = clock();
        for(s=0; s<n_bits*S; ++s){
            frank_nobits.iterate();
        }
        end_nobits = clock();
        frank_nobits.outfile.close();
        cout << endl << "\tTime for [n_bits*S] Iterate()s without bits = \t\t " << std::scientific << ((double)(end_nobits - start_nobits))/CLOCKS_PER_SEC << "s" <<  endl;
        
        
        
        // speed test with bits  *********************
        start_bits = clock();
        for(s=0; s<S; ++s){
            frank_bits.Iterate();
        }
        end_bits = clock();
        frank_bits.outfile.close();
        cout << "\tTime for S Iterate()s with bits = \t\t         " << std::scientific << ((double)(end_bits - start_bits))/CLOCKS_PER_SEC << "s" <<  endl << endl;
        
        
        gain_av += ((double)(end_nobits - start_nobits)) / ((double)(end_bits - start_bits));
        gain_pow_2_av += gsl_pow_2(((double)(end_nobits - start_nobits)) / ((double)(end_bits - start_bits)));
        
        cout << "\tGain bits/nobits = " << ((double)(end_nobits - start_nobits)) / ((double)(end_bits - start_bits)) << endl;
        
        cout << endl << "\tdummy print";
        frank_bits.L.PrintBase10(cout);
        cout << "\t" << frank_bits.tau << frank_nobits.tau << frank_nobits.z << frank_nobits.rhs << endl;
        //
        
    }
    
    gain_av /= ((double)T);
    gain_pow_2_av /= ((double)T);
    cout << "\nlog_2(N),log_10(T),log_10(S),<Gain bits/nobits>,err_<Gain bits/nobits>:\n"
    << log(N)/log(2) << ","
    << log(T)/log(10.0) << ","
    << log(S)/log(10.0)  << ","
    << gain_av << ","
    << sqrt(gain_pow_2_av - gsl_pow_2(gain_av))
    << endl;
    
    cout << endl;
    return 0;
    */
     
    /*
     double x = gsl_rng_uniform(Frank.ran);
     Fraction R;
     vector<bool> v(n_bits_mantissa);
     R.Resize(n_bits_mantissa);
     R.Fraction::SetAllFromDoubleMantissa(x, &v);
     
     
     cout << "x:" << x << endl;
     R.Fraction::PrintBase10("R");
     */
    
}
