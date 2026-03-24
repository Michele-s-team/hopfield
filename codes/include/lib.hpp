//
//  lib.h
//  gillespie
//
//  Created by Michele Castellana on 07/10/2024.
//

#ifndef lib_hpp
#define lib_hpp

#include <iostream>
#include <vector> 

using namespace std;

//this file includes miscellaneous functions that do not belong a classes

unsigned long long int two_pow(unsigned long long int);
unsigned int bits(unsigned long long int);
void bitwise_print_double(const double& x);
void GetMantissaFromDouble(vector<bool>* result, double x);

#endif /* lib_h */
