//
//  system_nobits.cpp
//  gillespie
//
//  Created by Michele on 07/02/2024.
//

#include "system_nobits.hpp"

#include "lib.hpp"
#include "main.hpp"

#include "gsl_math.h"
#include "gsl_randist.h"


inline SystemNoBits::SystemNoBits(void){
    
}


//inline 
SystemNoBits::SystemNoBits(unsigned long long int N_in, unsigned int seed_in, const string& out_directory_in){
    
    unsigned int i, *n;
    const double p[] = {1.0, 1.0, 2.0};
    
    
    
    n = new unsigned int [3];
    ran = gsl_rng_alloc(gsl_rng_gfsr4);
    
    gsl_rng_set(ran, seed);
    
    N = N_in;
    seed = seed_in;
    out_directory = out_directory_in;
    
    outfile_name.str("");
    outfile_name << out_directory << "/out_nobits.dat";
    outfile.open(outfile_name.str().c_str());
    
    N_N_m_1_2 = N*(N-1)/2;
    t=0;
    
    x.resize(3);
    x_t.resize(3);
    //    c.resize(3);
    a.resize(3);
    
    
    
    //draw the initial number of particles of the three species according to a multinomial distribution
    gsl_ran_multinomial(ran, 3, (unsigned int)N, p, n);
    
    for(i=0; i<3; ++i){
        
        //write the initial number of particles in x
        x[i] = n[i];
        //draw the values of c and write them in c
        //        c[i] = (unsigned int)gsl_rng_uniform_int(ran, M);
        
    }
    
    a[0] = x[0]*x[2];
    a[1] = x[0]*x[1];
    a[2] = x[1]*x[2];
    a[3] = x[0]*(x[0]-1)/2;
    a[4] = x[2]*(x[2]-1)/2;
    a[5] = x[1]*(x[1]-1)/2;
    
    delete [] n;
    
}


//inline 
void SystemNoBits::iterate(void){
    
    unsigned int i;
    
    //draw the random numbers
    r1 = gsl_rng_uniform(ran);
    tau = -1.0/((double)N_N_m_1_2)*log(r1);
    R = gsl_rng_uniform_int(ran, N_N_m_1_2);
    
    for (L=0, compare_old=true, i=0; i<3; i++) {
        
        L += a[i];
        
        compare_new = (L <= R);
        changer = (compare_old ^ compare_new);
        
        x_t[0] = (changer ? x[0] : 0);
        x_t[1] = (changer ? x[1] : 0);
        x_t[2] = (changer ? x[2] : 0);
        
        switch (i) {
                
            case 0:{
                //the second reaction A+D -> 2 D  has been selected

                
                //update the as
                //update a[0]
                a[0] -= x_t[2];
                a[0] += x_t[0];
                a[0] -= changer;
                
                //update a[1]
                a[1] -= x_t[1];
                
                //update a[2]
                a[2] += x_t[1];
                
                //update a[3]
                a[3] -= x_t[0];
                a[3] += changer;
                
                //update a[4]
                a[4] += x_t[2];
                
                //update x
                x[0] -= changer;
                x[2] += changer;
                
                break;
                
            }
                
            case 1:{
                //the second reaction A+L -> 2 L  has been selected
                
                //update the as
                a[0] -= x_t[2];
                
                a[1] -= x_t[1];
                a[1] += x_t[0];
                a[1] -= changer;
         
                a[2] += x_t[2];
                
                a[3] -= x_t[0];
                a[3] += changer;
                
                a[5] += x_t[1];
                
                
                //update x
                x[0] -= changer;
                x[1] += changer;
  
                break;
                
            }
                
            case 2:{
                //the third reaction D+L -> 2 A  has been selected
                
                //update the as
                /*
                 a[0] = x[0]*x[2] -> (x[0]+2*changer)*(x[2]-changer) = a[0] + 2*changer*x[2] - changer*x[0] - 2*changer = a[0] + 2*x_t[2] - x_t[0] - 2*changer
                 a[1] = x[0]*x[1] -> (x[0]+2*changer)*(x[1]-changer) = a[1] + 2*x_t[1] - x_t[0] - 2*changer
                 a[2] = x[1]*x[2] -> (x[1]-changer)*(x[2]-changer) = a[2] - x_t[2] - x_t[1] + changer
                 a[3] = x[0]*(x[0]-1)/2 -> (x[0]+2*changer)*(x[0]-1+2*changer)/2 = a[3] + changer*(x[0]-1) + changer*x[0] + 2*changer = a[3] + 2*x_t[0] + changer
                 a[4] = x[2]*(x[2]-1)/2 -> (x[2]-changer)*(x[2]-1-changer)/2 = a[4] - changer*(x[2]-1)/2 - changer*x[2]/2 + changer/2 = a[4] - x_t[2] + changer
                 a[5] = x[1]*(x[1]-1)/2 -> (x[1]-changer)*(x[1]-1-changer)/2 = a[5] - changer*(x[1]-1)/2 - changer*x[1]/2 + changer/2 = a[5] - x_t[1] + changer

                 */
                
                //update a
                a[0] += 2*x_t[2];
                a[0] -= x_t[0];
                a[0] -= 2*changer;
                
                a[1] += 2*x_t[1];
                a[1] -= x_t[0];
                a[1] -= 2*changer;
                
                a[2] -= x_t[2];
                a[2] -= x_t[1];
                a[2] += changer;
                
                a[3] += 2*x_t[0] + changer;
                
                a[4] -= x_t[2];
                a[4] += changer;
                
                a[5] -= x_t[1];
                a[5] += changer;

                
                //update x
                x[2] -= changer;
                x[1] -= changer;
                //here A should be updated by 2 changer!
                x[0] += 2*changer;
                //here A should be updated by 2 changer!


                break;
                
            }

        }
                
        compare_old = compare_new;
        
    }
    
    outfile << x[0] << "\t" << x[1] << "\t" << x[2] << "\t" << tau << endl;
    
    ++t;
}
