//
//  metropolis.hpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//
// THIS CLASS STORES THE METROPOLIS PARAMETERS
// AND RANDOM NUMBER GENERATION UTILITIES.
//

#ifndef metropolis_hpp
#define metropolis_hpp

#include "gsl_rng.h"

class Metropolis {

protected:

    double beta;       // inverse temperature
    double inv2beta;   // 1/(2*beta)
    int N_sweeps;      // number of sweeps

public:

    Metropolis(double beta,
               int N_sweeps);

    virtual ~Metropolis() = default;

    void setBeta(double);
    void setNSweeps(int);

    double getBeta() const {
        return beta;
    }

    int getNSweeps() const {
        return N_sweeps;
    }

    // Metropolis random threshold generator
    // factor: multiplicative factor (default = 1.0 for backward compatibility)
    int randomNumber(gsl_rng* ran, int max_degrees, double factor = 1.0);
};

#endif