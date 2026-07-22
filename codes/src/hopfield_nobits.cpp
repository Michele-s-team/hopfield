//
//  hopfield_nobits.cpp
//  hopfield
//
//  Created by Bastien on 12/05/2026.
//

#include "hopfield_nobits.hpp"

#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"
#include <numeric>
#include <algorithm>

// =====================================================
// ENERGY COMPUTATION
// =====================================================

// Local energy cost of flipping spin i in realization r:
//   ΔE(i, r) = 2 * σ_i(r) * Σ_j J_ij(r) σ_j(r)
// For Hopfield: J_ij = (1/N) * Σ_p ξ_i^p ξ_j^p
int HopfieldNoBits::DeltaE(int spin, int r) {
    int sum = 0.0;
    for (int k = 0; k < neighbors[spin].size(); ++k) {
        int j = neighbors[spin][k];
        sum += couplings[spin][k*n_bits+r] * spins_set[r * N + j];
    }
    return spins_set[r * N + spin] * sum;
}

int HopfieldNoBits::DeltaE_pure_overlaps(int spin, int r) {
    int sum = 0.0;
    for (int mu = 0; mu < P; ++mu) {
        sum += patterns[spin * P * n_bits + mu * n_bits + r]*overlaps[mu * n_bits + r];
    }
    return spins_set[r * N + spin] * sum-P;
}

int HopfieldNoBits::DeltaE_overlaps_non_neighbors(int spin, int r) {
    int sum = 0.0;
    for (int mu = 0; mu < P; ++mu) {
        sum += patterns[spin * P * n_bits + mu * n_bits + r]*overlaps[mu * n_bits + r];
    }
    int sum_couplings;
    for (int i=0; i< non_neighbors[spin].size(); i++){
        int j = non_neighbors[spin][i];
        sum_couplings += couplings_nonneighbors[spin][i*n_bits+r] * spins_set[r * N + j];

    }
    return spins_set[r * N + spin] * (sum-sum_couplings)-P;
}

// =====================================================
// ENERGY COMPUTATION (all replicas at once)
// =====================================================
// Calcule DeltaE(spin, r) pour tous les réplicas r en une seule passe sur les
// voisins, au lieu de reboucler sur neighbors[spin] à chaque r séparément.
// Boucle k externe, r interne : accès contigu à couplings[spin][k][r].
void HopfieldNoBits::DeltaE_neighbors_all(int spin, vector<int>& delta_E) {
    delta_E.assign(n_bits, 0);
    for (int k = 0; k < neighbors[spin].size(); ++k) {
        int j = neighbors[spin][k];
        for (int r = 0; r < n_bits; ++r) {
            delta_E[r] += couplings[spin][k*n_bits+r] * spins_set[r * N + j];
        }
    }
    for (int r = 0; r < n_bits; ++r) {
        delta_E[r] *= spins_set[r * N + spin];
    }
}

void HopfieldNoBits::DeltaE_pure_overlaps_all(int spin, vector<int>& delta_E) {
    delta_E.assign(n_bits, 0);
    for (int mu = 0; mu < P; ++mu) {
        for (int r = 0; r < n_bits; ++r) {
            delta_E[r] +=  patterns[spin * P * n_bits + mu * n_bits + r]*overlaps[mu * n_bits + r];
        }
    }
    for (int r = 0; r < n_bits; ++r) {
        delta_E[r] *= spins_set[r * N + spin];
        delta_E[r]-=P;
    }
}

void HopfieldNoBits::DeltaE_overlaps_non_neighbors_all(int spin, vector<int>& delta_E) {
    delta_E.assign(n_bits, 0);
    for (int r = 0; r < n_bits; ++r) {

        for (int mu = 0; mu < P; ++mu) {
            delta_E[r] +=  patterns[spin * P * n_bits + mu * n_bits + r]*overlaps[mu * n_bits + r];
        }
        for (int i=0; i< non_neighbors[spin].size(); i++){
            int j = non_neighbors[spin][i];
            delta_E[r] -= couplings_nonneighbors[spin][i*n_bits+r] * spins_set[r * N + j];
        }

        delta_E[r] *= spins_set[r * N + spin];
        delta_E[r]-=P;
    }
}

// =====================================================
// SHARED RNG (same random threshold among replicas)
// =====================================================

void HopfieldNoBits::runSweepsSharedRNG(gsl_rng* ran, bool save, double freq){
     double alpha = (double)P / N;

    double mean_degree  = std::accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
    double mean_density = mean_degree / (N-1);

    if (alpha + 0.5 <= mean_density) {
        // Dense graph: overlap formulation over non-neighbors is cheaper
        compute_overlaps();
        runSweepsSharedRNG_overlaps_non_neighbors(ran, save, freq);
    } else {
        // Sparse graph: direct neighbor-coupling formulation is cheaper
        initCouplings();
        runSweepsSharedRNG_neighbors(ran, save, freq);
    }
}

void HopfieldNoBits::runSweepsIndependentRNG(gsl_rng* ran, bool save, double freq){
     double alpha = (double)P / N;

    double mean_degree  = std::accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
    double mean_density = mean_degree / (N-1);

    if (alpha + 0.5 <= mean_density) {
        // Dense graph: overlap formulation over non-neighbors is cheaper
        compute_overlaps();
        runSweepsIndependentRNG_overlaps_non_neighbors(ran, save, freq);
    } else {
        // Sparse graph: direct neighbor-coupling formulation is cheaper
        initCouplings();
        runSweepsIndependentRNG_neighbors(ran, save, freq);
    }
}

void HopfieldNoBits::runSweepsSharedRNG_neighbors(gsl_rng* ran, bool save, double freq) {

    cout << "Shared RNG: Neighbors couplings spin update algorithm" << endl;

    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            const int spin = gsl_rng_uniform_int(ran, N);
            int deg_spin  = degrees[spin];
            int rng = randomNumber(ran, P * deg_spin, N);
            // Branch 1: unconditional flip
            if (rng >= P * deg_spin) {
                for (int r = 0; r < n_bits; ++r)
                    spins_set[r * N + spin] *= -1;
            } else {

                // Branch 2: compute all ΔE once
                DeltaE_neighbors_all(spin, delta_E);
                for (int r = 0; r < n_bits; ++r) {
                    const int dE = delta_E[r];
                    if (dE <= 0 || rng >= dE) {
                        spins_set[r * N + spin] *= -1;
                    }
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << endl;
}

void HopfieldNoBits::runSweepsSharedRNG_overlaps(gsl_rng* ran, bool save, double freq) {

    cout << "Shared RNG: Overlaps spin update algorithm" << endl;

    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            const int spin = gsl_rng_uniform_int(ran, N);
            int deg_spin  = degrees[spin];
            int rng = randomNumber(ran, P * deg_spin, N);
            // Branch 1: unconditional flip
            if (rng >= P * deg_spin) {
                for (int r = 0; r < n_bits; ++r){
                    for (int mu=0; mu<P; mu++){
                        overlaps[mu * n_bits + r]-=
                         2*patterns[spin * P * n_bits + mu * n_bits + r]*spins_set[r * N + spin];
                    }
                    spins_set[r * N + spin] *= -1;
                }
            } else {

                // Branch 2: compute all ΔE once
                DeltaE_pure_overlaps_all(spin, delta_E);
                for (int r = 0; r < n_bits; ++r) {
                    const int dE = delta_E[r];
                    if (dE <= 0 || rng >= dE) {
                        for (int mu=0; mu<P; mu++){
                         overlaps[mu * n_bits + r]-=
                         2*patterns[spin * P * n_bits + mu * n_bits + r]*spins_set[r * N + spin];
                    }
                        spins_set[r * N + spin] *= -1;
                    }
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << endl;
}

void HopfieldNoBits::runSweepsSharedRNG_overlaps_non_neighbors(gsl_rng* ran, bool save, double freq) {


    int min_deg = *min_element(degrees.begin(), degrees.end());
    if (min_deg == N - 1) { runSweepsSharedRNG_overlaps(ran, save, freq); return; }

    cout << "Shared RNG: Overlaps and non-neighbors couplings spin update algorithm" << endl;

    initCouplings_nonNeighbors();


    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            const int spin = gsl_rng_uniform_int(ran, N);
            int deg_spin  = degrees[spin];
            int rng = randomNumber(ran, P * deg_spin, N);
            // Branch 1: unconditional flip
            if (rng >= P * deg_spin) {
                for (int r = 0; r < n_bits; ++r){
                    for (int mu=0; mu<P; mu++){
                        overlaps[mu * n_bits + r]-=
                         2*patterns[spin * P * n_bits + mu * n_bits + r]*spins_set[r * N + spin];
                    }
                    spins_set[r * N + spin] *= -1;
                }
            } else {

                // Branch 2: compute all ΔE once
                DeltaE_overlaps_non_neighbors_all(spin, delta_E);
                for (int r = 0; r < n_bits; ++r) {
                    const int dE = delta_E[r];
                    if (dE <= 0 || rng >= dE) {
                        for (int mu=0; mu<P; mu++){
                         overlaps[mu * n_bits + r]-=
                         2*patterns[spin * P * n_bits + mu * n_bits + r]*spins_set[r * N + spin];
                    }
                        spins_set[r * N + spin] *= -1;
                    }
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << endl;
}

// =====================================================
// INDEPENDENT RNG (different random threshold among replicas)
// =====================================================
void HopfieldNoBits::runSweepsIndependentRNG_neighbors(gsl_rng* ran, bool save, double freq) {

    cout << "Independent RNG: Neighbors couplings spin update algorithm" << endl;

    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {
            const int spin = gsl_rng_uniform_int(ran, N);
            const int deg_spin = degrees[spin];

            // Compute ΔE once per spin
            DeltaE_neighbors_all(spin, delta_E);
            for (int r = 0; r < n_bits; ++r) {
                const int dE = delta_E[r];
                // tirage indépendant par replica
                const int rng = randomNumber(ran, P * deg_spin, N);

                bool flip;
                if (rng >= P * deg_spin) {spins_set[r * N + spin] *= -1;}
                if (dE <= 0 || rng >= dE){spins_set[r * N + spin] *= -1;}
                }
            }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << endl;
}


void HopfieldNoBits::runSweepsIndependentRNG_overlaps(gsl_rng* ran, bool save, double freq) {

    cout << "Independent RNG: Overlaps spin update algorithm" << endl;

    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {

            const int spin     = gsl_rng_uniform_int(ran, N);
            const int deg_spin = degrees[spin];

            // ΔE calculé une seule fois pour les n_bits replicas
            DeltaE_pure_overlaps_all(spin, delta_E);

            for (int r = 0; r < n_bits; ++r) {
                const int dE = delta_E[r];

                // tirage indépendant par replica
                const int rng = randomNumber(ran, P * deg_spin, N);

                bool flip;
                if (rng >= P * deg_spin) {
                    // flip inconditionnel pour ce replica
                    flip = true;
                } else {
                    flip = (dE <= 0 || rng >= dE);
                }

                if (flip) {
                    for (int mu = 0; mu < P; mu++) {
                        overlaps[mu * n_bits + r] -=
                            2 * patterns[spin * P * n_bits + mu * n_bits + r]
                              * spins_set[r * N + spin];
                    }
                    spins_set[r * N + spin] *= -1;
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << endl;
}


void HopfieldNoBits::runSweepsIndependentRNG_overlaps_non_neighbors(gsl_rng* ran, bool save, double freq) {

    int min_deg = *min_element(degrees.begin(), degrees.end());
    if (min_deg == N - 1) { runSweepsIndependentRNG_overlaps(ran, save, freq); return; }

    cout << "Independent RNG: Overlaps and non-neighbors couplings spin update algorithm" << endl;


    initCouplings_nonNeighbors();

    const int total_sweeps = getNSweeps();
    const int progress_stride = std::max(1, total_sweeps / 10);
    const int save_stride = save ? std::max(1, (int)std::round(1.0 / freq)) : 0;

    std::vector<int> delta_E(n_bits);

    for (int sweep = 0; sweep < total_sweeps; ++sweep) {
        for (int step = 0; step < N; ++step) {

            const int spin     = gsl_rng_uniform_int(ran, N);
            const int deg_spin = degrees[spin];

            // ΔE calculé une seule fois pour les n_bits replicas
            DeltaE_overlaps_non_neighbors_all(spin, delta_E);

            for (int r = 0; r < n_bits; ++r) {
                const int dE = delta_E[r];

                // tirage indépendant par replica
                const int rng = randomNumber(ran, P * deg_spin, N);

                bool flip;
                if (rng >= P * deg_spin) {
                    // flip inconditionnel pour ce replica
                    flip = true;
                } else {
                    flip = (dE <= 0 || rng >= dE);
                }

                if (flip) {
                    for (int mu = 0; mu < P; mu++) {
                        overlaps[mu * n_bits + r] -=
                            2 * patterns[spin * P * n_bits + mu * n_bits + r]
                              * spins_set[r * N + spin];
                    }
                    spins_set[r * N + spin] *= -1;
                }
            }
        }

        if (save && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            std::cout << "\rSweep: " << (sweep + 1)
                      << " (" << (sweep + 1) * 100 / total_sweeps << "%)    "
                      << std::flush;
    }

    std::cout << endl;
}
