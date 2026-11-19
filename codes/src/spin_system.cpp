//
//  spin_system.cpp
//  hopfield
//
//  Created by Bastien on 22/04/2026.
//
#include "spin_system.hpp"

#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"
#include <unordered_set>
#include <math.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <iostream>

// =====================================================
// CONSTRUCTION
// =====================================================

SpinSystem::SpinSystem(int N)
    : N(N),
      neighbors(N),
      degrees(N, 0),
      spins_set(n_bits * N, 0)
{}


void SpinSystem::setSize(int new_N) {
    N = new_N;
    
    neighbors.assign(N, vector<int>());
    degrees.assign(N, 0);
    spins_set.assign(n_bits * N, 0);
}

// =====================================================
// INIT/GETTERS
// =====================================================

const vector<vector<int>>& SpinSystem::getNeighbors() const    { return neighbors; }
const vector<vector<int>>& SpinSystem::getNonNeighbors() const { return non_neighbors; }
const vector<int>&         SpinSystem::getDegrees() const      { return degrees; }

void SpinSystem::initNeighbors(const vector<vector<int>>& nb) {
    if (static_cast<int>(nb.size()) != N) {
        cerr << "[ERROR] initNeighbors: size " << nb.size() << " != N = " << N << "\n";
        return;
    }
    neighbors = nb;
}

void SpinSystem::initNonNeighbors(const vector<vector<int>>& nnb) {
    if (static_cast<int>(nnb.size()) != N) {
        cerr << "[ERROR] initNonNeighbors: size " << nnb.size() << " != N = " << N << "\n";
        return;
    }
    non_neighbors = nnb;
}

void SpinSystem::initDegrees(const vector<int>& deg) {
    if (static_cast<int>(deg.size()) != N) {
        cerr << "[ERROR] initDegrees: size " << deg.size() << " != N = " << N << "\n";
        return;
    }
    degrees = deg;
}

// =====================================================
// NETWORK TOPOLOGY
// =====================================================

// 2D square lattice with periodic boundary conditions
void SpinSystem::initNetwork_2D_PBC() {
    int L = sqrt(N);
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
            int i = x + L * y;
            neighbors[i] = {
                (x+1)%L + L*y,
                (x-1+L)%L + L*y,
                x + L*((y+1)%L),
                x + L*((y-1+L)%L)
            };
            degrees[i] = 4;
        }
    computeNonNeighbors();
}

// 2D square lattice with open boundary conditions (edge spins have fewer neighbors)
void SpinSystem::initNetwork_2D_OBC() {
    int L = sqrt(N);
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
            int i = x + L * y;
            neighbors[i].clear();
            if (x + 1 < L) neighbors[i].push_back((x+1) + L*y);
            if (x - 1 >= 0) neighbors[i].push_back((x-1) + L*y);
            if (y + 1 < L) neighbors[i].push_back(x + L*(y+1));
            if (y - 1 >= 0) neighbors[i].push_back(x + L*(y-1));
            degrees[i] = neighbors[i].size();
        }
    computeNonNeighbors();
}

// Fully connected network: every site is connected to every other site
void SpinSystem::initNetwork_FullyConnected(){
    for (int i = 0; i < N; ++i) {
        neighbors[i].clear();
        neighbors[i].reserve(N - 1);
        for (int j = 0; j < N; ++j) {
            if (j != i)
                neighbors[i].push_back(j);
        }
        degrees[i] = N - 1;
    }
    computeNonNeighbors();
}

// Erdos-Renyi random graph: each directed edge (i,j) included with probability p
void SpinSystem::initNetwork_random(gsl_rng* ran, double p) {
    neighbors.assign(N, vector<int>());
    fill(degrees.begin(), degrees.end(), 0);
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (gsl_rng_uniform(ran) < p) {
                neighbors[i].push_back(j);
                neighbors[j].push_back(i);
                degrees[i]++;
                degrees[j]++;
            }
        }
    }
    computeNonNeighbors();
}

// Random k-regular graph, k = round(rho*(N-1)).
// 1) deterministic circulant k-regular graph (always valid)
// 2) randomization by degree-preserving double edge swaps
void SpinSystem::initNetwork_RegularRandom(double rho, gsl_rng* ran, int swaps_per_edge /* = 10 */){
    int k = static_cast<int>(std::lround(rho * (N - 1)));
    k = std::max(0, std::min(k, N - 1));

    // A k-regular graph on N nodes needs N*k even
    if ((static_cast<long long>(N) * k) % 2 != 0) {
        k = (k > 0) ? k - 1 : k + 1;
        std::cerr << "[WARN] N*k odd, using k = " << k << "\n";
    }

    // adjacency matrix for O(1) edge lookup
    std::vector<char> adj(static_cast<size_t>(N) * N, 0);
    auto A = [&](int i, int j) -> char& { return adj[static_cast<size_t>(i) * N + j]; };

    std::vector<std::pair<int,int>> edges;
    edges.reserve(static_cast<size_t>(N) * k / 2);

    auto add_edge = [&](int i, int j) {
        if (i == j || A(i, j)) return;
        A(i, j) = A(j, i) = 1;
        edges.emplace_back(i, j);
    };

    // 1) circulant: i <-> i±1..±(k/2), plus the antipode i+N/2 if k is odd
    for (int i = 0; i < N; ++i) {
        for (int d = 1; d <= k / 2; ++d)
            add_edge(i, (i + d) % N);
        if (k % 2 == 1)
            add_edge(i, (i + N / 2) % N);
    }

    // 2) randomization: (a,b),(c,d) -> (a,d),(c,b)
    const size_t E = edges.size();
    if (E >= 2 && k < N - 1) {   // complete graph: nothing to randomize
        const long long n_swaps = static_cast<long long>(swaps_per_edge) * E;
        for (long long s = 0; s < n_swaps; ++s) {
            size_t e1 = gsl_rng_uniform_int(ran, E);
            size_t e2 = gsl_rng_uniform_int(ran, E);
            if (e1 == e2) continue;

            int a = edges[e1].first, b = edges[e1].second;
            int c = edges[e2].first, d = edges[e2].second;

            // random orientation of the second edge, so both swaps are reachable
            if (gsl_rng_uniform_int(ran, 2)) std::swap(c, d);

            if (a == c || a == d || b == c || b == d) continue;  // need 4 distinct nodes
            if (A(a, d) || A(c, b)) continue;                    // would create a multi-edge

            A(a, b) = A(b, a) = 0;
            A(c, d) = A(d, c) = 0;
            A(a, d) = A(d, a) = 1;
            A(c, b) = A(b, c) = 1;
            edges[e1] = {a, d};
            edges[e2] = {c, b};
        }
    }

    // fill the SpinSystem structures
    for (int i = 0; i < N; ++i) {
        neighbors[i].clear();
        neighbors[i].reserve(k);
        for (int j = 0; j < N; ++j)
            if (A(i, j)) neighbors[i].push_back(j);
        degrees[i] = static_cast<int>(neighbors[i].size());   // = k for all i
    }
    computeNonNeighbors();
}


void SpinSystem::computeNonNeighbors(){
    non_neighbors.assign(N, vector<int>());

    for (int i = 0; i < N; ++i) {
        // on met les voisins de i dans un set pour un lookup en O(log k) (ou unordered_set pour O(1))
        unordered_set<int> neigh_set(neighbors[i].begin(), neighbors[i].end());

        non_neighbors[i].reserve(N - 1 - neighbors[i].size());
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;              // on exclut le spin lui-même
            if (neigh_set.find(j) == neigh_set.end()) {
                non_neighbors[i].push_back(j);
            }
        }
    }
}

// =====================================================
// SPIN INITIALIZATION
// =====================================================

// Draw a random spin value in {-1, +1}
int SpinSystem::randomSpin(gsl_rng* ran){
    return 2 * gsl_rng_uniform_int(ran, 2) - 1;
}

// Draw a random binary value in {0, 1}
int SpinSystem::randomBit(gsl_rng* ran){
    return gsl_rng_uniform_int(ran, 2);
}

// Initialize all spins randomly across all realizations
void SpinSystem::initSpins(gsl_rng* ran) {
    for (int i = 0; i < n_bits * N; i++)
        spins_set[i] = randomSpin(ran);
}

// Set spins from an externally provided configuration
void SpinSystem::initSpinsFromConfig(vector<int>& initial_set) {
    spins_set = initial_set;
}

// =====================================================
// OBSERVABLES
// =====================================================

vector<int> SpinSystem::getSpinsConfig() {
    return spins_set;
}

// Compute magnetization m = (1/N) Σ σ_i for each realization //might need to be moved to Ising nobits
void SpinSystem::GetMagnetizations(vector<double>& magnetizations) {
    for (int r = 0; r < n_bits; r++) {
        double sum = 0;
        for (int i = 0; i < N; i++) sum += spins_set[r*N+i];
        magnetizations[r] = sum / N;
    }
}

// Average magnetization over all realizations
double SpinSystem::GetAverageMagnetization() {
    vector<double> magnetizations(n_bits);
    GetMagnetizations(magnetizations);
    double sum = 0;
    for (double m : magnetizations) sum += m;
    return sum / n_bits;
}