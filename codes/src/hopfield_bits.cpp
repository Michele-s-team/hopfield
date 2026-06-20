//
//  hopfield_bits.cpp
//  hopfield
//
//  Created by Bastien on 1/06/2026.
//

#include "hopfield_bits.hpp"
#include "unsigned_int.hpp"
#include <numeric>
#include <algorithm>
#include <filesystem>
#include "lib.hpp"
#include "main.hpp"
#include "gsl_math.h"
#include "gsl_randist.h"

using namespace std;


// =====================================================
// STATE CONVERSIONS
// =====================================================

void HopfieldBits::fromCanonical() {
    // =====================================================
    // 1. Convert spins
    // =====================================================
    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    Bits_Spins_Set.clear();
    Bits_Spins_Set.reserve(N);
    Neighbor_Count.clear();
    Neighbor_Count.reserve(N);
    P_times_Neighbor_Count.clear();
    P_times_Neighbor_Count.reserve(N);

    for (int i = 0; i < N; ++i) {
        Bits spin_tmp;
        for (int r = 0; r < n_bits; ++r) {
            int spin = (spins_set[r * N + i] + 1) / 2;
            spin_tmp.Set(r, spin);
        }
        Bits_Spins_Set.push_back(spin_tmp);

        UnsignedInt nc_tmp((unsigned long long)max_deg);
        nc_tmp.SetAll((unsigned long long)neighbor_count[i]);
        Neighbor_Count.push_back(nc_tmp);

        UnsignedInt pnk_tmp((unsigned long long)(P * max_deg));
        pnk_tmp.SetAll((unsigned long long)(P * neighbor_count[i]));
        P_times_Neighbor_Count.push_back(pnk_tmp);
    }

    // =====================================================
    // 2. Convert patterns
    // =====================================================
    Patterns.clear();
    Patterns.resize(P);
    for (int p = 0; p < P; ++p) {
        Patterns[p].resize(N);
        for (int i = 0; i < N; ++i) {
            Bits pat_tmp;
            for (int r = 0; r < n_bits; ++r) {
                pat_tmp.Set(r, (patterns[p][i][r] + 1) / 2);
            }
            Patterns[p][i] = pat_tmp;
        }
    }

    // =====================================================
    // 3. Convert couplings
    // =====================================================
    Couplings.clear();
    Couplings.resize(N);
    for (int i = 0; i < N; ++i) {
        Couplings[i].clear();
        Couplings[i].reserve(couplings[i].size());
        for (int j = 0; j < (int)couplings[i].size(); ++j) {
            UnsignedInt coupling_tmp((unsigned long long)2 * P);
            for (int r = 0; r < n_bits; ++r) {
                int val = couplings[i][j][r] + P;
                coupling_tmp.Set(r, val);
            }
            Couplings[i].push_back(coupling_tmp);
        }
    }
}

// bit representation {0,1} -> canonical spins {-1,+1}
void HopfieldBits::toCanonical(){
    for (int r = 0; r < n_bits; ++r)
        for (int i = 0; i < N; ++i)
            spins_set[r*N+i] = -1 + 2 * Bits_Spins_Set[i].Get(r);
}

// =====================================================
// OBSERVABLES
// =====================================================

// Compute magnetization m = (2*ones - N)/N for each realization 
void HopfieldBits::GetMagnetizations(vector<double>& magnetizations){
    magnetizations.resize(n_bits);
    vector<int> ones(n_bits, 0);

    for (int i = 0; i < N; ++i)
        for (int r = 0; r < n_bits; ++r)
            ones[r] += Bits_Spins_Set[i].Get(r);

    for (int r = 0; r < n_bits; ++r)
        magnetizations[r] = (2.0 * ones[r] - N) / N;
}

// Converts spin configurations (already in bits) into packed blocks for each realisation
// Uses the same MSB-first convention as PackBlock (and SavePatterns), so that
// spins and patterns are bit-comparable.
void HopfieldBits::GetSpinConfigurations(vector<vector<uint64_t>>& configs){
    const int n_blocks = num_blocks(N);
    configs.assign(n_bits, vector<uint64_t>(n_blocks, 0));

    for (int b = 0; b < n_blocks; ++b){
        int start = b * BITS_PER_BLOCK;
        int end   = min(N, start + BITS_PER_BLOCK);

        for (int i = start; i < end; ++i){
            for (int r = 0; r < n_bits; ++r){
                configs[r][b] <<= 1;
                if (Bits_Spins_Set[i].Get(r))
                    configs[r][b] |= 1ULL;
            }
        }
    }
}

// =====================================================
// METROPOLIS DYNAMICS
// =====================================================

void HopfieldBits::runSweeps(gsl_rng* ran, bool save, double freq){

    Bits c_ij, mask;

    const int total_sweeps    = getNSweeps();
    const int progress_stride = max(1, total_sweeps / 10);
    const int save_stride     = save ? max(1, (int)round(1.0 / freq)) : 0;

    int max_deg = *max_element(neighbor_count.begin(), neighbor_count.end());

    UnsignedInt sum_g   ((unsigned long long int) 2 * max_deg * P);
    UnsignedInt sum_c   ((unsigned long long int) max_deg);
    UnsignedInt sum_c_times_2P ((unsigned long long int) 4 * max_deg * P);
    UnsignedInt sum_cg  ((unsigned long long int) 2 * max_deg * P);
    UnsignedInt RANDOM  ((unsigned long long int) max_deg * P);
    UnsignedInt LHS     ((unsigned long long int) 5 * max_deg * P);
    UnsignedInt RHS     ((unsigned long long int) 5 * max_deg * P);
    UnsignedInt TwoP    ((unsigned long long int) 2 * P);

    TwoP.SetAll(2 * P);

    bool first_debug = false;

    for (int sweep = 0; sweep < total_sweeps; ++sweep){
        for (int step = 0; step < N; ++step){

            int i = gsl_rng_uniform_int(ran, N);
            int deg_i = neighbor_count[i];

            int random = randomNumber(ran, deg_i * P, N);

            Bits& S_i = Bits_Spins_Set[i];

            // unconditional flip
            if (random >= P * deg_i) {
                S_i.ComplementTo();
                continue;
            }

            sum_g.SetAll(0);
            sum_c.SetAll(0);
            sum_cg.SetAll(0);
            RANDOM.SetAll(random);
            LHS.SetAll(0);
            RHS.SetAll(0);
            sum_c_times_2P.SetAll(0);

            for (int k = 0; k < deg_i; ++k){

                int j = neighbors[i][k];

                Bits& S_j = Bits_Spins_Set[j];
                UnsignedInt& g_ij = Couplings[i][k];

                c_ij = ~(S_i ^ S_j);

                sum_g += &g_ij;
                sum_c += &c_ij;

                UnsignedInt tmp = g_ij;
                tmp &= &c_ij;

                sum_cg += &tmp;
            }

            // RHS = P*deg_i + 2*sum(c_ij*g_ij)
            RHS += &sum_cg;
            RHS.MultiplyByTwoTo();
           /* for(int r=0;r<8;r++)
{
    cout
        << "r=" << r
        << " Pdeg=" << P_times_Neighbor_Count[i].Get(r)
        << endl;
}*/
            RHS += &P_times_Neighbor_Count[i];

            // LHS = random + sum_g + 2P*sum_c
            LHS += &RANDOM;
            LHS += &sum_g;

            sum_c.Multiply(&TwoP, &sum_c_times_2P);
            LHS += &sum_c_times_2P;

            // =====================================================
            // FULL DEBUG
            // =====================================================
            if(first_debug)
            {
                cout << "\n====================================\n";
                cout << "FULL DEBUG\n";
                cout << "spin   = " << i << "\n";
                cout << "deg_i  = " << deg_i << "\n";
                cout << "random = " << random << "\n";
                cout << "====================================\n";

                cout << "\nSIZE CHECK\n";

                cout
                    << "sum_g=" << sum_g.GetSize()
                    << " sum_c=" << sum_c.GetSize()
                    << " sum_cg=" << sum_cg.GetSize()
                    << " LHS=" << LHS.GetSize()
                    << " RHS=" << RHS.GetSize()
                    << endl;

                cout << "\nLHS / RHS CHECK\n";

                for(int r = 0; r < min(8, n_bits); ++r)
                {
                    int Si = spins_set[r*N + i];

                    int sum_G_scalar  = 0;
                    int sum_c_scalar  = 0;
                    int sum_cg_scalar = 0;

                    for(int k = 0; k < deg_i; ++k)
                    {
                        int j = neighbors[i][k];

                        int Sj  = spins_set[r*N + j];
                        int Gij = couplings[i][k][r];
                        int gij = Gij + P;

                        int cij = (Si == Sj);

                        sum_G_scalar  += Gij;
                        sum_c_scalar  += cij;
                        sum_cg_scalar += cij * gij;
                    }

                    long long lhs_scalar =
                        random
                        + (sum_G_scalar + P * deg_i)
                        + 2LL * P * sum_c_scalar;

                    long long rhs_scalar =
                        P * deg_i
                        + 2LL * sum_cg_scalar;

                    cout
                        << "r=" << r
                        << " lhs_scalar=" << lhs_scalar
                        << " lhs_bits=" << LHS.Get(r)
                        << " rhs_scalar=" << rhs_scalar
                        << " rhs_bits=" << RHS.Get(r);

                    if(lhs_scalar != (long long)LHS.Get(r))
                        cout << " <<< LHS MISMATCH >>>";

                    if(rhs_scalar != (long long)RHS.Get(r))
                        cout << " <<< RHS MISMATCH >>>";

                    cout << endl;
                }

                mask = (RHS <= LHS);

                cout << "\nCOMPARATOR CHECK\n";

                for(int r = 0; r < min(8, n_bits); ++r)
                {
                    int local_field = 0;

                    for(int k = 0; k < deg_i; ++k)
                    {
                        int j = neighbors[i][k];

                        local_field +=
                            couplings[i][k][r]
                            * spins_set[r*N + j];
                    }

                    int sigma_i = spins_set[r*N + i];

                    int deltaE =
                        sigma_i * local_field;

                    bool scalar_flip =
                        (random >= deltaE);

                    bool bits_flip =
                        mask.Get(r);

                    cout
                        << "r=" << r
                        << " deltaE=" << deltaE
                        << " scalar=" << scalar_flip
                        << " bits=" << bits_flip;

                    if(bits_flip != scalar_flip)
                        cout << " <<< MISMATCH >>>";

                    cout << endl;
                }

                abort();
            }

            mask = (RHS <= LHS);
            S_i ^= &mask;
        }

        if (save && sweep > 0 && (sweep % save_stride == 0))
            SaveSpinConfigurations(sweep);

        if ((sweep + 1) % progress_stride == 0)
            cout << "\rSweep: "
                 << sweep + 1
                 << " ("
                 << (sweep + 1) * 100 / total_sweeps
                 << "%)    "
                 << flush;
    }

    cout << "\n";
}

void HopfieldBits::debugStep(int i, int r) {
    // Pour la réalisation r et le spin i, affiche toutes les quantités intermédiaires
    // et les compare à ce que donnerait le calcul scalaire direct

    int deg_i = neighbor_count[i];
    
    // --- version scalaire directe ---
    int sum_G_scalar = 0;   // sum G_ij (non shifté)
    int sum_c_scalar = 0;
    int sum_cg_scalar = 0;
    int Si = spins_set[r * N + i];  // ±1

    for (int k = 0; k < deg_i; ++k) {
        int j = neighbors[i][k];
        int Sj = spins_set[r * N + j];
        int G_ij = couplings[i][k][r];   // in [-P, P]
        int g_ij = G_ij + P;             // in [0, 2P]
        int c_ij = (Si == Sj) ? 1 : 0;

        sum_G_scalar += G_ij;
        sum_c_scalar += c_ij;
        sum_cg_scalar += c_ij * g_ij;
    }

    int lhs_scalar = 0 /*R: ignore*/ + (sum_G_scalar + P * deg_i) + 2 * P * sum_c_scalar;
    int rhs_scalar = P * deg_i + 2 * sum_cg_scalar;
    // note: sum_g_scalar en bits = sum(g_ij) = sum(G_ij + P) = sum_G_scalar + P*deg_i

    cout << "=== DEBUG step, spin=" << i << ", realization=" << r << " ===" << endl;
    cout << "Si=" << Si << ", deg_i=" << deg_i << endl;
    cout << "[SCALAR] sum_G=" << sum_G_scalar 
         << "  sum_g(shifted)=" << sum_G_scalar + P*deg_i
         << "  sum_c=" << sum_c_scalar 
         << "  sum_cg=" << sum_cg_scalar << endl;
    cout << "[SCALAR] LHS(sans R)=" << lhs_scalar 
         << "  RHS=" << rhs_scalar << endl;

    // --- version bitwise ---
    UnsignedInt sum_g_b ((unsigned long long) 2 * deg_i * P);
    UnsignedInt sum_c_b ((unsigned long long) deg_i);
    UnsignedInt sum_cg_b((unsigned long long) 2 * deg_i * P);
    UnsignedInt TwoP_b  ((unsigned long long) 2 * P);
    UnsignedInt sum_c_2P((unsigned long long) 4 * deg_i * P);
    TwoP_b.SetAll(2 * P);
    sum_g_b.SetAll(0); sum_c_b.SetAll(0); sum_cg_b.SetAll(0);

    Bits& S_i = Bits_Spins_Set[i];
    Bits c_ij;

    for (int k = 0; k < deg_i; ++k) {
        int j = neighbors[i][k];
        Bits& S_j = Bits_Spins_Set[j];
        UnsignedInt& g_ij = Couplings[i][k];

        c_ij = ~(S_i ^ S_j);

        sum_g_b  += &g_ij;
        sum_c_b  += &c_ij;

        UnsignedInt tmp = g_ij;
        tmp &= &c_ij;
        sum_cg_b += &tmp;
    }

    sum_c_b.Multiply(&TwoP_b, &sum_c_2P);

    // Extraire la valeur pour la réalisation r
    cout << "[BITS]   sum_g=" << sum_g_b.Get(r)
         << "  sum_c=" << sum_c_b.Get(r)
         << "  sum_cg=" << sum_cg_b.Get(r)
         << "  2P*sum_c=" << sum_c_2P.Get(r) << endl;

    // Vérifications
    cout << "[CHECK]  sum_g: scalar=" << sum_G_scalar + P*deg_i 
         << " bits=" << sum_g_b.Get(r)
         << (sum_g_b.Get(r) == (unsigned long long)(sum_G_scalar + P*deg_i) ? " OK" : " *** MISMATCH ***") << endl;

    cout << "[CHECK]  sum_c: scalar=" << sum_c_scalar 
         << " bits=" << sum_c_b.Get(r)
         << (sum_c_b.Get(r) == (unsigned long long)sum_c_scalar ? " OK" : " *** MISMATCH ***") << endl;

    cout << "[CHECK]  sum_cg: scalar=" << sum_cg_scalar 
         << " bits=" << sum_cg_b.Get(r)
         << (sum_cg_b.Get(r) == (unsigned long long)sum_cg_scalar ? " OK" : " *** MISMATCH ***") << endl;

    cout << "[CHECK]  2P*sum_c: scalar=" << 2*P*sum_c_scalar 
         << " bits=" << sum_c_2P.Get(r)
         << (sum_c_2P.Get(r) == (unsigned long long)(2*P*sum_c_scalar) ? " OK" : " *** MISMATCH ***") << endl;
}

// =====================================================
// PUBLIC API
// =====================================================

// Run simulation without saving (thermalization)
void HopfieldBits::evolve(gsl_rng* ran, const string& filename){
    fromCanonical();
    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    runSweeps(ran, /*save=*/false, 0.0);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve called, closing: " << filename << endl;
    toCanonical();
}

// Run simulation and save magnetizations at the given frequency
void HopfieldBits::evolve_save(gsl_rng* ran, double freq, const string& filename){
    fromCanonical();
    cout << "Bitwise conversion done"<<endl;
    OpenSpinFiles(filename);
    SaveSpinConfigurations(0);
    cout << "evolve_save called, opening: " << filename << endl;
    runSweeps(ran, /*save=*/true, freq);
    SaveSpinConfigurations(getNSweeps());
    CloseSpinFiles();
    cout << "evolve_save called, closing: " << filename << endl;
    toCanonical();
}