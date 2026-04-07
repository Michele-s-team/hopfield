#include <iostream>
#include <cstdio>
#include <cmath>
#include <math.h>
#include <vector>
#include <fstream>
#include <strstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <algorithm>

//include per il mac

#include <gsl_errno.h>
#include <gsl_roots.h>
#include <gsl_math.h>
#include <gsl_rng.h>
#include <gsl_randist.h>
#include <gsl_sf_pow_int.h>

/*Change this to parallelize the code*/
#include <mpi.h>

/*
 Compile on mac:
 
 clear; clear; mpic++ multi_spin_coding_hea_9_q_mod.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated -I ./ -I/usr/local/include/gsl/ -o multi_spin_coding_hea_9_q_mod.o -Wall -DHAVE_INLINE
 clear; clear; mpic++ multi_spin_coding_hea_9_q_mod.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated -I ./ -I/usr/local/include/gsl/ -o multi_spin_coding_hea_9_q_mod.o -DHAVE_INLINE
 clear; clear; g++ multi_spin_coding_hea_9_q_mod.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated -I ./ -I/usr/local/include/gsl/ -o multi_spin_coding_hea_9_q_mod.o -Wall -DHAVE_INLINE
 clear; clear; g++ multi_spin_coding_hea_9_q_mod.cpp -llapack -lgsl -lcblas -lm -O3 -Wno-deprecated -I ./ -I/usr/local/include/gsl/ -o multi_spin_coding_hea_9_q_mod.o -Wall -DHAVE_INLINE -g
 
 
 
 Compile on curie calcsub:
 mpic++ multi_spin_coding_hea_9_q_mod.cpp -I /usr/include/gsl/ -llapack -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o  multi_spin_coding_hea_9_q_mod.o -DHAVE_INLINE
 
 Compile on mesopsl:
 mpicc multi_spin_coding_hea_9_q_mod.cpp -I/usr/include/gsl -lstdc++ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o multi_spin_coding_hea_9_q_mod.o -DHAVE_INLINE

 Compile on curie abacus:
 mpic++ multi_spin_coding_hea_9_q_mod.cpp -I /mnt/beegfs/home/mcastel1/gsl/include/gsl  -I/mnt/beegfs/home/mcastel1/gsl/include/ -L/mnt/beegfs/home/mcastel1/gsl/lib/ -lgsl -lgslcblas -lm -O3 -Wno-deprecated  -o  multi_spin_coding_hea_9_q_mod.o -DHAVE_INLINE
 
 mpirun -np 2 ./multi_spin_coding_hea_9_q_mod.o -k 3 -s 0.6 -b 1.0 -B 2.0 -r 2 -T 1024 -S 4 -c 16 -w 16 -i /Users/michele/Desktop/ -o /Users/michele/Desktop/
 valgrind --leak-check=full ./multi_spin_coding_hea_9_q_mod.o -k 6 -s 0.6 -b 0.2 -B 5.0 -r 32 -T 1024 -S 2 -c 16 -w 32 -i /Users/michele/Desktop/ -o /Users/michele/Desktop/
 
 
 */

/*
 Notes:
 
 H = - \sum_{i j} J_{ij} S_i S_j
 
 */

#define n_bits 64
#define seed_shift_S 1
#define seed_shift_J 1
#define s_test 63
#define eps_abs 0
#define eps_rel (1e-14)
//#define t_mea 16

//#define data_directory "/Users/michelecastellana_old/Documents/rg_sg/"
//#define data_directory "/scratch/gpfs/michelec/hea5/"
//#define data_directory "/Genomics/grid/users/michelec/data_hea1/"
//#define in_directory "/tigress/michelec/data_hea/noH/k9/"
//#define in_directory "/scratch/gpfs/michelec/hea1/"

const unsigned long long int ullong_1 = 1;

using namespace std;

struct model_parameters{
    unsigned long long int k;
    double sigma, z;
};

vector< vector<int> > S_test;

void generate(int N, vector<int> S){
    
    if(S.size() < N){
        
        vector<int> v=S;
        v.resize(S.size()+1);
        
        v[v.size()-1]=1;
        generate(N, v);
        
        v[v.size()-1]=-1;
        generate(N, v);
        
    }else{
        S_test.push_back(S);
    }
    
}


inline unsigned long long int two_pow(unsigned long long int i){return ((unsigned long long int)gsl_sf_pow_int(2.0,i));}

int inline bits(int n){
    
    int s;
    
    for(s=0; two_pow(s) <= (unsigned long long int)n; s++){}
    
    return s;
    
}

// n_2 has bits(n) components.

void inline convert(int n, unsigned long long int * n_2){
    
    int s, i, bits_of_n = bits(n);
    
    for(i=n, s=0; s<bits_of_n; s++){
        n_2[s]=(i % 2);
        i -= (i % 2);
        i=(int)(((double)i)/2.0);
    }
    
}

int ud(int a, int b){
    
    int i, ja, jb;
    
    i=0;
    do{
        for(ja=0; !(((int)gsl_sf_pow_int(2,i)*ja <= a) && (a < (int)gsl_sf_pow_int(2,i)*(ja+1))); ja++){};
        for(jb=0; !(((int)gsl_sf_pow_int(2,i)*jb <= b) && (b < (int)gsl_sf_pow_int(2,i)*(jb+1))); jb++){};
        i++;
    }while(ja != jb);
    
    return(i-1);
    
}

//This function returns E[\sum_{i>1} I(J_{1i} !=0 )] - z
double my_F(double C, void *q) {
    
    struct model_parameters *p = (struct model_parameters *)q;
    unsigned int l;
    double result;
    
    for(result = 0.0, l=1; l<(p->k)+1; l++){
        result += pow(2.0,((double)(l-1))) * (1.0 - exp(-C*pow(2.0,-2.0*(p->sigma)*((double)(l-1)))));
    }
    
    return(result - (p->z));
    
}

int main(int argc, char *argv[]){
    
    unsigned long long int N, N_2, N_m_1, N_l, n_l, n_all, k=0, n_64samples=0, i, j, l, p, t, t_tot=0, t_swa=0, t_wri=0, all_0, all_1, r, **S, **J, *i_J, *i_J0, *X_swe, *a_swe, *carry_X_swe, **ne, rho, *X_rho, changer, check, **a_swa/*, *Y, *carry_Y*/, tau, *X_tau, carry, ullong, changer_swa, Sa_te, Sb_te, *n_links_2, *tab_two_pow_k_swa, *****add_L_swa, *****add_R_swa, *L, *R, a_swa_a_te, a_swa_b_te, *N_2_2, /*Uncomment this to check acceptance rates*/ /**acc_swe = NULL,*/ *acc_swa = NULL, *a_mea, *X_mea, *carry_X_mea, *tab_two_pow_k_mea, ****add_L_mea, ****add_R_mea, *I, *I0;
    double sigma = 0.0, z = 6.0, z_J, beta_m = 0.0, beta_M = 0.0, *beta, C, C_lo, C_hi, status, iter, max_iter = 1024/*beta with geometric sequence, c_beta = 0.0*/;
    int options,/*Change this to parallelize the code*/ rank, size, nr=0, a, b, c, f, g, m, n, o, q, s, n_links, *n_links_per_spin, bits_n_links, *bits_n_links_per_spin, bits_N_2, bits_N_m_1, *d, nz_bits_n_links, *bits_p_p_1, *k_swa, **tab_swa, nz_bits_N_2, *k_mea, **tab_mea;
    ofstream outfile_Q, outfile_S, outfile_S_save;
    ifstream infile_S;
    string input_directory, output_directory, dummy;
    stringstream filename_Q, filename_S, filename_S_save, infilename_S;
    model_parameters MODEL_PARAMETERS;
    const gsl_root_fsolver_type *solver_type;
    gsl_root_fsolver *solver;
    gsl_function F;
    /*Uncomment this to check time */
    
    clock_t start = 0, stop = 0, time_cl = 0;
    
    gsl_rng *myran_S, *myran_J;
    
    gsl_rng_env_setup();
    myran_S = gsl_rng_alloc(gsl_rng_gfsr4);
    myran_J = gsl_rng_alloc(gsl_rng_gfsr4);
    
    while ((options = getopt(argc, argv, ":k:s:b:B:r:T:S:c:w:i:o:")) != -1) {
        
        switch (options) {
                
            case 'k':
                k = (unsigned long long int)atoi(optarg);
                break;
                
            case 's':
                sigma = strtod(optarg,argv);
                break;
                
            case 'b':
                beta_m = strtod(optarg,argv);
                break;
                
            case 'B':
                beta_M = strtod(optarg,argv);
                break;
                
            case 'r':
                nr = (3*(int)atoi(optarg));
                break;
                
            case 'T':
                t_tot = (unsigned long long int)atoi(optarg);
                break;
                
            case 'S':
                n_64samples = (unsigned long long int)atoi(optarg);
                break;
                
            case 'c':
                t_swa = (unsigned long long int)atoi(optarg);
                break;
                
            case 'w':
                t_wri = (unsigned long long int)atoi(optarg);
                break;
                
            case 'i':
                input_directory.assign(optarg);
                break;
                
            case 'o':
                output_directory.assign(optarg);
                break;
                
        }
        
    }
    
    cout.precision(20);
    
    /*Change this to parallelize the code*/
    //
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //
    
    if((rank==0) && ((n_64samples % size) != 0)){cout << "\nn_64samples % size != 0!!"; flush(cout);}
    if(rank==0){
        cout << "\nInput directory = " << input_directory;
        cout << "\nOutput directory = " << output_directory;
        flush(cout);
    }
    
    (MODEL_PARAMETERS.sigma) = sigma;
    (MODEL_PARAMETERS.z) = z;
    (MODEL_PARAMETERS.k) = k;
    F.function = &my_F;
    F.params = &MODEL_PARAMETERS;
    
    N = two_pow(k);
    N_2 = N/2;
    N_m_1 = N-1;
    bits_N_m_1 = bits(N-1);
    bits_N_2 = bits(N_2);
    N_2_2 = new unsigned long long int [bits_N_2];
    convert(N_2,N_2_2);
    
    
    if(rank==0){cout << "\nk = " << k;}
    
    /*Open file for Q*/
    filename_Q << output_directory << "Q" << "_k" << k << "_s" << sigma << "_b" << beta_m << "_B" << beta_M << "_n_r" << nr << "_n_sw" << t_tot << "_n_64sa" << n_64samples/size << "_t_swa" << t_swa << "_t_wri" << t_wri << "_cpu" << rank << ".dat";
    outfile_Q.open(filename_Q.str().c_str());
    
    /*Open file for S*/
    filename_S << output_directory << "S" << "_k" << k << "_s" << sigma << "_b" << beta_m << "_B" << beta_M << "_n_r" << nr << "_n_sw" << t_tot << "_n_64sa" << n_64samples/size << "_t_swa" << t_swa << "_t_wri" << t_wri << "_cpu" << rank << ".dat";
    outfile_S.open(filename_S.str().c_str());

    /*Open set name for for outfile_S_save*/
    filename_S_save << output_directory << "S" << "_k" << k << "_s" << sigma << "_b" << beta_m << "_B" << beta_M << "_n_r" << nr << "_n_sw" << t_tot << "_n_64sa" << n_64samples/size << "_t_swa" << t_swa << "_t_wri" << t_wri << "_cpu" << rank << ".sav";

    
    /*Alloc*/
    
    S = new unsigned long long int* [nr];
    for(a=0; a<nr; a++){
        S[a] = new unsigned long long int [N];
    }
    J = new unsigned long long int* [N];
    i_J = new unsigned long long int [N_m_1];
    i_J0 = i_J;
    I = new unsigned long long int [N];
    I0 = I;
    ne = new unsigned long long int* [N];
    X_swe = new unsigned long long int [bits_N_m_1];
    X_mea = new unsigned long long int [bits_N_2];
    X_rho = new unsigned long long int [bits_N_m_1];
    X_tau = new unsigned long long int [n_bits];
    a_swe = new unsigned long long int [N_m_1];
    a_mea = new unsigned long long int [N_2];
    a_swa = new unsigned long long int* [nr];
    carry_X_swe = new unsigned long long int [bits_N_m_1];
    carry_X_mea = new unsigned long long int [bits_N_2];
    n_links_per_spin = new int [N];
    bits_n_links_per_spin = new int [N];
    bits_p_p_1 = new int [N-1];
    d = new int [N];
    beta = new double [nr];
    /*Uncomment this to compute acceptance rates*/
    
    if(rank==0){
        //acc_swe = new unsigned long long int [nr/3];
        acc_swa = new unsigned long long int [(nr/3)-1];
    }
    
    /*Tabluate I*/
    for(i=0; i<N; i++){I[i]=i;}
    
    for(i=0, s=0; s<bits_N_2; s++){
        if(N_2_2[s] != 0){i++;}
    }
    nz_bits_N_2 = i;
    
    k_mea = new int [nz_bits_N_2];
    tab_mea = new int* [nz_bits_N_2];
    tab_two_pow_k_mea = new unsigned long long int [nz_bits_N_2];
    
    for(i=0, s=bits_N_2-1; s>=0; s--){
        if(N_2_2[s] != 0){
            k_mea[i]=s;
            i++;
        }
    }
    
    for(s=0; s<nz_bits_N_2; s++){
        tab_mea[s] = new int [k_mea[s]];
        for(p=0; p<k_mea[s]; p++){tab_mea[s][p] = two_pow(k_mea[s]-(p+1));}
    }
    for(s=0; s<nz_bits_N_2; s++){tab_two_pow_k_mea[s] = two_pow(k_mea[s]);}
    
    /*Tabulate memory addresses for the meas sum*/
    add_L_mea = new unsigned long long int *** [nz_bits_N_2];
    add_R_mea = new unsigned long long int *** [nz_bits_N_2];
    for(p=0, s=0; s<nz_bits_N_2; s++){
        
        // cout << "\ns=" << s;
        add_L_mea[s] = new unsigned long long int ** [k_mea[s]];
        add_R_mea[s] = new unsigned long long int ** [k_mea[s]];
        
        for(m=0; m<k_mea[s]; m++){
            
            // cout << "\nm=" << m;
            add_L_mea[s][m] = new unsigned long long int * [two_pow(k_mea[s]-(m+1))];
            add_R_mea[s][m] = new unsigned long long int * [two_pow(k_mea[s]-(m+1))];
            
            for(j=0; j<two_pow(k_mea[s]-(m+1)); j++){
                
                add_L_mea[s][m][j] = a_mea+p+j*two_pow(m+1);
                add_R_mea[s][m][j] = add_L_mea[s][m][j]+two_pow(m);
                // cout << "\nj=" << j << "\tL = " << add_L_mea[s][m][j] << "\tR = " << add_R_mea[s][m][j];
                
            }
        }
        
        p+=two_pow(k_mea[s]);
    }
    
    /*Tabulate memory addresses for the swap sum*/
    add_L_swa = new unsigned long long int **** [nr];
    add_R_swa = new unsigned long long int **** [nr];
    
    
    
    gsl_rng_set(myran_S,seed_shift_S+rank);
    gsl_rng_set(myran_J,seed_shift_J+rank);
    
    /*Definisce all_0 e calcola all_1 */
    all_1 = 0; all_0 = 0;
    for(s=0; s<n_bits; s++){all_1 = all_1 | (((unsigned long long int)1) << s);}
    
    /*Compute betas*/
    /*Here betas are evenly spaced*/
    //
    for(a=0; a<nr/3; a++){
        beta[a] = beta_m + (beta_M-beta_m)/((double)((nr/3)-1))*(double)a;
        beta[(nr/3)+a] = beta[a];
        beta[2*(nr/3)+a] = beta[a];
    }
    //
    /*Here Ts are evenly spaced*/
    /*
     for(a=0; a<nr/3; a++){
     beta[a] = 1.0/(1.0/beta_m + ((1.0/beta_M)-(1.0/beta_m))/((double)((nr/3)-1))*(double)a);
     beta[(nr/3)+a] = beta[a];
     beta[2*(nr/3)+a] = beta[a];
     }
     */
    /*Here betas with geometric sequence*/
    /*
     c_beta = pow(beta_M/beta_m, 1.0/((double)(nr/3-1)));
     if(rank==0){
     cout << "\n1/c_beta = " << 1.0/c_beta;
     flush(cout);
     }
     for(a=0; a<nr/3; a++){
     beta[a] = beta_m*gsl_sf_pow_int(c_beta, a);
     beta[(nr/3)+a] = beta[a];
     beta[2*(nr/3)+a] = beta[a];
     }
     */
    
    //Solve for C
    //
    if(rank==0){
        cout << "\nSolving for C ...";
        flush(cout);
    }
    solver_type = gsl_root_fsolver_brent;
    solver = gsl_root_fsolver_alloc(solver_type);
    iter = 0;
    C_lo = 0.0;
    C_hi = 100.0;
    gsl_root_fsolver_set(solver, &F, C_lo, C_hi);
    if(rank==0){
        cout << "\nUsing " << gsl_root_fsolver_name(solver) << " metod.";
        cout <<  "\niter lower upper root err(est)";
        flush(cout);
    }
    
    do{
        iter++;
        status = gsl_root_fsolver_iterate(solver);
        C = gsl_root_fsolver_root(solver);
        C_lo = gsl_root_fsolver_x_lower(solver);
        C_hi = gsl_root_fsolver_x_upper(solver);
        status = gsl_root_test_interval(C_lo, C_hi, eps_abs, eps_rel);
        if((rank==0) && (status == GSL_SUCCESS)){
            cout << "\nConverged:";
            flush(cout);
        }
        if(rank==0){
            cout << "\n" << iter << "\t[" << C_lo << " , " << C_hi << "] \t " << C << "\t" <<  C_hi - C_lo;
            flush(cout);
        }
    }while((status == GSL_CONTINUE) && (iter < max_iter));
    if(status !=  GSL_SUCCESS){
        cout << "\nCannot solve for C!!";
        flush(cout);
        return 0;
    }
    if(rank==0){
        cout << "\n... done.";
        flush(cout);
    }
    C = (C_lo+C_hi)/2.0;
    cout << "\nRank = " << rank << "\t C = " << C;
    //
    
    
    
    /*MC over n_64samples*n_bits samples*/
    /*Uncomment this to check time / acceptance rates*/
    
    if(rank==0){
        //time_cl = 0;
        //for(a=0; a<nr/3; a++){acc_swe[a] = 0;}
        for(a=0; a<(nr/3)-1; a++){acc_swa[a] = 0;}
    }
    
    /*Read S from file*/
    /*
     infilename_S << input_directory << "S" << "_k" << k << "_s" << sigma << "_b" << beta_m << "_B" << beta_M << "_n_r" << nr << "_n_sw" << t_tot << "_n_64sa" << n_64samples/size << "_t_swa" << t_swa << "_t_wri" << t_wri << "_cpu" << rank << ".sav";
     infile_S.open(infilename_S.str().c_str());
     
     if(!infile_S){
     cout << "\nError opening file " << infilename_S.str().c_str()  << ".";
     return 0;
     }
     */
    
    
    for(q=0; q<n_64samples/size; q++){
        
        /*Uncomment this to check time*/
        if(rank==0){start = clock();}
        
        /*Draw the Js with the new method (same result as the old method)*/
        for(i=0; i<N; i++){
            J[i] = (unsigned long long int*)malloc(0);
            ne[i] = (unsigned long long int*)malloc(0);
        }
        
        for(i=0; i<N; i++){
            n_links_per_spin[i] = 0;
            d[i] = 0;
        }
        
        for(n_links=0, i=0; i<N; i++){
            
            I = I0;
            i_J = i_J0;
            n_all = 0;
            
            //cout << "\n\ni = " << i;
            for(l=k; l>0; l--){
                
                //cout << "\n\tl = " << l;
                N_l = two_pow(l-1);
                
                if(i<I[N_l]){
                    //Model of castellana2015non
                    /*
                     n_l = gsl_ran_binomial(myran_J, pow(2.0,-2.0*sigma*((double)(l-1))), N_l);
                     */
                    
                    //Model with fixed coordination number
                    //
                    n_l = gsl_ran_binomial(myran_J, (1.0-exp(-C*pow(2.0,-2.0*sigma*((double)(l-1))))), N_l);
                    //
                    
                    gsl_ran_choose(myran_J, i_J, n_l, I+N_l, N_l, sizeof(unsigned long long int));
                    
                    //cout << "\n\t\tN_l = " << N_l << "\tn_l = " << n_l;
                    //cout << "\n\t\tSites at distance " << l << ": ";
                    // for(j=N_l; j<2*N_l; j++){
                    //   cout << I[j] << " ";
                    // }
                    //cout << "\n\t\tNeighbors at distance " << l << ": ";
                    // for(j=0; j<n_l; j++){
                    //   cout << i_J[j] << " ";
                    // }
                    i_J += n_l;
                    n_all += n_l;
                }
                
                if(i>=I[N_l]){
                    I += N_l;
                }
                
            }
            
            i_J = i_J0;
            sort(i_J, i_J + n_all);
            
            //cout << "\n\tNumber of neighbors = " << n_all;
            //cout << "\n\tNeighbors: ";
            // for(j=0; j<n_all; j++){
            // 	cout << i_J[j] << " ";
            // }
            
            
            for(j=0; j<n_all; j++){
                
                n_links++;
                n_links_per_spin[i]++;
                n_links_per_spin[i_J[j]]++;
                J[i] = (unsigned long long int*)realloc(J[i], n_links_per_spin[i]*sizeof(unsigned long long int));
                J[i_J[j]] = (unsigned long long int*)realloc(J[i_J[j]], n_links_per_spin[i_J[j]]*sizeof(unsigned long long int));
                ne[i] = (unsigned long long int*)realloc(ne[i], n_links_per_spin[i]*sizeof(unsigned long long int));
                ne[i_J[j]] = (unsigned long long int*)realloc(ne[i_J[j]], n_links_per_spin[i_J[j]]*sizeof(unsigned long long int));
                d[i_J[j]]++;
                
                for(J[i][n_links_per_spin[i]-1]=0, s=0; s<n_bits; s++){
                    r = (((unsigned long long int)(gsl_rng_uniform_int(myran_J,2))) << s);
                    J[i][n_links_per_spin[i]-1] = ((J[i][n_links_per_spin[i]-1]) | r);
                }
                J[i_J[j]][n_links_per_spin[i_J[j]]-1] = J[i][n_links_per_spin[i]-1];
                
                ne[i][n_links_per_spin[i]-1] = i_J[j];
                ne[i_J[j]][n_links_per_spin[i_J[j]]-1] = i;
                
            }
            
        }
        
        I = I0;
        
        //
        
        /*Draw the Js with the old method*/
        /*
         for(i=0; i<N; i++){
         J[i] = (unsigned long long int*)malloc(0);
         ne[i] = (unsigned long long int*)malloc(0);
         }
         
         for(i=0; i<N; i++){
         n_links_per_spin[i] = 0;
         d[i] = 0;
         }
         
         for(n_links=0, i=0; i<N; i++){
         for(j=i+1; j<N; j++){
         //Old definition
         //if(gsl_rng_uniform(myran_J) < pow(2.0,-2.0*sigma*((double)ud(i,j)))){
         //New definition
         if(gsl_rng_uniform(myran_J) < pow(2.0,-2.0*sigma*((double)(ud(i,j)-1)))){
         n_links++;
         n_links_per_spin[i]++;
         n_links_per_spin[j]++;
         J[i] = (unsigned long long int*)realloc(J[i], n_links_per_spin[i]*sizeof(unsigned long long int));
         J[j] = (unsigned long long int*)realloc(J[j], n_links_per_spin[j]*sizeof(unsigned long long int));
         ne[i] = (unsigned long long int*)realloc(ne[i], n_links_per_spin[i]*sizeof(unsigned long long int));
         ne[j] = (unsigned long long int*)realloc(ne[j], n_links_per_spin[j]*sizeof(unsigned long long int));
         d[j]++;
         
         for(J[i][n_links_per_spin[i]-1]=0, s=0; s<n_bits; s++){
         r = (((unsigned long long int)(gsl_rng_uniform_int(myran_J,2))) << s);
         J[i][n_links_per_spin[i]-1] = ((J[i][n_links_per_spin[i]-1]) | r);
         }
         J[j][n_links_per_spin[j]-1] = J[i][n_links_per_spin[i]-1];
         
         ne[i][n_links_per_spin[i]-1] = j;
         ne[j][n_links_per_spin[j]-1] = i;
         // cout << "\n(" << i << "," << j << ") J=" << J[i][n_links_per_spin[i]-1];
         
         }
         }
         }
         */
        
        /*Calcola il numero di bit necessari per scrivere n_links e n_links_per_spin in base 2*/
        for(i=0; i<N; i++){
            bits_n_links_per_spin[i] = bits(n_links_per_spin[i]);
        }
        bits_n_links = bits(n_links);
        if(rank==0){
            for(z_J=0.0, i=0; i<N; i++){
                z_J += ((double)(n_links_per_spin[i]));
            }
            z_J/=((double)N);
            cout << "\nNumber of Js = " << n_links << "\nz = " << z_J;
            flush(cout);
        }
        /*Calcola il numero di bit necessari per scrivere p+1 in base 2*/
        for(p=0; p<N-1; p++){
            bits_p_p_1[p] = bits(p+1);
        }
        
        n_links_2 = new unsigned long long int [bits_n_links];
        convert(n_links,n_links_2);
        for(i=0, s=0; s<bits_n_links; s++){
            if(n_links_2[s] != 0){i++;}
        }
        nz_bits_n_links = i;
        
        k_swa = new int [nz_bits_n_links];
        tab_swa = new int* [nz_bits_n_links];
        tab_two_pow_k_swa = new unsigned long long int [nz_bits_n_links];
        
        for(i=0, s=bits_n_links-1; s>=0; s--){
            if(n_links_2[s] != 0){
                k_swa[i]=s;
                i++;
            }
        }
        
        for(s=0; s<nz_bits_n_links; s++){
            tab_swa[s] = new int [k_swa[s]];
            for(p=0; p<k_swa[s]; p++){tab_swa[s][p] = two_pow(k_swa[s]-(p+1));}
        }
        for(s=0; s<nz_bits_n_links; s++){tab_two_pow_k_swa[s] = two_pow(k_swa[s]);}
        
        /*
         cout << "\n";
         for(i=0; i<N; i++){
         cout << "\nn[" << i << "]=" << n_links_per_spin[i] << "\tbits_n[" << i << "]=" << bits_n_links_per_spin[i] << "\td[" << i << "] = " << d[i];
         for(j=0; j<n_links_per_spin[i]; j++){
         cout << "\n\tne[" << i << "][" << j << "] = " << ne[i][j];
         cout << "\nJ[" << i << "][" << j << "]=" << J[i][j];
         }
         }
         */
        
        /*Alloc a_swa*/
        for(a=0; a<nr; a++){
            a_swa[a] = new unsigned long long int [n_links];
        }
        /*
         Y = new unsigned long long int [bits_n_links];
         // X_swa_b = new unsigned long long int [bits_n_links];
         carry_Y = new unsigned long long int [bits_n_links];
         */
        
        /*Tabulate memory addresses for the swap sum*/
        for(a=0; a<nr; a++){
            add_L_swa[a] = new unsigned long long int *** [nz_bits_n_links];
            add_R_swa[a] = new unsigned long long int *** [nz_bits_n_links];
        }
        for(p=0, s=0; s<nz_bits_n_links; s++){
            
            // cout << "\ns=" << s;
            for(a=0; a<nr; a++){
                add_L_swa[a][s] = new unsigned long long int ** [k_swa[s]];
                add_R_swa[a][s] = new unsigned long long int ** [k_swa[s]];
            }
            
            for(m=0; m<k_swa[s]; m++){
                
                // cout << "\nm=" << m;
                for(a=0; a<nr; a++){
                    add_L_swa[a][s][m] = new unsigned long long int * [two_pow(k_swa[s]-(m+1))];
                    add_R_swa[a][s][m] = new unsigned long long int * [two_pow(k_swa[s]-(m+1))];
                }
                
                for(j=0; j<two_pow(k_swa[s]-(m+1)); j++){
                    
                    for(a=0; a<nr; a++){
                        add_L_swa[a][s][m][j] = a_swa[a]+p+j*two_pow(m+1);
                        add_R_swa[a][s][m][j] = add_L_swa[a][s][m][j]+two_pow(m);
                    }
                    // cout << "\nj=" << j << "\tL = " << add_L_swa[s][m][j] << "\tR = " << add_R_swa[s][m][j];
                    
                }
            }
            
            p+=two_pow(k_swa[s]);
        }
        
        
        
        
        /*Draw S*/
        //
        for(a=0; a<nr; a++){
            for(i=0; i<N; i++){
                S[a][i] = 0;
                for(s=0; s<n_bits; s++){S[a][i] = (S[a][i] | ((unsigned long long int)(gsl_rng_uniform_int(myran_S,2)) << s));}
            }
        }
        //
        
        /*Read S from file*/
        /*
         if(rank==0){cout << "\nInitial S:";}
         for(a=0; a<nr; a++){
         if(rank==0){cout << "\n";}
         for(i=0; i<N; i++){
         S[a][i] = 0;
         infile_S >> S[a][i];
         if(rank==0){cout << S[a][i] << " ";}
         }
         }
         if(rank==0){flush(cout);}
         getline(infile_S, dummy);
         getline(infile_S, dummy);
         
         */
        
        
        
        outfile_Q << "\n";
        
        if(rank==0){
            stop = clock();
            time_cl = stop - start;
            cout << "\nTime for initializing MC = " << ((double)time_cl)/((double)CLOCKS_PER_SEC) << " s";
            flush(cout);
        }
        
        
        /*Uncomment this to check time*/
        if(rank==0){start = clock();}
        
        for(t=0; t<t_tot; t++){
            
            /*MC sweep*/
            for(a=0; a<nr; a++){
                
                for(i=0; i<N; i++){
                    //                    run through all spins in the network and try to flip S_i
//                    convert 'delta' in binaryt form and write it in X_swe
                    for(s=0; s<bits_n_links_per_spin[i]; s++){X_swe[s] = 0;}
                    for(p=0; p<n_links_per_spin[i]; p++){
                        a_swe[p] = (J[i][p])^(S[a][i])^(S[a][ne[i][p]]);
                    }
                    
                    for(p=1, X_swe[0]=a_swe[0]; p<n_links_per_spin[i]; p++){
                        
                        for(s=0; s<bits_p_p_1[p]; s++){
                            if(s==0){
                                carry_X_swe[s] = ((X_swe[s]) & (a_swe[p]));
                                (X_swe[s])^= (a_swe[p]);
                            }else{
                                carry_X_swe[s] = ((X_swe[s]) & (carry_X_swe[s-1]));
                                (X_swe[s])^= (carry_X_swe[s-1]);
                            }
                        }
                        
                    }
                    
                    /*
                     int n = 0;
                     for(p=0; p<n_links_per_spin[i]; p++){n += ((a_swe[p] & (ullong_1 << s_test)) >> s_test);}
                     cout << "\n\nsum_{10} = " << n;
                     for(n=0, s=0; s<bits_n_links_per_spin[i]; s++){n += two_pow(s) * ((X_swe[s] & (ullong_1 << s_test)) >> s_test);}
                     cout << "\nsum_{2} = " << n;
                     */
                    
                    rho = (unsigned long long int)(((double)n_links_per_spin[i])/2.0 + 1.0/(4.0*beta[a])*gsl_ran_exponential(myran_S,1.0));
                    
                    if(rho >= n_links_per_spin[i]){
                        /*Here I flip all spins.*/
                        changer = all_0;
                    }else{
                        /*Here I may or may not flip. I have to compute the changer word.*/
                        
                        //cout << "\n\nrho_{10} = " << rho;
                        
                        /*Converto rho in binario mettendo i suoi primi bits_n_links_per_spin[i] bits in X_rho[]*/
                        for(s=0; s<bits_n_links_per_spin[i]; s++){
                            if(((rho >> s) & ullong_1) == all_0){
                                X_rho[s] = all_0;
                            }else{
                                X_rho[s] = all_1;
                            }
                        }
                        
                        /*
                         int n = 0;
                         for(n=0, s=0; s<bits_n_links_per_spin[i]; s++){n += two_pow(s) * ((X_rho[s] & (ullong_1 << s_test)) >> s_test);}
                         cout << "\nsum_{2} = " << n;
                         */
                        
                        /*Confronto X_rho[] con X_swe[] e scrivo in changer il risultato. changer è uguale a 1 se X_rho < X_swe e a 0 altrimenti.*/
                        //                        initialize changer for the last bit
                        changer = (~(X_rho[bits_n_links_per_spin[i] - 1])) & (X_swe[bits_n_links_per_spin[i] - 1]);
                        check = ((X_rho[bits_n_links_per_spin[i] - 1]) ^ (X_swe[bits_n_links_per_spin[i] - 1]));
                        
                        for(s=bits_n_links_per_spin[i] - 2; s >=0; s--){
                            changer = ((check & changer) | ((~check) & ((~(X_rho[s])) & (X_swe[s]))) );
                            check = (check | ((X_rho[s]) ^ (X_swe[s])));
                        }
                        
                        /*
                         // cout << "\ni = " << i;
                         for(s=0; s<n_bits; s++){
                         unsigned long long int m, n;
                         for(m=0, p=0; p<bits_n_links_per_spin[i]; p++){m += two_pow(p) * ((X_swe[p] & (ullong_1 << s)) >> s);}
                         for(n=0, p=0; p<bits_n_links_per_spin[i]; p++){n += two_pow(p) * ((X_rho[p] & (ullong_1 << s)) >> s);}
                         if(n<m){check = 1;}
                         else{check = 0;}
                         
                         if(((changer >> s) & ullong_1) != check){cout << "\n!!";}
                         //cout << "\n\tchanger[" << s << "] = " << ((changer >> s) & ullong_1) << "\t" << check;
                         }
                         */
                        
                    }
                    
                    /*
                     cout << "\nS[a]_bef : ";
                     for(s=0; s<n_bits; s++){cout << ((S[a][i] >> s) & ullong_1);}
                     */
                    
                    /*Update 64 spins */
                    (S[a][i])^=(~changer);
                    
                    /*
                     cout << "\nS[a]_aft : ";
                     for(s=0; s<n_bits; s++){cout << ((S[a][i] >> s) & ullong_1);}
                     */
                    
                    /*Uncomment this to check acceptance rates*/
                    /*
                     if(rank==0){
                     for(s=0; s<n_bits; s++){acc_swe[a % (nr/3)] += (((~changer) >> s) & ullong_1);}
                     }
                     */
                    
                }
                
            }
            
            /*Write the overlap*/
            if(t % t_wri == 0){
                
                outfile_Q << "\n";
                
                for(a=0; a<nr/3; a++){
                    
                    outfile_Q << "\n";
                    
                    for(f=0; f<3; f++){
                        
                        for(g=f+1; g<3; g++){
                            
                            /*Write Q_L*/
                            for(s=0; s<bits_N_2; s++){X_mea[s] = 0;}
                            for(i=0; i<N_2; i++){
                                a_mea[i] = (~((S[a+f*(nr/3)][i])^(S[a+g*(nr/3)][i])));
                            }
                            
                            /*Sum a_mea and writes the result into {X_mea}_s in base 2*/
                            for(o=0; o<nz_bits_N_2; o++){
                                
                                for(m=0; m<k_mea[o]; m++){
                                    for(n=0; n<tab_mea[o][m]; n++){
                                        
                                        L = add_L_mea[o][m][n];
                                        R = add_R_mea[o][m][n];
                                        
                                        carry = (L[0]) & (R[0]);
                                        (L[0])^=(R[0]);
                                        
                                        for(s=1; s<m+1; s++){
                                            ullong = ((R[s])^(carry));
                                            carry = ((L[s]) & ullong) | ( (R[s]) & (carry) );
                                            (L[s]) ^= ullong;
                                        }
                                        L[m+1] = carry;
                                        
                                    }
                                }
                                
                            }
                            //Sum binary trees
                            for(p=tab_two_pow_k_mea[0], m=1; m<nz_bits_N_2; m++){
                                
                                R=a_mea+p;
                                
                                carry = (a_mea[0]) & (R[0]);
                                (a_mea[0])^=(R[0]);
                                
                                for(s=1; s<k_mea[m]+1; s++){
                                    ullong = ((R[s]) ^ (carry));
                                    carry = ((a_mea[s]) & ullong) | ( (R[s]) & (carry) );
                                    (a_mea[s]) ^= ullong;
                                }
                                for(s=k_mea[m]+1; s<k_mea[0]+1; s++){
                                    ullong = ((a_mea[s]) & (carry));
                                    (a_mea[s]) ^= (carry);
                                    carry = ullong;
                                }
                                a_mea[k_mea[0]+1] = carry;
                                
                                p+=tab_two_pow_k_mea[m];
                                
                            }
                            
                            
                            for(s=0; s<bits_N_2; s++){
                                outfile_Q << a_mea[s] << " ";
                            }
                            
                            outfile_Q << "\t";
                            
                            /*Write Q_R*/
                            for(s=0; s<bits_N_2; s++){X_mea[s] = 0;}
                            for(i=0; i<N_2; i++){
                                a_mea[i] = (~((S[a+f*(nr/3)][N_2+i])^(S[a+g*(nr/3)][N_2+i])));
                            }
                            
                            /*Sum a_mea and writes the result into {X_mea}_s in base 2*/
                            for(o=0; o<nz_bits_N_2; o++){
                                
                                for(m=0; m<k_mea[o]; m++){
                                    for(n=0; n<tab_mea[o][m]; n++){
                                        
                                        L = add_L_mea[o][m][n];
                                        R = add_R_mea[o][m][n];
                                        
                                        carry = (L[0]) & (R[0]);
                                        (L[0])^=(R[0]);
                                        
                                        for(s=1; s<m+1; s++){
                                            ullong = ((R[s])^(carry));
                                            carry = ((L[s]) & ullong) | ( (R[s]) & (carry) );
                                            (L[s]) ^= ullong;
                                        }
                                        L[m+1] = carry;
                                        
                                    }
                                }
                                
                            }
                            //Sum binary trees
                            for(p=tab_two_pow_k_mea[0], m=1; m<nz_bits_N_2; m++){
                                
                                R=a_mea+p;
                                
                                carry = (a_mea[0]) & (R[0]);
                                (a_mea[0])^=(R[0]);
                                
                                for(s=1; s<k_mea[m]+1; s++){
                                    ullong = ((R[s]) ^ (carry));
                                    carry = ((a_mea[s]) & ullong) | ( (R[s]) & (carry) );
                                    (a_mea[s]) ^= ullong;
                                }
                                for(s=k_mea[m]+1; s<k_mea[0]+1; s++){
                                    ullong = ((a_mea[s]) & (carry));
                                    (a_mea[s]) ^= (carry);
                                    carry = ullong;
                                }
                                a_mea[k_mea[0]+1] = carry;
                                
                                p+=tab_two_pow_k_mea[m];
                                
                            }
                            
                            
                            for(s=0; s<bits_N_2; s++){
                                outfile_Q << a_mea[s] << " ";
                            }
                            
                            outfile_Q << "\t\t";
                            
                        }
                        
                    }
                    
                }
                
                flush(outfile_Q);
                
                
                /*save the current spin configurations in case the execution is halted, so I can start back from where I left without wasting computation time. This is interesting only when each core runs one single block of 64samples*/
                
                outfile_S_save.open(filename_S_save.str().c_str());

                for(a=0; a<nr; a++){
                    outfile_S_save << "\n";
                    for(i=0; i<N; i++){
                        outfile_S_save << S[a][i] << " ";
                    }
                }
                
                outfile_S_save << "\n\n";
                flush(outfile_S_save);
                
                outfile_S_save.close();

                
            }
            
   
            /*MC swap*/
            
            if(t % t_swa == 0){
                
                /*Uncomment this to check time
                 if(rank==0){start = clock();}
                 */
                
                for(a=0; a<nr; a++){
                    
                    /*
                     for(s=0; s<bits_n_links; s++){
                     Y[s] = 0;
                     X_swa_b[s] = 0;
                     }
                     */
                    
                    for(l=0, i=0; i<N; i++){
                        for(p=d[i]; p<n_links_per_spin[i]; p++){
                            a_swa[a][l] = (J[i][p])^(S[a][i])^(S[a][ne[i][p]]);
                            l++;
                        }
                    }
                    
                    /*Sommmo a_swa_a[l] e li metto in {X_swa_a[s]}_s in forma binaria, stessa cosa per la replica b.*/
                    /*
                     for(p=1, X_swa_a[0]=a_swa_a[0], X_swa_b[0]=a_swa_b[0]; p<n_links; p++){
                     
                     //Sum for replica a
                     for(s=0; s<bits_n_links; s++){
                     if(s==0){
                     carry_Y[s] = ((X_swa_a[s]) & (a_swa_a[p]));
                     (X_swa_a[s])^= (a_swa_a[p]);
                     }else{
                     carry_Y[s] = ((X_swa_a[s]) & (carry_Y[s-1]));
                     (X_swa_a[s])^= (carry_Y[s-1]);
                     }
                     }
                     
                     //Sum for replica b
                     for(s=0; s<bits_n_links; s++){
                     if(s==0){
                     carry_Y[s] = ((X_swa_b[s]) & (a_swa_b[p]));
                     (X_swa_b[s])^= (a_swa_b[p]);
                     }else{
                     carry_Y[s] = ((X_swa_b[s]) & (carry_Y[s-1]));
                     (X_swa_b[s])^= (carry_Y[s-1]);
                     }
                     }
                     
                     }
                     */
                    /*
                     for(m=0, n=0, p=0; p<n_links; p++){
                     m += ((a_swa_a[p] >> s_test) & ullong_1);
                     n += ((a_swa_b[p] >> s_test) & ullong_1);
                     }
                     cout << "\nA: X_swa_a = " << m << "\tX_swa_b = " << n;
                     */
                    
                    /*Somma per la replica a*/
                    for(o=0; o<nz_bits_n_links; o++){
                        
                        for(m=0; m<k_swa[o]; m++){
                            for(n=0; n<tab_swa[o][m]; n++){
                                
                                L = add_L_swa[a][o][m][n];
                                R = add_R_swa[a][o][m][n];
                                
                                carry = (L[0]) & (R[0]);
                                (L[0])^=(R[0]);
                                
                                for(s=1; s<m+1; s++){
                                    ullong = ((R[s])^(carry));
                                    carry = ((L[s]) & ullong) | ( (R[s]) & (carry) );
                                    (L[s]) ^= ullong;
                                }
                                L[m+1] = carry;
                                
                            }
                        }
                        
                    }
                    //Sum binary trees
                    for(p=tab_two_pow_k_swa[0], m=1; m<nz_bits_n_links; m++){
                        
                        R=a_swa[a]+p;
                        
                        carry = (a_swa[a][0]) & (R[0]);
                        (a_swa[a][0])^=(R[0]);
                        
                        for(s=1; s<k_swa[m]+1; s++){
                            ullong = ((R[s]) ^ (carry));
                            carry = ((a_swa[a][s]) & ullong) | ( (R[s]) & (carry) );
                            (a_swa[a][s]) ^= ullong;
                        }
                        for(s=k_swa[m]+1; s<k_swa[0]+1; s++){
                            ullong = ((a_swa[a][s]) & (carry));
                            (a_swa[a][s]) ^= (carry);
                            carry = ullong;
                        }
                        a_swa[a][k_swa[0]+1] = carry;
                        
                        p+=tab_two_pow_k_swa[m];
                        
                    }
                    
                    
                    /*
                     for(m=0, n=0, s=0; s<bits_n_links; s++){
                     m+= two_pow(s)*((a_swa_a[s] >> s_test) & ullong_1);
                     n+= two_pow(s)*((a_swa_b[s] >> s_test) & ullong_1);
                     }
                     cout << "\nB: X_swa_a = " << m << "\tX_swa_b = " << n;
                     */
                    
                }
                
                
                
                for(c=0; c<nr-3; c++){
                    
                    /*I pick two contigous replicas in the first group of nr/3 replicas and in the second and third group of nr/3 replicas. */
                    if(c<nr/3-1){
                        b=c;
                        a=c+1;
                    }else{
                        if(c < 2*(nr/3)-2){
                            b=c+1;
                            a=c+2;
                        }else{
                            b=c+2;
                            a=c+3;
                        }
                    }
                    
                    tau = (unsigned long long int)(gsl_ran_exponential(myran_S,1.0)/(2.0*(beta[a]-beta[b])));
                    
                    //      int tau_test = tau;
                    
                    /*
                     unsigned long long int tau_test, *X_swa_b_test, *sum;
                     X_swa_b_test = new unsigned long long int [n_bits];
                     sum = new unsigned long long int [n_bits];
                     
                     tau_test = tau;
                     for(s=0; s<n_bits; s++){
                     for(X_swa_b_test[s]=0, p=0; p<bits_n_links; p++){X_swa_b_test[s] += two_pow(p) * ((a_swa_b[p] & (ullong_1 << s)) >> s);}
                     }
                     //cout << "\n\ntau_{10} = " << o;
                     //cout << "\ta_swa_b_{2} = " << m;
                     */
                    
                    /*Converto tau in binario mettendolo in X_tau[]*/
                    for(s=0; s<n_bits; s++){
                        X_tau[s] = (tau % 2);
                        tau -= X_tau[s];
                        tau=(int)(((double)tau)/2.0);
                        
                        r = X_tau[s];
                        X_tau[s] = 0;
                        for(p=0; p<n_bits; p++){(X_tau[s]) |= (r << p);}
                    }
                    
                    
                    //for(n=0, s=0; s<n_bits; s++){n += two_pow(s) * ((X_tau[s] & (ullong_1 << s_test)) >> s_test);}
                    //cout << "\ttau{2} = " << n;
                    
                    
                    /*Sommo a X_tau[] a_swa_b[] e scrivo il risultato in X_tau[]*/
                    /*
                     for(s=0; s<bits_n_links; s++){
                     carry_tau[s] = ((X_tau[s]) & (a_swa_b[s]));
                     (X_tau[s])^=(a_swa_b[s]);
                     
                     for(p=s+1; p<n_bits; p++){
                     carry_tau[p] = ((X_tau[p]) & (carry_tau[p-1]));
                     (X_tau[p])^=(carry_tau[p-1]);
                     }
                     }
                     */
                    carry = (X_tau[0]) & (a_swa[b][0]);
                    (X_tau[0])^=(a_swa[b][0]);
                    
                    for(s=1; s<bits_n_links; s++){
                        ullong = ((X_tau[s]) & ((a_swa[b][s])^(carry))) | ( (a_swa[b][s]) & (carry) );
                        (X_tau[s]) ^= ((a_swa[b][s])^(carry));
                        carry = ullong;
                    }
                    for(s=bits_n_links; (s<n_bits) && (carry!=all_0); s++){
                        ullong = (X_tau[s]) & (carry) ;
                        (X_tau[s]) ^= (carry);
                        carry = ullong;
                    }
                    
                    /*
                     for(s=0; s<n_bits; s++){
                     for(sum[s]=0, p=0; p<n_bits; p++){sum[s] += two_pow(p) * ((X_tau[p] & (ullong_1 << s)) >> s);}
                     }
                     for(s=0; s<n_bits; s++){
                     //cout << "\n" << tau_test << "\t" << X_swa_b_test[s] << "\t" << sum[s];
                     if(tau_test + X_swa_b_test[s] != sum[s]){cout << "\n!!";}
                     }
                     */
                    
                    
                    /*Confronto X_tau[] con a_swa[a][] e scrivo in changer_swa il risultato. changer_swa è uguale a 1 se X_tau < a_swa[a] e a 0 altrimenti.*/
                    //Here  I assume that bits_of_n_links < n_bits.
                    
                    changer_swa = all_0;
                    check = X_tau[n_bits - 1];
                    
                    for(s=n_bits-2; s >=bits_n_links; s--){
                        changer_swa = (check & changer_swa);
                        check = (check | (X_tau[s]));
                    }
                    for(s=bits_n_links-1; s >=0; s--){
                        changer_swa = ((check & changer_swa) | ((~check) & ((~(X_tau[s])) & (a_swa[a][s]))) );
                        check = (check | ((X_tau[s]) ^ (a_swa[a][s])));
                    }
                    
                    /*
                     int m, n;
                     for(s=0; s<n_bits; s++){
                     for(m=0, p=0; p<bits_n_links; p++){m += two_pow(p) * ((a_swa_a[p] >> s) & ullong_1);}
                     for(n=0, p=0; p<bits_n_links; p++){n += two_pow(p) * ((a_swa_b[p] >> s) & ullong_1);}
                     
                     if(tau_test < m - n){check = 1;}
                     else{check = 0;}
                     // cout << "\n" << check << ((changer_swa >> s) & ullong_1);
                     if(check != ((changer_swa >> s) & ullong_1)){cout << "\n!!"; flush(cout);}
                     }
                     */
                    
                    /*Scambio la replica a con la replica b se changer_swa = 0, non la scambio se changer = 1*/
                    for(i=0; i<N; i++){
                        Sa_te = S[a][i];
                        Sb_te = S[b][i];
                        
                        S[a][i] = ( (Sa_te & Sb_te) | (Sa_te & changer_swa) | (Sb_te & (~changer_swa)) );
                        S[b][i] = ( (Sa_te & Sb_te) | (Sa_te & (~changer_swa)) | (Sb_te & changer_swa) );
                    }
                    for(p=0; p<bits_n_links; p++){
                        a_swa_a_te = a_swa[a][p];
                        a_swa_b_te = a_swa[b][p];
                        
                        a_swa[a][p] = ( (a_swa_a_te & a_swa_b_te) | (a_swa_a_te & changer_swa) | (a_swa_b_te & (~changer_swa)) );
                        a_swa[b][p] = ( (a_swa_a_te & a_swa_b_te) | (a_swa_a_te & (~changer_swa)) | (a_swa_b_te & changer_swa) );
                    }
                    
                    /*Uncomment this to check acceptance rates*/
                    //
                    if(rank==0){
                        for(s=0; s<n_bits; s++){
                            
                            if(c<nr/3-1){
                                acc_swa[c] += (((~changer_swa) >> s) & ullong_1);
                            }else{
                                if(c<2*(nr/3)-2){
                                    acc_swa[c+1-(nr/3)] += (((~changer_swa) >> s) & ullong_1);
                                }else{
                                    acc_swa[c+2-2*(nr/3)] += (((~changer_swa) >> s) & ullong_1);
                                }
                            }
                            
                        }
                    }
                    //
                    
                }
                
                /*Uncomment this to check time*/
                /*
                 if(rank==0){
                 stop = clock();
                 time_cl += stop - start;
                 }
                 
                 // cout << "\nTime for swap = " << stop - start;
                 // flush(cout);
                 */
                
            }
            
        }
        
        /*Write the final spin configurations*/
        for(a=0; a<nr; a++){
            outfile_S << "\n";
            for(i=0; i<N; i++){
                outfile_S << S[a][i] << " ";
            }
        }
        outfile_S << "\n\n";
        flush(outfile_S);
        
        
        
        /*Test with MC with no multispin coding*/
        /*
         int *S_nms, dE, E, **U_nms;
         S_nms = new int [N];
         U_nms = new int* [nr];
         for(a=0; a<nr; a++){U_nms[a] = new int [n_bits];}
         
         for(s=0; s<n_bits; s++){
         
         for(i=0; i<N; i++){S_nms[i] = 1-2*gsl_rng_uniform_int(myran_S,2);}
         for(a=0; a<nr; a++){U_nms[a][s] = 0;}
         for(a=0; a<nr; a++){
         
         for(t=0; t<t_tot; t++){
         for(i=0; i<N; i++){
         for(dE=0, j=0; j<n_links_per_spin[i]; j++){
         dE += (-1 + 2 *(double)((J[i][j] >> s) & ullong_1)) * S_nms[i] * S_nms[ne[i][j]];
         }
         dE*=2;
         if(dE < 0 || gsl_rng_uniform(myran_S) <= exp(-beta[a]*((double)dE))){S_nms[i]*=-1;}
         }
         
         if((t % t_mea == 0) && (t > t_tot/2-1)){
         
         for(E=0, i=0; i<N; i++){
         for(j=d[i]; j<n_links_per_spin[i]; j++){
         E += - (-1.0 + 2.0 *(double)((J[i][j] >> s) & ullong_1)) * (S_nms[i]) * (S_nms[ne[i][j]]);
         }
         }
         
         U_nms[a][s]+=E;
         }
         }
         
         }
         
         }
         
         for(s=0; s<n_bits; s++){
         cout << "\n\ns = " << s;
         for(a=0; a<nr; a++){
         cout << "\n" << a << "\t" << U[a][s]/((double)(N*t_tot/2/t_mea)) << "\t" << ((double)U_nms[a][s])/((double)(N*t_tot/2/t_mea));
         }
         }
         
         delete [] S_nms;
         for(a=0; a<nr; a++){delete [] U_nms[a];}
         delete [] U_nms;
         */
        
        if(rank==0){
            stop = clock();
            time_cl = stop - start;
            cout << "\nTime for MC = " << ((double)time_cl)/((double)CLOCKS_PER_SEC) << " s";
            flush(cout);
        }
        
        
        for(i=0; i<N; i++){
            free(J[i]);
            free(ne[i]);
        }
        for(a=0; a<nr; a++){
            delete [] a_swa[a];
        }
        /*
         delete [] Y;
         delete [] carry_Y;
         */
        delete [] n_links_2;
        for(s=0; s<nz_bits_n_links; s++){
            delete [] tab_swa[s];
        }
        delete [] tab_swa;
        delete [] tab_two_pow_k_swa;
        for(a=0; a<nr; a++){
            for(s=0; s<nz_bits_n_links; s++){
                for(m=0; m<k_swa[s]; m++){
                    delete [] add_L_swa[a][s][m];
                    delete [] add_R_swa[a][s][m];
                }
                delete [] add_L_swa[a][s];
                delete [] add_R_swa[a][s];
            }
            delete [] add_L_swa[a];
            delete [] add_R_swa[a];
        }
        delete [] k_swa;
    }
    
    
    
    
    
    /*Uncomment this to check time / acceptance rates*/
    //
    if(rank==0){
        //cout << "\nTotal time for swap = " << time_cl;
        
        // cout << "\nSweep rates: ";
        // for(a=0; a<nr/3; a++){
        //   cout << "\na=" << a << "\t" << ((double)(acc_swe[a]))/(double)(3*N*t_tot*n_bits*n_64samples/size);
        // }
        cout << "\nSwap rates: ";
        for(a=0; a<nr/3-1; a++){
            cout << "\na=" << a << "\t" << ((double)(acc_swa[a]))/(double)(3*t_tot/t_swa*n_bits*n_64samples/size);
        }
        flush(cout);
    }
    //
    
    
    
    /*Test with exact computation.*/
    /*
     double **H_test, E, Z;
     vector<int> SS(1);
     H_test = new double* [nr];
     for(a=0; a<nr; a++){
     H_test[a] = new double [n_bits];
     }
     SS[0] = 1;
     generate(N, SS);
     
     // for(p=0; p<two_pow(N-1); p++){
     // cout << "\n";
     // for(i=0; i<N; i++){cout << " " << S_test[p][i];}
     // }
     
     
     for(a=0; a<nr; a++){
     
     for(s=0; s<n_bits; s++){
     
     H_test[a][s] = 0.0;
     Z = 0.0;
     
     for(p=0; p<two_pow(N-1); p++){
     
     for(E=0.0, i=0; i<N; i++){
     for(j=d[i]; j<n_links_per_spin[i]; j++){
     E += - (-1.0 + 2.0 *(double)((J[i][j] >> s) & ullong_1)) * (S_test[p][i]) * (S_test[p][ne[i][j]]);
     
     // cout << "\n";
     // for(l=0; l<n_bits; l++){cout << " " << ((J[i][j] >> l) & ullong_1);}
     }
     }
     
     Z+= exp(-beta[a]*E);
     H_test[a][s]+= exp(-beta[a]*E) * E;
     
     }
     
     H_test[a][s]/=Z;
     
     }
     
     }
     
     // for(s=0; s<n_bits; s++){
     s=s_test;
     cout << "\n\ns = " << s;
     for(a=0; a<nr; a++){
     cout << "\n" << a << "\t" << H_test[a][s]/((double)N) << "\t" << U[a][s]/((double)(N*t_tot/2/t_mea));
     }
     // }
     
     
     for(a=0; a<nr; a++){delete [] H_test[a];}
     delete [] H_test;
     */
    
    
    /*Close files*/
    outfile_Q.close();
    outfile_S.close();
    /*Read S from file*/
    /*
     infile_S.close();
     */
    
    
    /*Free all*/
    gsl_rng_free(myran_S);
    gsl_rng_free(myran_J);
    for(a=0; a<nr; a++){delete [] S[a];}
    delete [] S;
    delete [] J;
    delete [] i_J;
    delete [] I;
    delete [] ne;
    delete [] X_swe;
    delete [] X_mea;
    delete [] X_rho;
    delete [] X_tau;
    delete [] a_swe;
    delete [] a_mea;
    delete [] a_swa;
    delete [] carry_X_swe;
    delete [] carry_X_mea;
    delete [] n_links_per_spin;
    delete [] bits_n_links_per_spin;
    delete [] bits_p_p_1;
    delete [] d;
    delete [] beta;
    /*Uncomment this to check acceptance rates*/
    
    if(rank==0){
        //delete [] acc_swe;
        delete [] acc_swa;
    }
    
    delete [] add_L_swa;
    delete [] add_R_swa;
    
    delete [] N_2_2;
    for(s=0; s<nz_bits_N_2; s++){
        delete [] tab_mea[s];
    }
    delete [] tab_mea;
    delete [] tab_two_pow_k_mea;
    for(s=0; s<nz_bits_N_2; s++){
        for(m=0; m<k_mea[s]; m++){
            delete [] add_L_mea[s][m];
            delete [] add_R_mea[s][m];
        }
        delete [] add_L_mea[s];
        delete [] add_R_mea[s];
    }
    delete [] add_L_mea;
    delete [] add_R_mea;
    delete [] k_mea;
    
    gsl_root_fsolver_free(solver);
    
    /*Change this to parallelize the code*/
    MPI_Finalize();
    
    cout << "\n";
    return(0);
    
}
