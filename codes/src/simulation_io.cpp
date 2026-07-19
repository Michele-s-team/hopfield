//
//  simulation_io.cpp
//  hopfield
//
//  Created by Bastien on 02/06/2026.
//
//


#include "simulation_io.hpp"
#include "main.hpp"
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <algorithm>


using namespace std;

SimulationIO::~SimulationIO(){
    CloseMagnetizationFiles();
    CloseSpinFiles();
}

int SimulationIO::SpinToBit(int s){
    return (s > 0) ? 1 : 0;
}


int SimulationIO::num_blocks(int N){
    return (N + BLOCK_MASK) / BITS_PER_BLOCK;
}

uint64_t SimulationIO::PackBlock(const int* data, int start, int end){
    uint64_t packed = 0;
    for (int i = start; i < end; ++i){
        packed <<= 1;
        if (data[i] > 0)
            packed |= 1ULL;
    }
    return packed;
}


static inline string format_beta(double beta){
    ostringstream oss;
    oss << fixed << setprecision(6) << beta;
    return oss.str();
}


void SimulationIO::OpenMagnetizationFiles(const string& folder, int N, double beta){

    namespace fs = filesystem;
    fs::create_directories(folder);

    m_magnetization_files.resize(n_bits);

    for (int r = 0; r < n_bits; ++r){
        string path =
            folder +
            "_N" + to_string(N) +
            "_beta" + format_beta(beta) +
            "_r" + to_string(r) +
            ".csv";
        m_magnetization_files[r].open(path, ios::app);
        if (m_magnetization_files[r].tellp() == 0)
            m_magnetization_files[r] << "sweep,m\n";
    }
}

void SimulationIO::OpenSpinFiles(
    const std::string& folder,
    int N,
    double beta)
{
    namespace fs = std::filesystem;
    fs::path base_folder(folder);
    fs::path spins_folder = base_folder / "spins";
    fs::create_directories(spins_folder);

    const int n_blocks = num_blocks(N);
    m_spin_files.resize(n_bits);
    for (int r = 0; r < n_bits; ++r){
        fs::path path = spins_folder / ("spins_r" + std::to_string(r) + ".csv");
        m_spin_files[r].open(path, std::ios::out | std::ios::app);
        if (!m_spin_files[r].is_open())
            continue;
        if (m_spin_files[r].tellp() == 0)
        {
            m_spin_files[r] << "sweep";
            for (int b = 0; b < n_blocks; ++b)
                m_spin_files[r] << ",block" << b;
            m_spin_files[r] << "\n";
        }
    }
}

void SimulationIO::CloseSpinFiles(){
    for (auto& f : m_spin_files){
        if (f.is_open())
            f.close();
    }
    m_spin_files.clear();
}

void SimulationIO::CloseMagnetizationFiles(){
    for (auto& f : m_magnetization_files){
        if (f.is_open())
            f.close();
    }
    m_magnetization_files.clear();
}

void SimulationIO::SaveMagnetizations(const int sweep, const vector<double>& magnetizations){
    for (int r = 0; r < n_bits; ++r){
        if (!m_magnetization_files[r])
            continue;
        m_magnetization_files[r]
            << sweep
            << ","
            << magnetizations[r]
            << "\n";
    }
}

void SimulationIO::SaveSpinConfigurations(int sweep, const vector<vector<uint64_t>>& configs){
    for (int r = 0; r < n_bits; ++r){
        if (!m_spin_files[r])
            continue;

        m_spin_files[r] << sweep;
        for (uint64_t config : configs[r])
            m_spin_files[r] << "," << config;
        m_spin_files[r] << "\n";
    }
}

void SimulationIO::SavePatterns(
    const std::string& folder,
    const std::vector<int>& patterns,
    int N, int P){
    namespace fs = std::filesystem;
    fs::path base_folder(folder);
    fs::path full_folder = base_folder / "patterns";
    fs::create_directories(full_folder);

    const int n_blocks = num_blocks(N);
    std::vector<int> binary(N);

    for (int r = 0; r < n_bits; ++r) {

        fs::path path =
            full_folder /
            ("patterns_r" + std::to_string(r) + ".csv");
        std::ofstream file(path);
        if (!file)
            continue;

        file << "mu";
        for (int b = 0; b < n_blocks; ++b)
            file << ",block" << b;
        file << "\n";

        for (int mu = 0; mu < P; ++mu){
            file << mu;
            for (int i = 0; i < N; ++i)
                binary[i] = SpinToBit(patterns[(i * P + mu) * n_bits + r]);

            for (int b = 0; b < n_blocks; ++b) {
                int start = b * BITS_PER_BLOCK;
                int end   = std::min(N, start + BITS_PER_BLOCK);
                uint64_t config = PackBlock(binary.data(), start, end);
                file << "," << config;
            }
            file << "\n";
        }
    }
}

void SimulationIO::check_equality_configs(const vector<int>& neurons_before,
                   const vector<int>& neurons_nobits,
                   const vector<int>& neurons_bits,
                   int N_neurons, int prefix_width, int col_width) {

    bool all_equal = true;

    for (int r = 0; r < n_bits; r++) {

        ostringstream oss;
        //Uncomment to print the spins when comparing
        /*
        oss << "Realization" << right << setw(3) << r+1;
        cout << oss.str() << "\n";

        cout << left << setw(prefix_width) << "       before: ";
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_before[r*N_neurons+i] << " ";
        cout << "\n";

        cout << left << setw(prefix_width) << "after classic: ";
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_nobits[r*N_neurons+i] << " ";
        cout << "\n";

        cout << left << setw(prefix_width) << "   after bits: ";
        for (int i = 0; i < N_neurons; i++)
            cout << right << setw(col_width) << neurons_bits[r*N_neurons+i] << " ";
        cout << "\n\n";
        */

        for (int i = 0; i < N_neurons; i++) {
            if (neurons_nobits[r*N_neurons+i] != neurons_bits[r*N_neurons+i]) {
                all_equal = false;
                cout << "Mismatch at r=" << r<< " i=" << i << " : bits="<< neurons_bits[r*N_neurons+i] << "  nobits =" <<neurons_nobits[r*N_neurons+i] << endl;
            }
        }
    }

    if (all_equal)
        cout << "OK: classic and bitwise configurations are identical\n";
    else
        cout << "WARNING: differences detected between classic and bitwise configurations\n" << endl;
}