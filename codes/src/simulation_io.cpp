//
//  simulation_io.cpp
//  hopfield
//
//  Created by Bastien on 02/06/2026.
//
//


#include "simulation_io.hpp"
#include <fstream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "main.hpp"
using namespace std;

SimulationIO::~SimulationIO(){
    CloseMagnetizationFiles();
    CloseSpinFiles();
}

static inline string format_beta(double beta){
    long long scaled = static_cast<long long>(round(beta * 1e6));
    return to_string(scaled);
}

inline int SimulationIO::num_blocks(int N){
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

void SimulationIO::OpenMagnetizationFiles(const string& folder, int N, double beta){
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

void SimulationIO::OpenSpinFiles(const string& folder, int N, double beta){
    const int n_blocks = num_blocks(N);
    m_spin_files.resize(n_bits);
    for (int r = 0; r < n_bits; ++r){
        string path =
            folder +
            "_spins_N" + to_string(N) +
            "_beta" + format_beta(beta) +
            "_r" + to_string(r) +
            ".csv";
        m_spin_files[r].open(path, ios::app);

        if (m_spin_files[r].tellp() == 0){
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

void SimulationIO::SaveSpinConfigurations(int N, int sweep, const vector<int>& spins_set){
    const int n_blocks = num_blocks(N);

    for (int r = 0; r < n_bits; ++r){
        if (!m_spin_files[r])
            continue;

        m_spin_files[r] << sweep;

        const int* spins = &spins_set[r * N];

        for (int b = 0; b < n_blocks; ++b){
            int start = b * BITS_PER_BLOCK;
            int end   = min(N, start + BITS_PER_BLOCK);

            uint64_t config = PackBlock(spins, start, end);

            m_spin_files[r] << "," << config;
        }

        m_spin_files[r] << "\n";
    }
}

void SimulationIO::SavePatterns(const string& folder, int N, const vector<vector<vector<int>>>& patterns){
    const int P = patterns.size();
    const int n_blocks = num_blocks(N);
    for (int r = 0; r < n_bits; ++r){
        string path =
            folder +
            "_patterns_N" + to_string(N) +
            "_r" + to_string(r) +
            ".csv";
        ofstream file(path);
        if (!file)
            continue;
        file << "p";
        for (int b = 0; b < n_blocks; ++b)
            file << ",block" << b;
        file << "\n";
        for (int p = 0; p < P; ++p){
            file << p;

            for (int b = 0; b < n_blocks; ++b){
                int start = b * BITS_PER_BLOCK;
                int end   = min(N, start + BITS_PER_BLOCK);

                uint64_t pattern =
                    PackBlock(patterns[p].data()->data() /* impossible ici */ ,
                            start,
                            end);
            }
        }
    }
}