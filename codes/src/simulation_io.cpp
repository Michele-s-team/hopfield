#include "simulation_io.hpp"

#include <fstream>
#include <string>
#include <stdexcept>
#include <cmath>

#include "main.hpp"

using namespace std;

SimulationIO::~SimulationIO(){
    CloseMagnetizationFiles();
    CloseSpinFiles();
}

static inline string format_beta(double beta)
{
    long long scaled = static_cast<long long>(round(beta * 1e6));
    return "beta" + to_string(scaled);
}

static inline int num_blocks(int N)
{
    return (N + SimulationIO::BLOCK_MASK) / SimulationIO::BITS_PER_BLOCK;
}

void SimulationIO::OpenMagnetizationFiles(
    const string& folder,
    int N,
    double beta)
{
    m_magnetization_files.resize(n_bits);

    for (int r = 0; r < n_bits; ++r)
    {
        string path =
            folder +
            "_N" + to_string(N) +
            "_beta" + format_beta(beta) +
            "_r" + to_string(r) +
            ".csv";

        m_magnetization_files[r].open(path, ios::app);

        if (m_magnetization_files[r].tellp() == 0)
            m_magnetization_files[r] << "N,m\n";
    }
}

void SimulationIO::OpenSpinFiles(
    const string& folder,
    int N,
    double beta)
{
    m_spin_files.resize(n_bits);

    for (int r = 0; r < n_bits; ++r)
    {
        string path =
            folder +
            "_spins_N" + to_string(N) +
            "_beta" + format_beta(beta) +
            "_r" + to_string(r) +
            ".csv";

        m_spin_files[r].open(path, ios::app);

        if (m_spin_files[r].tellp() == 0)
            m_spin_files[r] << "mc_step,block,config\n";
    }
}

void SimulationIO::CloseSpinFiles()
{
    for (auto& f : m_spin_files){
        if (f.is_open())
            f.close();
    }

    m_spin_files.clear();
}

void SimulationIO::SaveMagnetizations(
    int N,
    const vector<double>& magnetizations)
{
    for (int r = 0; r < n_bits; ++r)
    {
        if (!m_magnetization_files[r])
            continue;

        m_magnetization_files[r]
            << magnetizations[r]
            << "\n";
    }
}

void SimulationIO::SaveSpinConfigurations(
    int N,
    int mc_step,
    const vector<int>& spins_set)
{
    const int n_blocks = num_blocks(N);

    for (int r = 0; r < n_bits; ++r)
    {
        if (!m_spin_files[r])
            continue;

        m_spin_files[r] << mc_step;

        for (int b = 0; b < n_blocks; ++b)
        {
            uint64_t config = 0;

            int start = b * BITS_PER_BLOCK;
            int end = min(N, start + BITS_PER_BLOCK);

            for (int i = start; i < end; ++i)
            {
                config <<= 1;

                if (spins_set[i * n_bits + r] > 0)
                    config |= 1ULL;
            }

            m_spin_files[r] << "," << config;
        }

        m_spin_files[r] << "\n";
    }
}


void SimulationIO::SavePatterns(
    const string& folder,
    int N,
    const vector<vector<vector<int>>>& patterns)
{
    const int P = patterns.size();
    const int n_blocks = num_blocks(N);

    for (int r = 0; r < n_bits; ++r)
    {
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

        for (int p = 0; p < P; ++p)
        {
            file << p;

            for (int b = 0; b < n_blocks; ++b)
            {
                uint64_t pattern = 0;

                int start = b * BITS_PER_BLOCK;
                int end = min(N, start + BITS_PER_BLOCK);

                for (int i = start; i < end; ++i)
                {
                    pattern <<= 1;

                    if (patterns[p][i][r] > 0)
                        pattern |= 1ULL;
                }

                file << "," << pattern;
            }

            file << "\n";
        }
    }
}