#include "simulation_io.hpp"
#include <fstream>
#include <string>
using namespace std;

#include "main.hpp"


// Open one CSV file per (L, r), header contains beta (or betaJ), N, m
void SimulationIO::OpenCSVFiles(const string& folder, int L) {
    m_L      = L;
    m_n_bits = n_bits;
    m_csv_files.resize(n_bits);

    for (int r = 0; r < n_bits; ++r) {
        string path = folder
            + "/L" + to_string(L)
            + "_r" + to_string(r)
            + ".csv";

        ifstream test(path);
        bool exists = test.good();
        test.close();

        m_csv_files[r].open(path, ios::app);
        if (m_csv_files[r] && !exists)
            m_csv_files[r] << "T,N,m\n";
    }
}

void SimulationIO::CloseCSVFiles() {
    for (auto& f : m_csv_files)
        if (f.is_open()) f.close();
    m_csv_files.clear();
}

void SimulationIO::SaveMagnetizations(int N, double B, // either betaJ when Ising or just beta for Sin glass/Hofield
                                      const vector<double>& magnetizations) {
    double T = 1.0 / B;
    for (int r = 0; r < m_n_bits; ++r) {
        if (!m_csv_files[r]) continue;
        m_csv_files[r] << T << "," << N << "," << magnetizations[r] << "\n";
    }
}