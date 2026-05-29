#include "simulation_io.hpp"
#include <fstream>
#include <string>
using namespace std;

#include "main.hpp"


void SimulationIO::OpenCSVFiles(const string& folder, int N) {
    m_csv_files.resize(n_bits);

    for (int r = 0; r < n_bits; ++r) {
        string path = folder
            + "_N" + to_string(N)
            + "_r" + to_string(r)
            + ".csv";

        m_csv_files[r].open(path, ios::app);

        if (m_csv_files[r].tellp() == 0)
            m_csv_files[r] << "T,N,m\n";
    }
}

void SimulationIO::CloseCSVFiles() {
    for (auto& f : m_csv_files)
        if (f.is_open()) f.close();
    m_csv_files.clear();
    //cout <<"files closed"<<endl;
}

void SimulationIO::SaveMagnetizations(int N, double B, // either betaJ when Ising or just beta for Sin glass/Hofield
                                      const vector<double>& magnetizations) {
    double T = 1.0 / B;
    for (int r = 0; r < n_bits; ++r) {
        if (!m_csv_files[r]) continue;
        m_csv_files[r] << T << "," << N << "," << magnetizations[r] << "\n";
    }
}