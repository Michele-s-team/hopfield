#ifndef simulation_io_hpp
#define simulation_io_hpp

#include <string>
#include <vector>
#include <fstream>
#include "main.hpp"
using namespace std;

class SimulationIO {
public:
    SimulationIO() = default;
    ~SimulationIO() { CloseCSVFiles(); }

    void OpenCSVFiles(const string& folder, int N);
    void CloseCSVFiles();
    void SaveMagnetizations(int N, double beta, // either betaJ when Ising or just beta for Sin glass/Hofield
                            const vector<double>& magnetizations);

private:
    vector<ofstream> m_csv_files;
    int m_L     = 0;
    int m_n_bits = 0;
};

#endif