//
//  simulation_io.hpp
//  hopfield
//
//  Created by Bastien on 02/06/2026.
//
//


#ifndef simulation_io_hpp
#define simulation_io_hpp

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

using namespace std;

class SimulationIO {
public:
   
string m_base_folder;

    SimulationIO() = default;
    ~SimulationIO();

    int SpinToBit(int s);

    void OpenMagnetizationFiles(const string& folder, int N, double beta);
    void CloseMagnetizationFiles();

    void OpenSpinFiles(const string& folder, int N, double beta);
    void CloseSpinFiles();

    void SaveMagnetizations(const int sweep, const vector<double>& magnetizations);
    void SaveSpinConfigurations(int sweep, const vector<vector<uint64_t>>& configs);
    void SavePatterns(const string& folder,const vector<vector<vector<int>>>& patterns, int N);

private:
    static inline int num_blocks(int N);
    static uint64_t PackBlock(const int* data, int start, int end);

    vector<ofstream> m_magnetization_files;
    vector<ofstream> m_spin_files;
    
};

#endif