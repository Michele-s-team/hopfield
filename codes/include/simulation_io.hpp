#ifndef simulation_io_hpp
#define simulation_io_hpp

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

using namespace std;

class SimulationIO {
public:

    static constexpr int BITS_PER_BLOCK = 64;
    static constexpr int BLOCK_MASK = BITS_PER_BLOCK - 1; // 63
    SimulationIO() = default;
    ~SimulationIO();

    void OpenMagnetizationFiles(const string& folder, int N, double beta);
    void CloseMagnetizationFiles();

    void OpenSpinFiles(const string& folder, int N, double beta);
    void CloseSpinFiles();

    void SaveMagnetizations(
        int N,
        const vector<double>& magnetizations);

    void SaveSpinConfigurations(
        int N,
        int mc_step,
        const vector<int>& spins_set);

    void SavePatterns(
        const string& folder,
        int N,
        const vector<vector<vector<int>>>& patterns);

private:

    static inline int num_blocks(int N);
    vector<ofstream> m_magnetization_files;
    vector<ofstream> m_spin_files;
    
};

#endif