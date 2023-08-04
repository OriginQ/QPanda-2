#pragma once

#include "Core/Utilities/Benchmark/BenchmarkingGate.h"
#include "Core/Utilities/Benchmark/CrossEntropyBenchmarking.h"

QPANDA_BEGIN

#if defined(USE_CURL)

/**
* @class RandomizedBenchmarking
* @ingroup Utilities
* @brief
*/
class RandomizedBenchmarking
{
public:
    using CliffordsSeq = std::vector<std::vector<std::shared_ptr<BenchmarkSingleGate>>>;
    struct Cliffords
    {
        CliffordsSeq c1_in_xy;
        CliffordsSeq c1_in_xz;
        CliffordsSeq s1;
        CliffordsSeq s1_x;
        CliffordsSeq s1_y;
    };

    RandomizedBenchmarking(QuantumMachine* machine);
    RandomizedBenchmarking(QCloudTaskConfig config);
    ~RandomizedBenchmarking();

    std::map<int, double>single_qubit_rb(Qubit* qbit, 
        const std::vector<int>& clifford_range, 
        int num_circuits, 
        int shots,
        RealChipType chip_type = RealChipType::ORIGIN_WUYUAN_D5,
        const std::vector<QGate>& interleaved_gates = {});

    std::map<int, double> single_qubit_rb(int qbit,
        const std::vector<int>& clifford_range,
        int num_circuits,
        const std::vector<QGate> &interleaved_gates);

    std::map<int, double> two_qubit_rb(Qubit* qbit0, 
        Qubit* qbit1, 
        const std::vector<int>& clifford_range, 
        int num_circuits, 
        int shots,
        RealChipType chip_type = RealChipType::ORIGIN_WUYUAN_D5,
        const std::vector<QGate>& interleaved_gates = {});

    std::map<int, double> two_qubit_rb(int qbit0,
        int qbit1,
        const std::vector<int>& clifford_range,
        int num_circuits,
        const std::vector<QGate>& interleaved_gates = {});

private:
    Cliffords _single_qubit_cliffords();
    QCircuit _random_single_q_clifford(Qubit* qbit, int num_cfds, const CliffordsSeq& cfd, const std::vector<QStat>& cfd_matrices, const std::vector<QGate>& interleaved_gates);

    QCircuit _two_qubit_clifford_starters(Qubit* q_0, Qubit* q_1, int idx_0, int idx_1, const Cliffords& cfds);
    QCircuit _two_qubit_clifford_mixers(Qubit* q_0, Qubit* q_1, int idx_2, const Cliffords& cfds);
    std::vector<int >_split_two_q_clifford_idx(int idx);
    QCircuit _random_two_q_clifford(Qubit* q_0, Qubit* q_1, int num_cfds, const Cliffords& cfds, const std::vector<QStat>& cfd_matrices, const std::vector<QGate>& interleaved_gates);
    std::vector <QStat>_two_qubit_clifford_matrices(Qubit* q_0, Qubit* q_1, const Cliffords& cfds);

private:

    QCloudMachine m_qcloud;
    QCloudTaskConfig m_cloud_config;

    QMachineType m_machine_type;
    QuantumMachine* m_machine_ptr;
};

/*
* @brief get single qubit fidelity by fitting rb_result with y = a\times b^x + c
* @param[in] rb_result
* @return double fidelity
*/
double calc_single_qubit_fidelity(const std::map<int, double>& rb_result);

/**
 * @brief  single gate rb experiment
 * @param[in] QCloudMachine*  cloud quantum machine
 * @param[in] Qubit* qubit
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] const std::vector<QGate>  interleaved gates
 * @return std::map<int, double>  rb result of each layer
 */
std::map<int, double> single_qubit_rb(QuantumMachine* machine,
    Qubit* qbit, 
    const std::vector<int>& clifford_range, 
    int num_circuits, 
    int shots,
    RealChipType chip_type = RealChipType::ORIGIN_WUYUAN_D5,
    const std::vector<QGate>& interleaved_gates = {});


/**
 * @brief  double gate rb experiment
 * @param[in] QCloudMachine*  cloud quantum machine
 * @param[in] Qubit* qubit0
 * @param[in] Qubit* qubit1
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] const std::vector<QGate>  interleaved gates
 * @return std::map<int, double>  rb result of each layer
 */
std::map<int, double> double_qubit_rb(QuantumMachine* machine,
    Qubit* qbit0,
    Qubit* qbit1,
    const std::vector<int>& clifford_range, 
    int num_circuits, 
    int shots, 
    RealChipType chip_type = RealChipType::ORIGIN_WUYUAN_D5,
    const std::vector<QGate>& interleaved_gates = {});


std::map<int, double> single_qubit_rb(QCloudTaskConfig config,
    int qbit,
    const std::vector<int>& clifford_range,
    int num_circuits,
    const std::vector<QGate>& interleaved_gates = {});

std::map<int, double> double_qubit_rb(QCloudTaskConfig config,
    int qbit0,
    int qbit1,
    const std::vector<int>& clifford_range,
    int num_circuits,
    const std::vector<QGate>& interleaved_gates = {});

#endif

QPANDA_END