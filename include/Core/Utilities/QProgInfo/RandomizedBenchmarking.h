#ifndef _RANDOMIZED_BENCHMARKING_H
#define _RANDOMIZED_BENCHMARKING_H



#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/QProgInfo/CrossEntropyBenchmarking.h"


QPANDA_BEGIN


class RBGate
{
public:
    virtual QGate qgate(Qubit* qbit) = 0;
    virtual QStat unitary() = 0;
    virtual ~RBGate() {}
};

/**
* @class RandomizedBenchmarking
* @ingroup Utilities
* @brief
*/
class RandomizedBenchmarking
{
public:
    using CliffordsSeq = std::vector<std::vector<std::shared_ptr <RBGate> >>;
    struct Cliffords
    {
        CliffordsSeq c1_in_xy;
        CliffordsSeq c1_in_xz;
        CliffordsSeq s1;
        CliffordsSeq s1_x;
        CliffordsSeq s1_y;
    };

    RandomizedBenchmarking(MeasureQVMType type, QuantumMachine* qvm);
    ~RandomizedBenchmarking();
    std::map<int, double>single_qubit_rb(Qubit* qbit, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates = {});

    std::map<int, double> two_qubit_rb(Qubit* qbit0, Qubit* qbit1, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates = {});

private:
    Cliffords _single_qubit_cliffords();
    QCircuit _random_single_q_clifford(Qubit* qbit, int num_cfds, const CliffordsSeq& cfd, const std::vector<QStat>& cfd_matrices, const std::vector<QGate>& interleaved_gates);

    QCircuit _two_qubit_clifford_starters(Qubit* q_0, Qubit* q_1, int idx_0, int idx_1, const Cliffords& cfds);
    QCircuit _two_qubit_clifford_mixers(Qubit* q_0, Qubit* q_1, int idx_2, const Cliffords& cfds);
    std::vector<int >_split_two_q_clifford_idx(int idx);
    QCircuit _random_two_q_clifford(Qubit* q_0, Qubit* q_1, int num_cfds, const Cliffords& cfds, const std::vector<QStat>& cfd_matrices, const std::vector<QGate>& interleaved_gates);
    std::vector <QStat>_two_qubit_clifford_matrices(Qubit* q_0, Qubit* q_1, const Cliffords& cfds);

private:
    MeasureQVMType m_qvm_type;
    NoiseQVM* m_mea_qvm;

    QCloudMachine* m_cloud_qvm;
};


/**
 * @brief  single gate rb experiment
 * @param[in] NoiseQVM*  noise quantum machine
 * @param[in] Qubit* qubit
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] const std::vector<QGate>  interleaved gates
 * @return std::map<int, double>  rb result of each layer
 */
std::map<int, double> single_qubit_rb(NoiseQVM* qvm, Qubit* qbit, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates = {});


/**
 * @brief  single gate rb experiment
 * @param[in] NoiseQVM*  noise quantum machine
 * @param[in] Qubit* qubit0
 * @param[in] Qubit* qubit1
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] const std::vector<QGate>  interleaved gates
 * @return std::map<int, double>  rb result of each layer
 */
std::map<int, double> double_qubit_rb(NoiseQVM* qvm, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates = {});


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
std::map<int, double> single_qubit_rb(QCloudMachine* qvm, Qubit* qbit, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates = {});


/**
 * @brief  single gate rb experiment
 * @param[in] QCloudMachine*  cloud quantum machine
 * @param[in] Qubit* qubit0
 * @param[in] Qubit* qubit1
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] const std::vector<QGate>  interleaved gates
 * @return std::map<int, double>  rb result of each layer
 */
std::map<int, double> double_qubit_rb(QCloudMachine* qvm, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates = {});

QPANDA_END
#endif