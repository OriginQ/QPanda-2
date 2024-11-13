#pragma once

#include "Core/QuantumCloud/QCloudMachine.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN

struct QCloudTaskConfig
{
    std::string cloud_token = "";
    RealChipType chip_id = RealChipType::ORIGIN_72;
    int shots = 1000;
    bool open_amend = true;
    bool open_mapping = true;
    bool open_optimization = true;
};

#if defined(USE_CURL)

enum class MeasureQVMType 
{
	NOISE,  /**< CPU QUANTUM MACHINE WITH NOISE */
	ORIGIN_REAL_CHIP /**< ORIGIN REAL CHIP */
};

using multi_probs = std::vector<std::vector<std::vector<double>>>;

/**
* @class CrossEntropyBenchmarking
* @ingroup Utilities
* @brief use cross entropy benchmarking (XEB) to calibrate general single- and two-qubit gates
*/
class CrossEntropyBenchmarking
{
public:

	CrossEntropyBenchmarking(QuantumMachine* qvm);
    CrossEntropyBenchmarking(QCloudTaskConfig config);

	~CrossEntropyBenchmarking();

	/**
	 * @brief  calculate xeb fidelity
	 * @param[in] GateType  gate type for calculating fidelity, must be double gate
	 * @param[in] Qubit*  qubit0
	 * @param[in] Qubit*  qubit1  , Must be adjacent to qubit0
	 * @param[in] const std::vector<int>&  the size of each layer
	 * @param[in] int  number of circuits of each layer
	 * @param[in] int  measure shot number
	 * @return std::map<int, double>  xeb result of each layer
	 */
    std::map<int, double> calculate_xeb_fidelity(GateType gt, 
        Qubit* qbit0, 
        Qubit* qbit1, 
        const std::vector<int>& cycle_range, 
        int num_circuits, 
        int shots, 
        RealChipType chip_type = RealChipType::ORIGIN_WUYUAN_D5);

    std::map<int, double> calculate_xeb_fidelity(
        GateType gate_name,
        int qubit_0,
        int qubit_1,
        const std::vector<int>& cycle_range,
        int num_circuits);

private:

    std::function<QGate(Qubit*, Qubit*)> get_benchmarking_gate(GateType gate_name);
	
	/**
	 * @brief build single gate set 
	 * @param[in] int  layers number
	 */
	void random_half_rotations(int num_layers);

	/**
	 * @brief  build xeb circuits
	 * @param[out] std::vector<QProg>&  ideal circuits
	 * @param[out] std::vector<QProg>&  measure circuits
	 */
	void build_xeb_circuits(std::vector<QProg>& exp_prog, std::vector<QProg>& mea_prog, GateType gate_name);

	/**
	 * @brief execute calculate
	 * @param[out] const multi_probs&  ideal circuits probs
	 * @param[out] const multi_probs&  measure circuits probs
	 * @return std::map<int, double>  xeb result of each layer
	 */
	std::map<int, double> _xeb_fidelities(const multi_probs& ideal_probs, const multi_probs& actual_probs);
	
private:

    QCloudMachine m_qcloud;
    QCloudTaskConfig m_cloud_config;

	std::vector<int> m_cycle_range;
	int m_num_circuits;

	QVec m_mea_qubits;
	std::vector<ClassicalCondition > m_mea_cc;
	std::vector<QCircuit > m_mea_single_rots;

	QVec m_exp_qubits;
	std::vector<QCircuit > m_exp_single_rots;

    QMachineType m_machine_type;
    QuantumMachine* m_machine_ptr;
};

/**
 * @brief  calculate double gate xeb
 * @param[in] QCloudMachine*  cloud quantum machine
 * @param[in] Qubit* qubit0
 * @param[in] Qubit* qubit1
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] RealChipType type
 * @param[in] GateType  gate type
 * @return std::map<int, double>  xeb result of each layer
 */
std::map<int, double> double_gate_xeb(QuantumMachine* machine,
    Qubit* qbit0,
    Qubit* qbit1,
    const std::vector<int>& range, 
    int num_circuits, 
    int shots, 
    RealChipType type = RealChipType::ORIGIN_WUYUAN_D5,
    GateType gate_type = GateType::CZ_GATE);

std::map<int, double> double_gate_xeb(QCloudTaskConfig config,
    int qubit_0,
    int qubit_1,
    const std::vector<int>& range,
    int num_circuits,
    GateType gate_type = GateType::CZ_GATE);

#endif

QPANDA_END
