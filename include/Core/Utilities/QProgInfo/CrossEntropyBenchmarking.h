#ifndef _CROSS_ENTROPY_BENCHMARKING_H
#define _CROSS_ENTROPY_BENCHMARKING_H
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/QCloudMachine.h"
QPANDA_BEGIN

/**
* @class MeasureQVMType
* @brief  calibrated machine type
*/
enum class MeasureQVMType {
	NOISE,  /**< Cpu quantum machine with  noise */
	WU_YUAN /**<Wu Yuan real chip */
};

/**
* @class CrossEntropyBenchmarking
* @ingroup Utilities
* @brief use cross entropy benchmarking (XEB) to calibrate general single- and two-qubit gates
*/
class CrossEntropyBenchmarking
{
public:
	using ProbsDict =  std::vector <std::vector<std::vector<double>>>;

	CrossEntropyBenchmarking(MeasureQVMType type, QuantumMachine* qvm);
	~CrossEntropyBenchmarking();

	/**
	 * @brief  calculate xeb fidelity
	 * 	@param[in] GateType  gate type for calculating fidelity, must be double gate
	 * @param[in] Qubit*  qubit0
	 * @param[in] Qubit*  qubit1  , Must be adjacent to qubit0
	 * @param[in] const std::vector<int>&  the size of each layer
	 * @param[in] int  number of circuits of each layer
	 * @param[in] int  measure shot number
	 * @return std::map<int, double>  xeb result of each layer
	 */
	std::map<int, double> calculate_xeb_fidelity(GateType gt, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& cycle_range, int num_circuits, int shots);

private:

	/**
	 * @brief use double gate build  entangling layers
	 * @param[in] GateType  gate type
	 */
	void _build_entangling_layers(GateType gt);
	
	/**
	 * @brief build single gate set 
	 * @param[in] int  layers number
	 */
	void _random_half_rotations(int num_layers);

	/**
	 * @brief  build xeb circuits
	 * @param[out] std::vector<QProg>&  ideal circuits
	 * @param[out] std::vector<QProg>&  measure circuits
	 */
	void _build_xeb_circuits(std::vector<QProg>& exp_prog, std::vector<QProg>& mea_prog);

	/**
	 * @brief execute calculate
	 * @param[out] const ProbsDict&  ideal circuits probs
	 * @param[out] const ProbsDict&  measure circuits probs
	 * @return std::map<int, double>  xeb result of each layer
	 */
	std::map<int, double> _xeb_fidelities(const ProbsDict& ideal_probs, const ProbsDict& actual_probs);
	
private:
	std::vector<int> m_cycle_range;
	int m_num_circuits;

	QCloudMachine* m_cloud_qvm;

	MeasureQVMType m_mea_qvm_type;

	NoiseQVM* m_mea_qvm;
	QVec m_mea_qubits;
	std::vector<ClassicalCondition > m_mea_cc;
	std::vector<QCircuit > m_mea_single_rots;

	CPUQVM* m_exp_qvm;
	QVec m_exp_qubits;
	std::vector<QCircuit > m_exp_single_rots;

	std::function<QGate(Qubit*, Qubit*)> m_double_gate_func;
};


/**
 * @brief  calculate double gate xeb 
 * @param[in] NoiseQVM*  noise quantum machine
 * @param[in] Qubit* qubit0
 * @param[in] Qubit* qubit1
 * @param[in] const std::vector<int>&   number of layer  
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] GateType  gate type
 * @return std::map<int, double>  xeb result of each layer
 */
std::map<int, double> double_gate_xeb(NoiseQVM* qvm, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& range, int num_circuits, int shots, GateType gt = GateType::CZ_GATE);

/**
 * @brief  calculate double gate xeb
 * @param[in] QCloudMachine*  cloud quantum machine
 * @param[in] Qubit* qubit0
 * @param[in] Qubit* qubit1
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of circuit per layer
 * @param[in] int run number
 * @param[in] GateType  gate type
 * @return std::map<int, double>  xeb result of each layer
 */
std::map<int, double> double_gate_xeb(QCloudMachine* qvm,  Qubit* qbit0, Qubit* qbit1, const std::vector<int>& range, int num_circuits, int shots, GateType gt = GateType::CZ_GATE);

QPANDA_END
#endif