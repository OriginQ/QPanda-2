#ifndef  _QUANTUM_VOLUME_H_
#define  _QUANTUM_VOLUME_H_

#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/QProgInfo/CrossEntropyBenchmarking.h"


QPANDA_BEGIN

/**
* @class QuantumVolume
* @ingroup Utilities
* @brief Calculate the quantum volume of the chip
*/
class QuantumVolume
{
	struct QvCircuit
	{
		QProg cir;
		int depth;
		int trial;
		QVec qv;
		std::vector < ClassicalCondition > cv;
		prob_vec result;
		float heavy_output;
		int shots;
		int qvm_type;
	};

public:
	QuantumVolume(MeasureQVMType type, QuantumMachine* qvm);
	~QuantumVolume();

	size_t calcQuantumVolume(const std::vector<std::vector<int>> &qubit_lists, int ntrials, int shots);

private:
	void createQvCircuits(std::vector<std::vector<int> > qubit_lists, int ntrials,
		std::vector<std::vector <QvCircuit> >& circuits, std::vector<std::vector <QvCircuit> >& circuits_nomeas);

	void calcIdealResult(std::vector<std::vector <QvCircuit> >& circuits_nomeas);

	void calcINoiseResult(std::vector<std::vector <QvCircuit> >& circuits, int shots);

	void calcStatistics();
	void calcHeavyOutput(int trial, int depth, prob_vec probs);
	size_t volumeResult();

	std::vector<int> randomPerm(int depth);

private:
	int m_shots;
	CPUQVM *m_qvm;
	NoiseQVM *m_noise_qvm;
	QCloudMachine * m_qcm;
	std::vector<std::vector<int> > m_qubit_lists;
	std::vector<int> m_depth_list;
	int m_ntrials;
	std::map<int, std::string> m_gate_type_map;
	std::vector<std::vector <QvCircuit> > m_circuits;
	std::vector<std::vector <QvCircuit> >m_circuits_nomeas;
	std::map<std::pair<int, int >, std::vector<std::string>> m_heavy_outputs;  /**<  eg:  <depth, trial> , <"001", "010", ...>    */
	std::map<std::pair<int, int >, double> m_heavy_output_prob_ideal;  /**<  eg:  <depth, trial> , 0.88888888...    */
	std::map<std::pair<int, int >, size_t> m_heavy_output_counts;  /**<  eg:  <depth, trial> , 1000...   */
	std::vector<std::pair <bool, float>> m_success_list;
	std::vector<std::vector <float>> m_ydata;
	MeasureQVMType m_qvm_type;
};

/**
 * @brief  calculate quantum volume
 * @param[in] NoiseQVM*  noise quantum machine
 * @param[in] std::vector <std::vector<int>> qubit_lists, eg: {{1,2}, {1,2,3,4,5}}
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of random iterations
 * @param[in] int shots  
 * @return size_t  quantum volume
 */
size_t calculate_quantum_volume(NoiseQVM * qvm, std::vector <std::vector<int> >qubit_lists, int ntrials, int shots = 1000);

/**
 * @brief  calculate quantum volume
 * @param[in] QCloudMachine*  real chip
 * @param[in] std::vector <std::vector<int>> qubit_lists, eg: {{1,2}, {1,2,3,4,5}}
 * @param[in] const std::vector<int>&   number of layer
 * @param[in] int number of random iterations
 * @param[in] int shots  
 * @return size_t  quantum volume
 */
size_t calculate_quantum_volume(QCloudMachine* qvm, std::vector <std::vector<int> >qubit_lists, int ntrials, int shots = 1000);


QPANDA_END
#endif // !_QUANTUM_VOLUME_H_
