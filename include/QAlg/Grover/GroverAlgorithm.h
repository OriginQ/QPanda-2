
#ifndef  _GROVER_ALGORITHM_H
#define  _GROVER_ALGORITHM_H

#include <vector>
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "QAlg/Oracle/SearchDataType.h"
#include "QAlg/Oracle/SearchSpace.h"
#include "QAlg/Oracle/SearchCondition.h"
#include "QAlg/Oracle/Oracle.h"
#include "DiffusionCircuit.h"
#include "QuantumCounting.h"

QPANDA_BEGIN

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#define PTraceQCircuit(string, cir) (std::cout << string << endl << cir << endl)
#define PTraceQCirMat(string, cir) {auto m = getCircuitMatrix(cir); std::cout << string << endl << m << endl;}
#else
#define PTrace
#define PTraceMat(mat)
#define PTraceMat(string, cir)
#define PTraceQCirMat(string, cir)
#define PTraceQCircuit(string, cir)
#endif

using grover_oracle = Oracle<QVec, Qubit*>;

/**
* @brief  Grover Algorithm
* @ingroup Grover_Algorithm
* @param[in] size_t target number
* @param[in] size_t search range
* @param[in] QuantumMachine* Quantum machine ptr
* @param[in] grover_oracle Grover Algorithm oracle
* @return    QProg
* @note  
*/
QProg groverAlgorithm(size_t target,
	size_t search_range,
	QuantumMachine * qvm,
	grover_oracle oracle);

inline QProg grover_alg(QCircuit cir_oracle, 
	QCircuit cir_diffusion, 
	const QVec &data_index_qubits,
	const QVec &ancilla_qubits,
	size_t repeat) {
	QProg grover_prog;

	//initial state prepare
	QCircuit circuit_prepare = apply_QGate(data_index_qubits, H);
	grover_prog << circuit_prepare;

	//anclilla qubits
	grover_prog << X(ancilla_qubits.back()) << H(ancilla_qubits.back());
	/*for (const auto qubit : ancilla_qubits)
	{
		grover_prog << X(qubit) << H(qubit);
	}*/

	//repeat oracle
	for (size_t i = 0; i < repeat; ++i)
	{
		grover_prog << cir_oracle << cir_diffusion;
	}

	return grover_prog;
}

/**
* @brief  Grover search Algorithm
* @ingroup Grover_Algorithm
* @param[in] std::vector<T> search space
* @param[in] ClassicalCondition search_condition
* @param[in] QuantumMachine* the quantum virtual machine
* @param[out] QVec& the target measure qubits
* @param[in] size_t iterations number
* @return the grove algorithm's QProg
* @note
*/
template <class T>
QProg build_grover_alg_prog(const std::vector<T> &data_vec,
	ClassicalCondition condition,
	QuantumMachine * qvm,
	QVec& measure_qubits,
	size_t repeat = 0)
{
	static_assert(std::is_base_of<AbstractSearchData, T>::value, "Bad search data Type, PLEASE see AbstractSearchData.");

	auto grover_prog = QProg();

	//oracle
	OracleBuilder<T> oracle_builder(data_vec, condition, qvm);
	const QVec &ancilla_qubits = oracle_builder.get_ancilla_qubits();
	QCircuit mark_cir = get_mark_circuit(static_cast<QGate(*)(Qubit *)>(&X), ancilla_qubits.back());
	QCircuit cir_oracle = oracle_builder.build_oracle_circuit(mark_cir);

	//diffusion
	DiffusionCirBuilder diffusion_op;
	QCircuit cir_diffusion = build_diffusion_circuit(oracle_builder.get_index_qubits(), diffusion_op);
	PTraceQCircuit("cir_diffusion", cir_diffusion);

	//quantum counting
	if (0 == repeat)
	{
		PTrace("Strat quantum-counting.\n");
		QuantumCounting quantum_count_alg(qvm, cir_oracle, cir_diffusion, oracle_builder.get_index_qubits(), oracle_builder.get_ancilla_qubits());
		repeat = quantum_count_alg.qu_counting();
	}

	//grover
	grover_prog = grover_alg(cir_oracle, cir_diffusion, oracle_builder.get_index_qubits(), oracle_builder.get_ancilla_qubits(), repeat);
	measure_qubits = oracle_builder.get_index_qubits();

	return grover_prog;
}

inline QProg build_grover_prog(const std::vector<uint32_t> &data_vec,
	ClassicalCondition condition,
	QuantumMachine * qvm,
	QVec& measure_qubits,
	size_t repeat = 0)
{
	std::vector<SearchDataByUInt> target_data_vec(data_vec.begin(), data_vec.end());
	return build_grover_alg_prog(target_data_vec, condition, qvm, measure_qubits, repeat);
}

std::vector<size_t> search_target_from_measure_result(const prob_dict& measure_result);

/**
* @brief  Grover search Algorithm example
* @ingroup Grover_Algorithm
* @param[in] std::vector<T> search space
* @param[in] ClassicalCondition search_condition
* @param[out] std::vector<T>& vector of the search result
* @param[in] QuantumMachine* the quantum virtual machine
* @param[in] size_t iterations number for oracle circuit, default is 2
* @return the grove algorithm's QProg 
* @note
*/
template <class T>
QProg grover_alg_search_from_vector(const std::vector<T> &data_vec,
	ClassicalCondition condition,
	std::vector<size_t> &result_index_vec,
	QuantumMachine * qvm,
	size_t repeat = 2)
{
	QVec data_qubits;
	QProg grover_prog = build_grover_alg_prog(data_vec, condition, qvm, data_qubits, repeat);

	//measure
	PTrace("Strat pmeasure.\n");
	auto result = probRunDict(grover_prog, data_qubits);

	//get result
	result_index_vec = search_target_from_measure_result(result);

	return grover_prog;
}

QPANDA_END

#endif