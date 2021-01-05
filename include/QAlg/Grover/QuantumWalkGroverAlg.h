
#ifndef  QUANTUM_WALK_GROVER_ALG_H
#define  QUANTUM_WALK_GROVER_ALG_H

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
#include "GroverAlgorithm.h"

QPANDA_BEGIN

QCircuit build_coin_circuit(QVec &coin_qubits, QVec &index_qubits, QCircuit cir_mark);

QProg quantum_walk_alg(QCircuit cir_oracle,
	QCircuit cir_coin,
	const QVec &index_qubits,
	const QVec &ancilla_qubits,
	size_t repeat);

/**
* @brief build quantum-walk search Algorithm QProg
* @ingroup Grover_Algorithm
* @param[in] std::vector<T> search space
* @param[in] ClassicalCondition search_condition
* @param[in] QuantumMachine* the quantum virtual machine
* @param[out] QVec& the target measure qubits
* @param[in] size_t iterations number
* @return the quantum-walk algorithm's QProg
* @note
*/
template <class T>
QProg build_quantum_walk_alg_prog(const std::vector<T> &data_vec,
	ClassicalCondition condition,
	QuantumMachine * qvm,
	QVec& measure_qubits,
	size_t repeat = 2)
{
	static_assert(std::is_base_of<AbstractSearchData, T>::value, "Bad search data Type, PLEASE see AbstractSearchData.");

	auto quantum_walk_prog = QProg();

	//oracle
	OracleBuilder<T> oracle_builder(data_vec, condition, qvm);
	const QVec &ancilla_qubits = oracle_builder.get_ancilla_qubits();
	QCircuit mark_cir = get_mark_circuit(U1, ancilla_qubits.back(), PI/2.0);
	QCircuit cir_oracle = oracle_builder.build_oracle_circuit(deepCopy(mark_cir));

	//build coin circuit
	QVec index_qubits = oracle_builder.get_index_qubits();
	QVec coin_quits = qvm->allocateQubits(index_qubits.size());
	QCircuit cir_coin = build_coin_circuit(coin_quits, index_qubits, deepCopy(mark_cir));

	//quantum counting
	if (0 == repeat)
	{
		printf("Strat quantum-counting.\n");
		DiffusionCirBuilder diffusion_op;
		QCircuit cir_diffusion = build_diffusion_circuit(oracle_builder.get_index_qubits(), diffusion_op);
		QuantumCounting quantum_count_alg(qvm, cir_oracle, cir_diffusion, oracle_builder.get_index_qubits(), oracle_builder.get_ancilla_qubits());
		repeat = quantum_count_alg.qu_counting();
	}

	//quantum walk
	quantum_walk_prog = quantum_walk_alg(cir_oracle, cir_coin, oracle_builder.get_index_qubits(), oracle_builder.get_ancilla_qubits(), repeat);
	sub_cir_optimizer(quantum_walk_prog);
	measure_qubits = index_qubits;

	return quantum_walk_prog;
}

inline QProg build_quantum_walk_prog(const std::vector<int> &data_vec,
	ClassicalCondition condition,
	QuantumMachine * qvm,
	QVec& measure_qubits,
	size_t repeat = 0)
{
	std::vector<SearchDataByUInt> target_data_vec(data_vec.begin(), data_vec.end());
	return build_quantum_walk_alg_prog(target_data_vec, condition, qvm, measure_qubits, repeat);
}

/**
* @brief  quantum walk search Algorithm example
* @ingroup quantum_walk_Algorithm
* @param[in] std::vector<T> search space
* @param[in] ClassicalCondition search_condition
* @param[out] std::vector<T> vector of the search result
* @param[in] size_t oracle circuit iterations number, default is 2
* @return the quantum walk algorithm's QProg
* @note
*/
template <class T>
QProg quantum_walk_alg_search_from_vector(const std::vector<T> &data_vec,
	ClassicalCondition condition,
	QuantumMachine * qvm,
	std::vector<size_t> &result_index_vec,
	size_t repeat = 2)
{
	QVec measure_qubits;
	QProg quantum_walk_prog = build_quantum_walk_alg_prog(data_vec, condition, qvm, measure_qubits, repeat);

	//measure
	//printf("Strat pmeasure.\n");
	auto result = probRunDict(quantum_walk_prog, measure_qubits);

	//get result
	result_index_vec = search_target_from_measure_result(result, data_vec.size());

	return quantum_walk_prog;
}

QPANDA_END

#endif