
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

QPANDA_BEGIN

QCircuit build_coin_circuit(QVec &coin_qubits, QVec &index_qubits, QCircuit cir_mark);

QProg quantum_walk_alg(QCircuit cir_oracle,
	QCircuit cir_coin,
	const QVec &index_qubits,
	const QVec &ancilla_qubits,
	size_t repeat);

/**
* @brief  quantum walk search Algorithm example
* @ingroup quantum_walk_Algorithm
* @param[in] std::vector<T> search space
* @param[in] ClassicalCondition search_condition
* @param[out] std::vector<T> vector of the search result
* @return the quantum walk algorithm's QProg
* @note
*/
template <class T>
QProg quantum_walk_alg_search_from_vector(const std::vector<T> &data_vec,
	ClassicalCondition condition,
	std::vector<size_t> &result_index_vec,
	QuantumMachine * qvm)
{
	static_assert(std::is_base_of<AbstractSearchData, T>::value, "Bad search data Type, PLEASE see AbstractSearchData.");

	auto quantum_walk_prog = QProg();

	//oracle
	OracleBuilder<T> oracle_builder(data_vec, condition, qvm);
	const QVec &ancilla_qubits = oracle_builder.get_ancilla_qubits();
	QCircuit mark_cir = get_mark_circuit(U1, ancilla_qubits.back(), PI/2.0);
	QCircuit cir_oracle = oracle_builder.build_oracle_circuit(deepCopy(mark_cir));
	//PTraceQCircuit("cir_oracle", cir_oracle);

	//build coin circuit
	QVec index_qubits = oracle_builder.get_index_qubits();
	QVec coin_quits = qvm->allocateQubits(index_qubits.size());
	QCircuit cir_coin = build_coin_circuit(coin_quits, index_qubits, deepCopy(mark_cir));
	//PTraceQCircuit("cir_coin", cir_coin);

	//quantum counting
	size_t repeat = 2;
	
    //quantum walk
	quantum_walk_prog = quantum_walk_alg(cir_oracle, cir_coin, oracle_builder.get_index_qubits(), oracle_builder.get_ancilla_qubits(), repeat);
	//PTraceQCircuit("quantum_walk_prog", quantum_walk_prog);

	//measure
	PTrace("Strat pmeasure.\n");
	auto result = probRunDict(quantum_walk_prog, index_qubits);

	//get result
	double total_val = 0.0;
	for (auto& var : result) { total_val += var.second; }
	const double average_probability = total_val / result.size();
	size_t search_result_index = 0;

	PTrace("pmeasure result:\n");
	for (auto aiter : result)
	{
		PTrace("%s:%5f\n", aiter.first.c_str(), aiter.second);
		if (average_probability < aiter.second)
		{
			result_index_vec.push_back(search_result_index);
		}
		++search_result_index;
	}

	return quantum_walk_prog;
}

QPANDA_END

#endif