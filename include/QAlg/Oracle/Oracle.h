#ifndef  _ORACLE_H
#define  _ORACLE_H

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>
#include "SearchDataType.h"
#include "SearchSpace.h"
#include "SearchCondition.h"

QPANDA_BEGIN


template <class gate_fun, class... Args>
QCircuit get_mark_circuit(gate_fun &&op_gate, Args && ... args) {
	QCircuit mark_cir;
	auto mark_gate = op_gate(std::forward<Args>(args)...);
	mark_cir << mark_gate;
	return mark_cir;
}

template <class T>
class OracleBuilder
{
public:
	OracleBuilder(const std::vector<T> &data_vec, ClassicalCondition condition, QuantumMachine *qvm)
	:m_data_vec(data_vec), m_condition(condition), m_qvm(*qvm)
		, m_search_space(qvm, condition), m_search_condition(qvm, condition)
	{
		/* 辅助qubit可提高Grover算法成功率，不可省略! */
		create_ancilla_qubits();

		// build U circuit
		m_cir_u = m_search_space.build_to_circuit(m_data_vec);
	}

	~OracleBuilder() {}

	QCircuit build_oracle_circuit(QCircuit cir_mark) {
		QCircuit circuit_oracle;

		//build search circuit
		QVec data_qubits = m_search_space.get_data_qubits();
		QCircuit target_mark_cir = cir_mark.control(data_qubits);

		QCircuit cir_search = m_search_condition.build_to_circuit(data_qubits, m_search_space.get_mini_data());

		m_cir_search << cir_search << target_mark_cir << cir_search;

		//build oracle
		circuit_oracle << m_cir_u << m_cir_search << m_cir_u.dagger();

		return circuit_oracle;
	}

	const QVec& get_index_qubits() { 
		return m_search_space.get_index_qubits();
	}

	const QVec& get_oracle_qubits() {
		return m_search_space.get_data_qubits();
	}

	const QVec& get_ancilla_qubits() {
		return m_ancilla_qubits;
	}

protected:
	void create_ancilla_qubits(size_t qubit_number = 1) {
		m_ancilla_qubits = m_qvm.allocateQubits(qubit_number);
	}

private:
	const std::vector<T> &m_data_vec;
	ClassicalCondition m_condition;
	QuantumMachine& m_qvm;
	SearchSpace<T> m_search_space;
	SearchCondition<T> m_search_condition;
	QVec m_ancilla_qubits;
	QCircuit m_cir_u;
	QCircuit m_cir_search;
};

QPANDA_END

#endif
