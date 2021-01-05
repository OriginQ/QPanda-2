
#ifndef  _SEARCH_CONDITION_H
#define  _SEARCH_CONDITION_H

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>
QPANDA_BEGIN

//class AbstractSearchCondition
//{
//public:
//	AbstractSearchCondition(QuantumMachine *qvm, ClassicalCondition condition)
//		:m_qvm(*qvm), m_condition(condition)
//	{}
//
//	virtual ~AbstractSearchCondition() {}
//
//	virtual void load_search_condition() = 0;
//	virtual QCircuit build_to_circuit(QVec oracle_qubits, QVec ancilla_qubits) = 0;
//
//protected:
//	QuantumMachine &m_qvm;
//	ClassicalCondition m_condition;
//};

//class SearchCondition : public AbstractSearchCondition
template <class T>
class SearchCondition
{
public:
	SearchCondition(QuantumMachine *qvm, ClassicalCondition condition)
		:m_qvm(*qvm), m_condition(condition)
	{}
	~SearchCondition() {}

	void load_search_condition() {}

	QCircuit build_to_circuit(QVec oracle_qubits, QVec ancilla_qubits, const AbstractSearchData &mini_data, QCircuit cir_mark) {
		auto ret_cir = QCircuit();
		std::string left_str = m_condition.getExprPtr()->getLeftExpr()->getName();
		std::string operator_str = m_condition.getExprPtr()->getName();
		std::string right_str = m_condition.getExprPtr()->getRightExpr()->getName();

		QCircuit condition_cir;

		if (0 == strcmp(operator_str.c_str(), "=="))
		{
			T search_data;
			search_data.set_val(right_str.c_str());
			condition_cir = search_data.build_to_condition_circuit(oracle_qubits, cir_mark, mini_data);
		}
		else
		{
			//throw error
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: unsupport operator.");
		}

		ret_cir << condition_cir;
		return ret_cir;
	}

private:
	QuantumMachine &m_qvm;
	ClassicalCondition m_condition;
};

QPANDA_END

#endif