
#ifndef  _SEARCH_SPACE_H
#define  _SEARCH_SPACE_H

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>
#include "QAlg/Base_QCircuit/base_circuit.h"

QPANDA_BEGIN

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#define PTraceQCircuit(string, cir) (std::cout << string << endl << cir << endl)
#else
#define PTrace
#define PTraceMat(mat)
#define PTraceMat(string, cir)
#endif

template <class T>
class SearchSpace
{
public:
	SearchSpace(QuantumMachine *qvm, ClassicalCondition condition)
		:m_qvm(*qvm), m_condition(condition)
	{}
	~SearchSpace() {}

	const QVec& get_index_qubits() {
		return m_index_qubits;
	}

	const QVec& get_data_qubits() {
		return m_data_qubits;
	}

	QCircuit build_to_circuit(const std::vector<T> &data_vec) {
		auto mini_data_itr = data_vec.begin();
		auto max_data_itr = data_vec.begin();
		for (auto itr = data_vec.begin(); itr != data_vec.end(); ++itr)
		{
			if ((*mini_data_itr) > (*itr))
			{
				mini_data_itr = itr;
			}

			if ((*max_data_itr) < (*itr))
			{
				max_data_itr = itr;
			}
		}

		m_mini_data = (*mini_data_itr);

		size_t data_vec_size = data_vec.size();
		size_t index_qubits_cnt = ceil(log2(data_vec_size));
		m_index_qubits = m_qvm.allocateQubits(index_qubits_cnt);

		T max_weight_data = (*max_data_itr);
		max_weight_data = max_weight_data - (*mini_data_itr);
		size_t need_data_qubits_num = max_weight_data.check_max_need_qubits();
		m_data_qubits = m_qvm.allocateQubits(need_data_qubits_num);

		size_t index = 0;
		QCircuit index_cir = index_to_circuit(index, m_index_qubits);
		for (const auto &item : data_vec)
		{
			auto tmp_cir = item.build_to_circuit(m_data_qubits, need_data_qubits_num, *mini_data_itr);
			if (tmp_cir.getFirstNodeIter() != tmp_cir.getEndNodeIter())
			{
				tmp_cir.setControl(m_index_qubits);
			}
			
			m_cir_U << index_cir << tmp_cir;
			const auto pre_index = index;
			index_cir = index_to_circuit(++index, m_index_qubits, pre_index, true);
		}
		m_cir_U << index_cir;

		return m_cir_U;
	}

	const AbstractSearchData& get_mini_data() {
		return m_mini_data;
	}

private:
	QuantumMachine &m_qvm;
	ClassicalCondition m_condition;
	QVec m_index_qubits;
	QVec m_data_qubits;
	QCircuit m_cir_U;
	T m_mini_data;
};

QPANDA_END

#endif