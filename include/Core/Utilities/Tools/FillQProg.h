#pragma once
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include <memory>
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

class FillQProg
{
public:
	FillQProg(QProg &input_prog)
	{
		pickUpNode(m_input_prog, input_prog, input_prog.getFirstNodeIter(), input_prog.getEndNodeIter(), true);

		fill_by_I();
	}

	void fill_by_I() {
		//layer
		m_grapth_match.get_topological_sequence(m_input_prog, m_seq);

		get_all_used_qubits(m_input_prog, m_vec_qubits_in_use);

		// rebuild the output prog
		const QProgDAG& prog_dag = m_grapth_match.getProgDAG();
		for (auto &seq_item : m_seq)
		{
			QVec vec_qubits_used_in_layer;
			for (auto &seq_node_item : seq_item)
			{
				SequenceNode n = seq_node_item.first;
				if (-1 == n.m_node_type)
				{
					std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(prog_dag.get_vertex(n.m_vertex_num));
					QMeasure tmp_measure_node(p_measure);
					vec_qubits_used_in_layer.push_back(tmp_measure_node.getQuBit());
					m_output_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(tmp_measure_node)).getImplementationPtr()));
				}
				else
				{
					std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(prog_dag.get_vertex(n.m_vertex_num));
					QGate tmp_gate_node(p_gate);
					QVec gate_qubits;
					tmp_gate_node.getQuBitVector(gate_qubits);
					m_output_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(tmp_gate_node)).getImplementationPtr()));
					for (auto &itr : gate_qubits)
					{
						vec_qubits_used_in_layer.push_back(itr);
					}
				}
			}

			auto unused_qubits_vec = get_unused_qubits_in_layer(m_vec_qubits_in_use, vec_qubits_used_in_layer);
			for (auto itr : unused_qubits_vec)
			{
				m_output_prog << I(itr);
			}
		}
	}

	QProg& get_output_prog() { return m_output_prog; }

	QVec get_unused_qubits_in_layer(QVec &all_qubits, QVec &inuse_qubits) {
		std::sort(all_qubits.begin(), all_qubits.end());
		std::sort(inuse_qubits.begin(), inuse_qubits.end());

		QVec unused_qubits_vec;
		set_difference(all_qubits.begin(), all_qubits.end(), inuse_qubits.begin(), inuse_qubits.end(), std::back_inserter(unused_qubits_vec));
		return unused_qubits_vec;
	}

private:
	QProg m_input_prog;
	QProg m_output_prog;
	GraphMatch m_grapth_match;
	TopologicalSequence m_seq;
	QVec m_vec_qubits_in_use;
};

/**
* @brief  Fill the input QProg by I gate
* @param[in] The input Qprog
* @return the filled QProg
* @see
*/
inline QProg fill_qprog_by_I(QProg &input_prog) {
	FillQProg filler(input_prog);
	return filler.get_output_prog();
}

QPANDA_END