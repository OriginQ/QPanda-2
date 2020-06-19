#pragma once
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include <memory>
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

/**
* @brief Fill quantum program by I quantum gate
* @ingroup Utilities
*/
class FillQProg
{
public:
	FillQProg(QProg &input_prog)
	{
		pickUpNode(m_input_prog, input_prog, {}, input_prog.getFirstNodeIter(), input_prog.getEndNodeIter());

		fill_by_I();
	}

	/**
	* @brief  Fill the input QProg by I gate
	*/
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
				if (SequenceNodeType::MEASURE == n.m_node_type)
				{
					std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(prog_dag.get_vertex(n.m_vertex_num));
					QMeasure tmp_measure_node(p_measure);
					vec_qubits_used_in_layer.push_back(tmp_measure_node.getQuBit());
					m_output_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(tmp_measure_node)).getImplementationPtr()));
				}
				else if (SequenceNodeType::RESET == n.m_node_type)
				{
					std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(prog_dag.get_vertex(n.m_vertex_num));
					QReset tmp_reset_node(p_reset);
					vec_qubits_used_in_layer.push_back(tmp_reset_node.getQuBit());
					m_output_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(tmp_reset_node)).getImplementationPtr()));
				}
				else
				{
					std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(prog_dag.get_vertex(n.m_vertex_num));
					QGate tmp_gate_node(p_gate);
					QVec gate_qubits;
					tmp_gate_node.getQuBitVector(gate_qubits);
					vec_qubits_used_in_layer.insert(vec_qubits_used_in_layer.end(), gate_qubits.begin(), gate_qubits.end());

					//get control qubits
					gate_qubits.clear();
					tmp_gate_node.getControlVector(gate_qubits);
					vec_qubits_used_in_layer.insert(vec_qubits_used_in_layer.end(), gate_qubits.begin(), gate_qubits.end());

					m_output_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(tmp_gate_node)).getImplementationPtr()));
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
		auto sort_fun = [](Qubit*a, Qubit* b) {return a->getPhysicalQubitPtr()->getQubitAddr() < b->getPhysicalQubitPtr()->getQubitAddr(); };
		std::sort(all_qubits.begin(), all_qubits.end(), sort_fun);
		std::sort(inuse_qubits.begin(), inuse_qubits.end(), sort_fun);

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
* @ingroup Utilities
* @param[in] The input Qprog
* @return the filled QProg
* @see
*/
inline QProg fill_qprog_by_I(QProg &input_prog) {
	FillQProg filler(input_prog);
	return filler.get_output_prog();
}

QPANDA_END