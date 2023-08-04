#pragma once

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

/**
* @class GetSimpleCircuitTopo
* @ingroup Utilities
* @brief Get simple circuit topology, only support single gate and CX,CZ!
*/
class GetSimpleCircuitTopo : public /*GetAllUsedQubitAndCBit*/TraverseByNodeIter
{
public:
	template <typename _Ty>
	void traversal(_Ty &node)
	{
		TraverseByNodeIter::traverse_qprog(node);
	}

	std::map<uint32_t, std::string> get_topo_by_qubit()
	{
		return m_qubit2gate_str;
	}

protected:
	/*!
	* @brief  Execution traversal qgatenode
	* @param[in,out]  AbstractQGateNode*  quantum gate
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override
	{
		if (m_have_multi_ctrl_gate) {
			return;
		}
		if (cir_param.m_control_qubits.size() > 0) {
			m_qubit2gate_str.clear();
			m_qubit2gate_number.clear();
			m_have_multi_ctrl_gate = true;
			return;
		}
		QVec qv;
		cur_node->getQuBitVector(qv);
		cur_node->getControlVector(qv);

		if (cur_node->getQGate()->getOperationNum() == 1) {	/* single gate! */
			uint32_t qubit_number = qv[0]->get_phy_addr();
			uint32_t gate_order = m_qubit2gate_number[qubit_number]++;
			m_qubit2gate_str[qubit_number] += "G_" + std::to_string(gate_order) + "_" + std::to_string(qubit_number) + "|";
		}
		else if (cur_node->getQGate()->getOperationNum() == 2) {/* double gate! */
			int gate_type = cur_node->getQGate()->getGateType();
			if (gate_type == GateType::CNOT_GATE || gate_type == GateType::CZ_GATE) {
				uint32_t ctrl_qubit_number		= qv[0]->get_phy_addr();
				uint32_t target_qubit_number	= qv[1]->get_phy_addr();
				uint32_t ctrl_gate_order		= m_qubit2gate_number[ctrl_qubit_number]++;
				uint32_t target_gate_order		= m_qubit2gate_number[target_qubit_number]++;
				m_qubit2gate_str[ctrl_qubit_number] += ((gate_type == GateType::CNOT_GATE) ? "CX_" : "CZ_")
					+ std::to_string(ctrl_gate_order) + "_" + std::to_string(ctrl_qubit_number)
					+ "_" + std::to_string(target_qubit_number) + "|";
				m_qubit2gate_str[target_qubit_number] += ((gate_type == GateType::CNOT_GATE) ? "CX_" : "CZ_")
					+ std::to_string(target_gate_order) + "_" + std::to_string(ctrl_qubit_number)
					+ "_" + std::to_string(target_qubit_number) + "|";
			}
		}
	}

	/*!
	* @brief  Execution traversal measure node
	* @param[in,out]  AbstractQuantumMeasure*  measure node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override
	{
		uint32_t qubit_number = cur_node->getQuBit()->get_phy_addr();
		m_qubit2gate_str[qubit_number] += "Measure_" + std::to_string(m_qubit2gate_number[qubit_number]) + "_" + std::to_string(qubit_number);
	}

private:

	bool m_have_multi_ctrl_gate {false};
	std::map<uint32_t, std::string> m_qubit2gate_str;
	std::map<uint32_t, uint32_t> m_qubit2gate_number;
};

QPANDA_END