#ifndef _GET_ALL_USED_QUBIT_AND_CBIT_H
#define _GET_ALL_USED_QUBIT_AND_CBIT_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

/**
* @class GetAllUsedQubitAndCBit
* @ingroup Utilities
* @brief get all used qubit and cbit
*/
class GetAllUsedQubitAndCBit : public TraverseByNodeIter
{
public:
	GetAllUsedQubitAndCBit() {}
	~GetAllUsedQubitAndCBit() {}

	template <typename _Ty>
	void traversal(_Ty &node)
	{
		/*execute(node.getImplementationPtr(), nullptr);*/
		TraverseByNodeIter::traverse_qprog(node);
	}
	
	const QVec& get_used_qubits() { return m_used_qubits; }
	const std::vector<int>& get_used_cbits() { return m_used_cbits; }

	/*!
	* @brief  Execution traversal qgatenode
	* @param[in,out]  AbstractQGateNode*  quantum gate
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		cur_node->getQuBitVector(m_used_qubits);
		cur_node->getControlVector(m_used_qubits);
		m_used_qubits.insert(m_used_qubits.end(), cir_param.m_control_qubits.begin(), cir_param.m_control_qubits.end());
	}

	/*!
	* @brief  Execution traversal measure node
	* @param[in,out]  AbstractQuantumMeasure*  measure node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		m_used_qubits.push_back(cur_node->getQuBit());
		m_used_qubits.insert(m_used_qubits.end(), cir_param.m_control_qubits.begin(), cir_param.m_control_qubits.end());

		m_used_cbits.push_back(cur_node->getCBit()->getValue());
	}

	/*!
   * @brief  Execution traversal reset node
   * @param[in,out]  AbstractQuantumReset*  reset node
   * @param[in]  AbstractQGateNode*  quantum gate
   * @return     void
   */
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)override {
		m_used_qubits.push_back(cur_node->getQuBit());
		m_used_qubits.insert(m_used_qubits.end(), cir_param.m_control_qubits.begin(), cir_param.m_control_qubits.end());
	}
	
private:
	QVec m_used_qubits;
	std::vector<int> m_used_cbits;
};


QPANDA_END
#endif // _GET_ALL_USED_QUBIT_AND_CBIT_H