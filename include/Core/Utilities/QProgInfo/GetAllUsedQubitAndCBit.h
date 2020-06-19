#ifndef _GET_ALL_USED_QUBIT_AND_CBIT_H
#define _GET_ALL_USED_QUBIT_AND_CBIT_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/Tools/Traversal.h"

QPANDA_BEGIN

/**
* @class GetAllUsedQubitAndCBit
* @ingroup Utilities
* @brief get all used qubit and cbit
*/
class GetAllUsedQubitAndCBit : public TraversalInterface<>
{
public:
	GetAllUsedQubitAndCBit() {}
	~GetAllUsedQubitAndCBit() {}

	template <typename _Ty>
	void traversal(_Ty &node)
	{
		execute(node.getImplementationPtr(), nullptr);
	}
	
	const QVec& get_used_qubits() { return m_used_qubits; }
	const std::vector<int>& get_used_cbits() { return m_used_cbits; }

	/*!
	* @brief  Execution traversal qgatenode
	* @param[in,out]  AbstractQGateNode*  quantum gate
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node) override {
		cur_node->getQuBitVector(m_used_qubits);
		cur_node->getControlVector(m_used_qubits);
	}

	/*!
	* @brief  Execution traversal measure node
	* @param[in,out]  AbstractQuantumMeasure*  measure node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) override {
		m_used_qubits.push_back(cur_node->getQuBit());

		m_used_cbits.push_back(cur_node->getCBit()->getValue());
	}

	/*!
   * @brief  Execution traversal reset node
   * @param[in,out]  AbstractQuantumReset*  reset node
   * @param[in]  AbstractQGateNode*  quantum gate
   * @return     void
   */
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)override {
		m_used_qubits.push_back(cur_node->getQuBit());
	}

	/*!
	* @brief  Execution traversal control flow node
	* @param[in,out]  AbstractControlFlowNode*  control flow node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node) override{
		Traversal::traversal(cur_node, *this);
	}


	/*!
	* @brief  Execution traversal qcircuit
	* @param[in,out]  AbstractQuantumCircuit*  quantum circuit
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node) override{
		Traversal::traversal(cur_node, false, *this);
	}
	/*!
	* @brief  Execution traversal qprog
	* @param[in,out]  AbstractQuantumProgram*  quantum prog
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node) override{
		Traversal::traversal(cur_node, *this);
	}
	/*!
	* @brief  Execution traversal qprog
	* @param[in,out]  AbstractClassicalProg*  quantum prog
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node) override{
		m_used_cbits.push_back(cur_node->get_val());
	}

private:
	QVec m_used_qubits;
	std::vector<int> m_used_cbits;
};


QPANDA_END
#endif // _GET_ALL_USED_QUBIT_AND_CBIT_H