#ifndef ADD_BARRIER_H
#define ADD_BARRIER_H

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

class AddBarrier : protected TraverseByNodeIter
{
public:
	AddBarrier() {}
	~AddBarrier() {}

	template <typename T>
	void auto_add_barrier(T &node){
		TraverseByNodeIter::traverse_qprog(node);
	}

protected:
	/*!
	* @brief  Execution traversal qgatenode
	* @param[in,out]  AbstractQGateNode*  quantum gate
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, 
		QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		if (BARRIER_GATE == cur_node->getQGate()->getGateType()){
			return ;
		}

		QVec qv;
		cur_node->getQuBitVector(qv);
		cur_node->getControlVector(qv);
		qv.insert(qv.end(), cir_param.m_control_qubits.begin(), cir_param.m_control_qubits.end());
		if (qv.size() > 1)
		{
			auto pre_node_itr = cur_node_iter;
			std::dynamic_pointer_cast<AbstractNodeManager>(parent_node)->insertQNode(
				--pre_node_itr, std::dynamic_pointer_cast<QNode>(BARRIER(qv).getImplementationPtr()));
		}
		
	}

private:
};

template <typename T>
void auto_add_barrier_before_mul_qubit_gate(T& node) {
	flatten(node);
	AddBarrier().auto_add_barrier(node);
}

QPANDA_END
#endif // !ADD_BARRIER_H
