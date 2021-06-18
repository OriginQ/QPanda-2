#ifndef _GET_ALL_USED_QUBIT_AND_CBIT_H
#define _GET_ALL_USED_QUBIT_AND_CBIT_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include <set>

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
	
	QVec get_used_qubits() { 
		QVec qv;
		for (const auto& _q : m_used_qubits){
			qv.emplace_back(_q);
		}
		sort(qv.begin(), qv.end(), [](const Qubit* a, const Qubit* b) {
			return a->get_phy_addr() < b->get_phy_addr(); });

		return qv;
	}

	std::vector<int> get_used_cbits() { 
		std::vector<int> cv;
		for (const auto& _m : m_measure_nodes) {
			cv.emplace_back(_m->getCBit()->get_addr());
		}
		sort(cv.begin(), cv.end(), [](const int& a, const int& b) { return a < b; });

		return cv;
	}

	std::vector<std::pair<uint32_t, uint32_t>> get_measure_info() {
		std::vector<std::pair<uint32_t, uint32_t>> measure_info;
		for (const auto& _m : m_measure_nodes) {
			measure_info.emplace_back(std::make_pair(_m->getQuBit()->get_phy_addr(), _m->getCBit()->get_addr()));
		}

		return measure_info;
	}

protected:
	/*!
	* @brief  Execution traversal qgatenode
	* @param[in,out]  AbstractQGateNode*  quantum gate
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		QVec qv;
		cur_node->getQuBitVector(qv);
		cur_node->getControlVector(qv);
		qv.insert(qv.end(), cir_param.m_control_qubits.begin(), cir_param.m_control_qubits.end());
		for (const auto& _q : qv) {
			m_used_qubits.emplace(_q);
		}
	}

	/*!
	* @brief  Execution traversal measure node
	* @param[in,out]  AbstractQuantumMeasure*  measure node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		m_used_qubits.emplace(cur_node->getQuBit());
		m_measure_nodes.emplace(cur_node);
	}

	/*!
   * @brief  Execution traversal reset node
   * @param[in,out]  AbstractQuantumReset*  reset node
   * @param[in]  AbstractQGateNode*  quantum gate
   * @return     void
   */
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)override {
		if (cir_param.m_control_qubits.size() > 0) {
			QCERR_AND_THROW(run_fail, "Error: illegal control-qubits on reset node.");
		}
		m_used_qubits.emplace(cur_node->getQuBit());
	}
	
private:
	std::set<Qubit *> m_used_qubits;
	std::set<std::shared_ptr<AbstractQuantumMeasure>> m_measure_nodes;
};


QPANDA_END
#endif // _GET_ALL_USED_QUBIT_AND_CBIT_H