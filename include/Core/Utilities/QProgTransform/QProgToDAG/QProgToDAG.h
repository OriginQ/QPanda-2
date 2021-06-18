#ifndef  QPROG_TO_DAG_H_
#define  QPROG_TO_DAG_H_

#include <map>
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"

QPANDA_BEGIN

/**
* @class QProgDAG
* @ingroup Utilities
* @brief transform QProg to DAG
* @note
*/
class QProgToDAG : protected TraverseByNodeIter
{
	class QCirParamForDAG : public QCircuitParam
	{
	public:
		QCirParamForDAG(QProgDAG& dag)
			:m_dag(dag)
		{}

		std::shared_ptr<QCircuitParam> clone() override {
			return std::make_shared<QCirParamForDAG>(*this);
		}

	public:
		QProgDAG& m_dag;
	};

public:
    /**
    * @brief  traversal QProg
    * @param[in]  _Ty& node
    * @param[in]  QProgDAG& prog_dag
    * @return     void
    */
    void traversal(QProg prog, QProgDAG& prog_dag)
    {
		NodeIter NullItr;
		QCirParamForDAG parm(prog_dag);
		TraverseByNodeIter::execute(prog.getImplementationPtr(), nullptr, parm, NullItr);
    }

protected:
    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>, QCircuitParam&, NodeIter&) override;
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>, QCircuitParam&, NodeIter&)override;
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>, QCircuitParam&, NodeIter&)override;
	void execute(std::shared_ptr<AbstractQuantumReset>, std::shared_ptr<QNode>, QCircuitParam&, NodeIter&)override;
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>, QCircuitParam&, NodeIter&)override;

    void transformQGate(std::shared_ptr<AbstractQGateNode>, QCircuitParam&, NodeIter&);

	template <typename _T>
	void transform_non_gate_node(_T cur_node, QProgDAG& prog_dag, NodeIter& cur_iter, int node_type)
	{
		auto p_node_info = std::make_shared<QProgDAGNode>();
		p_node_info->m_qubits_vec.emplace_back(cur_node->getQuBit());
		p_node_info->m_itr = cur_iter;

		prog_dag.add_vertex(p_node_info, (DAGNodeType)(node_type));
	}
};

/**
* @brief  QProg to DAG
* @ingroup Utilities
* @param[in] QProg 
* @return QProgDAG
* @see QProgDAG
*/
std::shared_ptr<QProgDAG> qprog_to_DAG(QProg prog);

QPANDA_END
#endif
