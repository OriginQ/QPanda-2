#include "Core/Utilities/QProgInfo/GetAdjacentNodes.h"

USING_QPANDA
using namespace std;

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#else
#define PTrace
#endif

void AdjacentQGates::HaveNotFoundTargetNode::handle_QGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	/* find target
		   if found target, flage = 1; continue
	*/
	if (m_parent.m_target_node_itr == cur_node_iter)
	{
		m_parent.change_traversal_statue(new AdjacentQGates::ToFindBackNode(m_parent, TO_FIND_BACK_NODE));
	}
	else
	{
		m_parent.update_front_iter(cur_node_iter, cir_param);

		//test
		GateType gt = m_parent.get_node_ype(cur_node_iter);
		PTrace(">>gatyT=%d ", gt);
	}
}

void AdjacentQGates::HaveNotFoundTargetNode::handle_QMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	/* find target
		   if found target, flage = 1; continue
		*/
	if (m_parent.m_target_node_itr == cur_node_iter)
	{
		m_parent.change_traversal_statue(new ToFindBackNode(m_parent, TO_FIND_BACK_NODE));
	}
	else
	{
		m_parent.update_front_iter(cur_node_iter, cir_param);

		//test
		PTrace(">>measure_node ");
	}
}

void AdjacentQGates::HaveNotFoundTargetNode::handle_QReset(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	/* find target
		   if found target, flage = 1; continue
		*/
	if (m_parent.m_target_node_itr == cur_node_iter)
	{
		m_parent.change_traversal_statue(new ToFindBackNode(m_parent, TO_FIND_BACK_NODE));
	}
	else
	{
		m_parent.update_front_iter(cur_node_iter, cir_param);

		//test
		PTrace(">>reset_node ");
	}
}

void AdjacentQGates::traverse_qprog()
{
	m_traversal_statue = (new(std::nothrow) HaveNotFoundTargetNode(*this, HAVE_NOT_FOUND_TARGET_NODE));
	if (nullptr == m_traversal_statue)
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Memery error, failed to new traversal-statue obj.");
	}
	else
	{
		TraverseByNodeIter::traverse_qprog();
	}
}

void AdjacentQGates::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	if (nullptr == cur_node)
	{
		QCERR("control_flow_node is nullptr");
		throw std::invalid_argument("control_flow_node is nullptr");
	}

	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

	if (nullptr == pNode)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}
	auto iNodeType = pNode->getNodeType();

	if (WHILE_START_NODE == iNodeType)
	{
		m_traversal_statue->on_enter_QWhile(cur_node, parent_node, cir_param, cur_node_iter);
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		m_traversal_statue->on_leave_QWhile(cur_node, parent_node, cir_param, cur_node_iter);
	}
	else if (QIF_START_NODE == iNodeType)
	{
		m_traversal_statue->on_enter_QIf(cur_node, parent_node, cir_param, cur_node_iter);
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		auto false_branch_node = cur_node->getFalseBranch();

		if (nullptr != false_branch_node)
		{
			Traversal::traversalByType(false_branch_node, pNode, *this, cir_param, cur_node_iter);
		}
		m_traversal_statue->on_leave_QIf(cur_node, parent_node, cir_param, cur_node_iter);
	}
}

GateType AdjacentQGates::get_node_ype(const NodeIter &ter)
{
	std::shared_ptr<QNode> tmp_node = *(ter);
	if (nullptr != tmp_node)
	{
		if (GATE_NODE == tmp_node->getNodeType())
		{
			std::shared_ptr<OriginQGate> gate = std::dynamic_pointer_cast<OriginQGate>(tmp_node);
			return (GateType)(gate->getQGate()->getGateType());
		}
	}

	return GATE_UNDEFINED;
}

std::string AdjacentQGates::get_node_type_str(const NodeIter &ter)
{
	std::shared_ptr<QNode> tmp_node = *(ter);
	if (nullptr != tmp_node.get())
	{
		const NodeType t = tmp_node->getNodeType();
		if (t == GATE_NODE)
		{
			std::shared_ptr<OriginQGate> gate = std::dynamic_pointer_cast<OriginQGate>(tmp_node);
			return TransformQGateType::getInstance()[(GateType)(gate->getQGate()->getGateType())];
		}
		else if (t == MEASURE_GATE)
		{
			return std::string("MEASURE_NODE");
		}
		else if (t == RESET_NODE)
		{
			return std::string("RESET_NODE");
		}
	}

	return std::string("Null");
}