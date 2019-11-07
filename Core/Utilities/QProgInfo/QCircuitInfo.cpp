/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include <algorithm>
#include "Core/Core.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"

USING_QPANDA
using namespace std;

#define ENUM_TO_STR(x) #x

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceMat(mat)
#endif

/*******************************************************************
*                      class TraversalNodeIter
********************************************************************/
void TraverseByNodeIter::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	if (nullptr == cur_node)
	{
		QCERR("pQCircuit is nullptr");
		throw std::invalid_argument("pQCircuit is nullptr");
	}

	auto aiter = cur_node->getFirstNodeIter();

	if (aiter == cur_node->getEndNodeIter())
		return;

	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

	if (nullptr == pNode)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}

	bool cur_node_is_dagger = cur_node->isDagger() ^ (cir_param.m_is_dagger);
	QVec ctrl_qubits;
	cur_node->getControlVector(ctrl_qubits);

	auto tmp_param = cir_param.clone();
	tmp_param->m_is_dagger = cur_node_is_dagger;
	tmp_param->append_control_qubits(QCircuitParam::get_real_append_qubits(ctrl_qubits, cir_param.m_control_qubits));
	if (cur_node_is_dagger)
	{
		auto aiter = cur_node->getLastNodeIter();
		if (nullptr == *aiter)
		{
			return;
		}
		while (aiter != cur_node->getHeadNodeIter())
		{
			if (aiter == nullptr)
			{
				break;
			}
			Traversal::traversalByType(*aiter, pNode, *this, *tmp_param, aiter);
			--aiter;
		}

	}
	else
	{
		auto aiter = cur_node->getFirstNodeIter();
		while (aiter != cur_node->getEndNodeIter())
		{
			auto next = aiter.getNextIter();
			Traversal::traversalByType(*aiter, pNode, *this, *tmp_param, aiter);
			aiter = next;
		}
	}
}

void TraverseByNodeIter::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	if (nullptr == cur_node)
	{
		QCERR("param error");
		throw std::invalid_argument("param error");
	}

	auto aiter = cur_node->getFirstNodeIter();

	if (aiter == cur_node->getEndNodeIter())
		return;


	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

	if (nullptr == pNode)
	{
		QCERR("pNode is nullptr");
		throw std::invalid_argument("pNode is nullptr");
	}

	while (aiter != cur_node->getEndNodeIter())
	{
		auto next = aiter.getNextIter();
		Traversal::traversalByType(*aiter, pNode, *this, cir_param, aiter);
		aiter = next;
	}
}

void TraverseByNodeIter::traverse_qprog()
{
	NodeIter itr = NodeIter();
	auto param = std::make_shared<QCircuitParam>();
	execute(m_prog.getImplementationPtr(), nullptr, *param, itr);
}

/*******************************************************************
*                      class GetAllNodeType
********************************************************************/
void GetAllNodeType::execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle QGate node
	GateType gt = (GateType)(cur_node->getQGate()->getGateType());

	//get quantum bits
	QVec gate_qubits;
	cur_node->getQuBitVector(gate_qubits);

	std::string gateTypeStr = TransformQGateType::getInstance()[gt];
	std::string gateParamStr;
	get_gate_parameter(cur_node, gateParamStr);

	gateTypeStr.append("(");
	for (auto &itr : gate_qubits)
	{
		gateTypeStr.append(std::string("q[") + std::to_string(itr->getPhysicalQubitPtr()->getQubitAddr()) + "], ");
	}

	if (gateParamStr.size() > 0)
	{
		gateParamStr = gateParamStr.substr(1, gateParamStr.length() - 2);
		gateTypeStr.append(gateParamStr);
	}
	else
	{
		gateTypeStr = gateTypeStr.substr(0, gateTypeStr.size() - 2);
	}
	gateTypeStr.append(")");

	if (cur_node->isDagger() ^ (cir_param.m_is_dagger))
	{
		gateTypeStr.append(".dagger()");
	}

	QVec ctrl_qubits;
	if ((0 < cur_node->getControlVector(ctrl_qubits)) || (cir_param.m_control_qubits.size() > 0))
	{
		//get control info
		auto increased_control = QCircuitParam::get_real_append_qubits(cir_param.m_control_qubits, ctrl_qubits);
		ctrl_qubits.insert(ctrl_qubits.end(), increased_control.begin(), increased_control.end());

		gateTypeStr.append(".controled(");
		for (auto& itr : ctrl_qubits)
		{
			gateTypeStr.append(std::to_string(itr->getPhysicalQubitPtr()->getQubitAddr()) + ", ");
		}
		gateTypeStr = gateTypeStr.substr(0, gateTypeStr.size() - 2);
		gateTypeStr.append(")");
	}

	sub_circuit_indent();
	m_output_str.append(std::string("<<") + gateTypeStr);
}

void GetAllNodeType::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle measure node
	sub_circuit_indent();

	//std::string gateTypeStr;
	char measure_buf[258] = "";
	snprintf(measure_buf, 256, "<<Measure(q[%d], c[%d])", cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr(), cur_node->getCBit()->getValue());
	m_output_str.append(measure_buf);
}

void GetAllNodeType::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	// handle flow control node
	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);
	if (nullptr == pNode)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}
	auto iNodeType = pNode->getNodeType();
	std::string node_name;
	if (WHILE_START_NODE == iNodeType)
	{
		node_name = "QWhile";
	}
	else if (QIF_START_NODE == iNodeType)
	{
		node_name = "QIf";
	}

	++m_indent_cnt;
	m_output_str.append(get_indent_str() + "Enter flow control node: " + node_name + ":");
	Traversal::traversal(cur_node, *this, cir_param, cur_node_iter);

	m_output_str.append(get_indent_str() + "Leave flow control node." + node_name + ":");
	--m_indent_cnt;
	if (0 < m_indent_cnt)
	{
		m_output_str.append(get_indent_str());
	}
}

void GetAllNodeType::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	++m_indent_cnt;
	m_output_str.append(get_indent_str() + "Enter sub circuit: ");
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	m_output_str.append(get_indent_str() + "Leave sub circuit.");
	--m_indent_cnt;
	if (0 < m_indent_cnt)
	{
		m_output_str.append(get_indent_str());
	}
}

void GetAllNodeType::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	++m_indent_cnt;
	m_output_str.append(get_indent_str() + "Enter sub program: ");
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	m_output_str.append(get_indent_str() + "Leave sub program.");
	--m_indent_cnt;
	if (0 < m_indent_cnt)
	{
		m_output_str.append(get_indent_str());
	}
}

/*******************************************************************
*                      class AdjacentQGates
********************************************************************/
void AdjacentQGates::HaveNotFoundTargetNode::handleQGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	/* find target
		   if found target, flage = 1; continue
	*/
	if (m_parent.m_target_node_itr == cur_node_iter)
	{
		m_parent.changeTraversalStatue(new AdjacentQGates::ToFindBackNode(m_parent, TO_FIND_BACK_NODE));
	}
	else
	{
		m_parent.updateFrontIter(cur_node_iter);

		//test
		GateType gt = m_parent.getItrNodeType(cur_node_iter);
		PTrace(">>gatyT=%d ", gt);
	}
}

void AdjacentQGates::HaveNotFoundTargetNode::handleQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	/* find target
		   if found target, flage = 1; continue
		*/
	if (m_parent.m_target_node_itr == cur_node_iter)
	{
		m_parent.changeTraversalStatue(new ToFindBackNode(m_parent, TO_FIND_BACK_NODE));
	}
	else
	{
		m_parent.updateFrontIter(cur_node_iter);

		//test
		PTrace(">>measureGate ");
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
		m_traversal_statue->onEnterQWhile(cur_node, parent_node, cir_param, cur_node_iter);
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		m_traversal_statue->onLeaveQWhile(cur_node, parent_node, cir_param, cur_node_iter);
	}
	else if (QIF_START_NODE == iNodeType)
	{
		m_traversal_statue->onEnterQIf(cur_node, parent_node, cir_param, cur_node_iter);
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		auto false_branch_node = cur_node->getFalseBranch();

		if (nullptr != false_branch_node)
		{
			Traversal::traversalByType(false_branch_node, pNode, *this, cir_param, cur_node_iter);
		}
		m_traversal_statue->onLeaveQIf(cur_node, parent_node, cir_param, cur_node_iter);
	}
}

GateType AdjacentQGates::getItrNodeType(const NodeIter &ter)
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

std::string AdjacentQGates::getItrNodeTypeStr(const NodeIter &ter)
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
			return std::string("MEASURE_GATE");
		}
	}

	return std::string("Null");
}

/*******************************************************************
*                      class JudgeTwoNodeIterIsSwappable
********************************************************************/
void JudgeTwoNodeIterIsSwappable::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
	std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);
	if (nullptr == pNode)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}

	m_judge_statue->onEnterCircuit(cur_node, cir_param);

	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);

	m_judge_statue->onLeaveCircuit(cur_node, cir_param);
}

void JudgeTwoNodeIterIsSwappable::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
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
		m_judge_statue->onEnterFlowCtrlNode();
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		m_judge_statue->onLeaveFlowCtrlNode();
	}
	else if (QIF_START_NODE == iNodeType)
	{
		m_judge_statue->onEnterFlowCtrlNode();
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		m_judge_statue->onLeaveFlowCtrlNode();

		auto false_branch_node = cur_node->getFalseBranch();

		if (nullptr != false_branch_node)
		{
			m_judge_statue->onEnterFlowCtrlNode();
			Traversal::traversalByType(false_branch_node, pNode, *this, cir_param, cur_node_iter);
			m_judge_statue->onLeaveFlowCtrlNode();
		}
	}
}

bool JudgeTwoNodeIterIsSwappable::getResult()
{
	return (COULD_BE_EXCHANGED == m_result);
}

int JudgeTwoNodeIterIsSwappable::judgeLayerInfo()
{
#if PRINT_TRACE
	cout << "test the pick result:" << endl;
	printAllNodeType(m_pick_prog);
#endif

	//get layer info
	GraphMatch grap_match;
	TopologicalSequence seq;
	grap_match.get_topological_sequence(m_pick_prog, seq);
	const QProgDAG &tmp_dag = grap_match.getProgDAG();

	int found_cnt = 0;
	for (auto &seq_item : seq)
	{
		for (auto &seq_node_item : seq_item)
		{
			if (m_nodeItr1 == tmp_dag.get_vertex_nodeIter(seq_node_item.first.m_vertex_num))
			{
				++found_cnt;
			}

			if (m_nodeItr2 == tmp_dag.get_vertex_nodeIter(seq_node_item.first.m_vertex_num))
			{
				++found_cnt;
			}
		}

		if (2 == found_cnt)
		{
			changeStatue(new CoubleBeExchange(*this, COULD_BE_EXCHANGED));
			return 0;
		}
		else if (1 == found_cnt)
		{
			changeStatue(new CanNotBeExchange(*this, CAN_NOT_BE_EXCHANGED));
			return 0;
		}
		else if (0 == found_cnt)
		{
			continue;
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: unknow error.");
			return -1;
		}
	}

	QCERR_AND_THROW_ERRSTR(runtime_error, "Error: get layer error.");
	return -1;
}


void JudgeTwoNodeIterIsSwappable::traverse_qprog()
{
	m_judge_statue = (new(std::nothrow) OnInitStatue(*this, INIT));
	if (nullptr == m_judge_statue)
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Memery error, failed to new traversal-statue obj.");
	}
	else
	{
		TraverseByNodeIter::traverse_qprog();
		m_judge_statue->onTraversalEnd();
	}
}

void JudgeTwoNodeIterIsSwappable::pickNode(const NodeIter iter)
{
	m_pick_prog.pushBackNode(*iter);
	if (iter == m_nodeItr1)
	{
		m_b_found_first_iter = true;
		m_nodeItr1 = m_pick_prog.getLastNodeIter();
	}
	else if (iter == m_nodeItr2)
	{
		m_b_found_second_iter = true;
		m_nodeItr2 = m_pick_prog.getLastNodeIter();
	}
}

/*******************************************************************
*                      class PickUpNodes
********************************************************************/
void PickUpNodes::execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle QGate node
	pickUp(cur_node_iter, std::mem_fn(&PickUpNodes::pickQGateNode), this, cur_node_iter, cir_param);
}

void PickUpNodes::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle measure node
	pickUp(cur_node_iter, std::mem_fn(&PickUpNodes::pickQMeasureNode), this, cur_node_iter);
}

void PickUpNodes::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	// handle flow control node
	if (m_b_picking)
	{
		//if there are Qif/Qwhile node, throw an exception 
		m_b_pickup_end = true;
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Illegal Qif/QWhile nodes.");
		m_output_prog.clear();
		return;
	}

	Traversal::traversal(cur_node, *this, cir_param, cur_node_iter);
}

void PickUpNodes::no_dagger_gate(QGate& gate)
{
#if 0
	const GateType type = (GateType)(gate.getQGate()->getGateType());
	switch (type)
	{
	case PAULI_X_GATE:
	case PAULI_Y_GATE:
	case PAULI_Z_GATE:
	case HADAMARD_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case SWAP_GATE:
		//for these gate, dagger is equal to itself
		gate.setDagger(false);
		break;

	case X_HALF_PI:
	case Y_HALF_PI:
	case Z_HALF_PI:
	case T_GATE:
	case S_GATE:
	case I_GATE:
	case U4_GATE:
	case U1_GATE:
	case RX_GATE:
	case RY_GATE:
	case RZ_GATE:
	case ISWAP_GATE:
	case SQISWAP_GATE:
	case TWO_QUBIT_GATE:
	case CPHASE_GATE:
	case CU_GATE:
	break;

	default:
		QCERR("Unsupported GateNode");
		break;
	}
#endif
}

void PickUpNodes::pickQGateNode(const NodeIter cur_node_iter, QCircuitParam &cir_param)
{
	auto gate = QGate(std::dynamic_pointer_cast<AbstractQGateNode>(*cur_node_iter));
	auto tmp_gate = deepCopy(gate);
	tmp_gate.setDagger(gate.isDagger() ^ (cir_param.m_is_dagger));
	no_dagger_gate(tmp_gate);

	QVec control_qubits;
	gate.getControlVector(control_qubits);
	QVec increased_control_qubits = QCircuitParam::get_real_append_qubits(cir_param.m_control_qubits, control_qubits);

	tmp_gate.setControl(increased_control_qubits);

	if (!check_control_qubits(tmp_gate))
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Illegal control qubits.");
	}

	m_output_prog.pushBackNode(dynamic_pointer_cast<QNode>(tmp_gate.getImplementationPtr()));
	if (cur_node_iter == m_end_iter)
	{
		//On case of _startIter == _endIter
		m_b_pickup_end = true;
	}
}

bool PickUpNodes::check_control_qubits(QGate& gate)
{
	const auto gate_type = gate.getQGate()->getGateType();
	QVec gate_target_qubits;
	QVec control_qubits;
	gate.getControlVector(control_qubits);
	gate.getQuBitVector(gate_target_qubits);

	if (0 == control_qubits.size())
	{
		return true;
	}

	std::vector<int> target_qubits_val;
	std::vector<int> control_qubits_val;

	for (auto &itr : control_qubits)
	{
		control_qubits_val.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
	}

	switch ((GateType)(gate_type))
	{
	case ISWAP_THETA_GATE:
	case ISWAP_GATE:
	case SQISWAP_GATE:
	case SWAP_GATE:
		target_qubits_val.push_back(gate_target_qubits.front()->getPhysicalQubitPtr()->getQubitAddr());
		target_qubits_val.push_back(gate_target_qubits.back()->getPhysicalQubitPtr()->getQubitAddr());
		break;

	case CU_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case CPHASE_GATE:
	{
		target_qubits_val.push_back(gate_target_qubits.back()->getPhysicalQubitPtr()->getQubitAddr());
	}
	break;

	default:
		break;
	}

	/*auto sort_func = [](Qubit* a, Qubit* b) {
		return (a->getPhysicalQubitPtr()->getQubitAddr()) < (a->getPhysicalQubitPtr()->getQubitAddr());
	};*/
	std::sort(target_qubits_val.begin(), target_qubits_val.end());
	std::sort(control_qubits_val.begin(), control_qubits_val.end());

	std::vector<int> result_vec;
	set_intersection(target_qubits_val.begin(), target_qubits_val.end(), control_qubits_val.begin(), control_qubits_val.end(), std::back_inserter(result_vec));
	return (result_vec.size() == 0);
}

void PickUpNodes::pickQMeasureNode(const NodeIter cur_node_iter)
{
	if (m_b_pick_measure_node)
	{
		auto measure = QMeasure(std::dynamic_pointer_cast<AbstractQuantumMeasure>(*cur_node_iter));
		m_output_prog.pushBackNode(dynamic_pointer_cast<QNode>(deepCopy(measure).getImplementationPtr()));
		if (cur_node_iter == m_end_iter)
		{
			//On case for _startIter == _endIter
			m_b_pickup_end = true;
			return;
		}
	}
	else
	{
		m_b_pickup_end = true;
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Illegal Measure nodes.");
		m_output_prog.clear();
	}

	return;
}

void PickUpNodes::reverse_dagger_circuit()
{
	QProg tmp_prog;
	const auto head_iter = m_output_prog.getHeadNodeIter();
	auto aiter = m_output_prog.getLastNodeIter();
	while (head_iter != aiter)
	{
		auto gate = QGate(std::dynamic_pointer_cast<AbstractQGateNode>(*aiter));
		gate.setDagger(!(gate.isDagger()));
		no_dagger_gate(gate);
		tmp_prog.pushBackNode(dynamic_pointer_cast<QNode>(gate.getImplementationPtr()));
		--aiter;
	}

	m_output_prog = tmp_prog;
}

/*******************************************************************
*                      class QprogToMatrix
********************************************************************/
QprogToMatrix::MatrixOfOneLayer::MatrixOfOneLayer(SequenceLayer& layer, const QProgDAG& prog_dag, std::vector<int> &qubits_in_use)
	:m_qubits_in_use(qubits_in_use), m_mat_I{1, 0, 0, 1}
{
	for (auto &layer_item : layer)
	{
		auto p_node = prog_dag.get_vertex(layer_item.first.m_vertex_num);
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(p_node);
		QVec qubits_vector;
		p_gate->getQuBitVector(qubits_vector);
		QVec control_qubits_vector;
		p_gate->getControlVector(control_qubits_vector);
		if (control_qubits_vector.size() > 0)
		{
			qubits_vector.insert(qubits_vector.end(), control_qubits_vector.begin(), control_qubits_vector.end());
			std::sort(qubits_vector.begin(), qubits_vector.end(), [](Qubit* a, Qubit* b) {
				return a->getPhysicalQubitPtr()->getQubitAddr() < b->getPhysicalQubitPtr()->getQubitAddr(); 
			});

			std::vector<int> tmp_vec;
			tmp_vec.push_back(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
			tmp_vec.push_back(qubits_vector.back()->getPhysicalQubitPtr()->getQubitAddr());

			m_controled_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, tmp_vec));
			continue;
		}

		if (qubits_vector.size() == 2)
		{
			std::vector<int> quBits;
			for (auto _val : qubits_vector)
			{
				quBits.push_back(_val->getPhysicalQubitPtr()->getQubitAddr());
			}

			m_double_qubit_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, quBits));
		}
		else if (qubits_vector.size() == 1)
		{
			std::vector<int> quBits;
			quBits.push_back(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
			m_single_qubit_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, quBits));
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: QGate type error.");
		}
	}

	//sort by qubit address spacing
	auto sorfFun = [](gateAndQubitsItem_t &a, gateAndQubitsItem_t &b) { return (abs(a.second.front() - a.second.back())) < (abs(b.second.front() - b.second.back())); };
	std::sort(m_controled_gates.begin(), m_controled_gates.end(), sorfFun);
	std::sort(m_double_qubit_gates.begin(), m_double_qubit_gates.end(), sorfFun);
	std::sort(m_single_qubit_gates.begin(), m_single_qubit_gates.end(), sorfFun);
}

QStat QprogToMatrix::MatrixOfOneLayer::reverseCtrlGateMatrixCX(QStat& src_mat)
{
	init(QMachineType::CPU);
	auto q = qAllocMany(6);
	QGate gate_H = H(q[0]);
	QStat mat_H;
	gate_H.getQGate()->getMatrix(mat_H);
	finalize();

	QStat result_mat;
	QStat mat_of_zhang_multp_two_H = QPanda::tensor(mat_H, mat_H);

	result_mat = (mat_of_zhang_multp_two_H * src_mat);
	result_mat = (result_mat * mat_of_zhang_multp_two_H);

	PTrace("reverseCtrlGateMatrixCX: ");
	PTraceMat(result_mat);
	return result_mat;
}

QStat QprogToMatrix::MatrixOfOneLayer::reverseCtrlGateMatrixCU(QStat& src_mat)
{
	init(QMachineType::CPU);
	auto q = qAllocMany(6);
	QGate gate_swap = SWAP(q[0], q[1]);
	QStat mat_swap;
	gate_swap.getQGate()->getMatrix(mat_swap);
	finalize();

	QStat result_mat;

	result_mat = (mat_swap * src_mat);
	result_mat = (result_mat * mat_swap);

	PTrace("reverseCtrlGateMatrixCX: ");
	PTraceMat(result_mat);
	return result_mat;
}

void QprogToMatrix::MatrixOfOneLayer::merge_two_crossed_matrix(const calcUintItem_t& calc_unit_1, const calcUintItem_t& calc_unit_2, calcUintItem_t& result)
{
	int qubit_start = (calc_unit_1.second[0] < calc_unit_2.second[0]) ? calc_unit_1.second[0] : calc_unit_2.second[0];
	int qubit_end = (calc_unit_1.second[1] > calc_unit_2.second[1]) ? calc_unit_1.second[1] : calc_unit_2.second[1];
	QStat tensored_calc_unit_1;
	QStat tensored_calc_unit_2;

	auto tensor_func = [this](const size_t &qubit_index, const calcUintItem_t &calc_unit, QStat &tensor_result) {
		if (qubit_index < calc_unit.second[0])
		{
			tensorByMatrix(tensor_result, m_mat_I);
		}
		else if (qubit_index == calc_unit.second[0])
		{
			tensorByMatrix(tensor_result, calc_unit.first);
		}
		else if (qubit_index > calc_unit.second[1])
		{
			tensorByMatrix(tensor_result, m_mat_I);
		}
	};

	for (size_t i = qubit_start; i < qubit_end + 1; ++i)
	{
		tensor_func(i, calc_unit_1, tensored_calc_unit_1);
		tensor_func(i, calc_unit_2, tensored_calc_unit_2);
	}

	result.first = tensored_calc_unit_1 * tensored_calc_unit_2;
	result.second.push_back(qubit_start);
	result.second.push_back(qubit_end);
}

//return true on cross, or else return false
bool QprogToMatrix::MatrixOfOneLayer::check_cross_calc_unit(calcUnitVec_t& calc_unit_vec, calcUnitVec_t::iterator target_calc_unit_itr)
{
	const auto& target_calc_qubits = target_calc_unit_itr->second;
	for (auto itr_calc_unit = calc_unit_vec.begin(); itr_calc_unit < calc_unit_vec.end(); ++itr_calc_unit)
	{
		if (((target_calc_qubits[0] > itr_calc_unit->second.front()) && (target_calc_qubits[0] < itr_calc_unit->second.back()))
			||
			((target_calc_qubits[1] > itr_calc_unit->second.front()) && (target_calc_qubits[1] < itr_calc_unit->second.back())))
		{
			//merge two crossed matrix
			calcUintItem_t merge_result_calc_unit;
			merge_two_crossed_matrix(*itr_calc_unit, *target_calc_unit_itr, merge_result_calc_unit);

			itr_calc_unit->first.swap(merge_result_calc_unit.first);
			itr_calc_unit->second.swap(merge_result_calc_unit.second);

			return true;
		}
	}

	return false;
}

void QprogToMatrix::MatrixOfOneLayer::tensorByQGate(QStat& src_mat, std::shared_ptr<AbstractQGateNode> &pGate)
{
	if (nullptr == pGate)
	{
		return;
	}

	if (src_mat.empty())
	{
		pGate->getQGate()->getMatrix(src_mat);
		if (pGate->isDagger())
		{
			dagger(src_mat);
		}
	}
	else
	{
		QStat single_gate_mat;
		pGate->getQGate()->getMatrix(single_gate_mat);
		if (pGate->isDagger())
		{
			dagger(single_gate_mat);
		}
		src_mat = QPanda::tensor(src_mat, single_gate_mat);
	}
}

void QprogToMatrix::MatrixOfOneLayer::tensorByMatrix(QStat& src_mat, const QStat& tensor_mat)
{
	if (src_mat.empty())
	{
		src_mat = tensor_mat;
	}
	else
	{
		src_mat = QPanda::tensor(src_mat, tensor_mat);
	}
}

void QprogToMatrix::MatrixOfOneLayer::getStrideOverQubits(const std::vector<int> &qgate_used_qubits, std::vector<int> &stride_over_qubits)
{
	stride_over_qubits.clear();

	for (auto &qubit_val : m_qubits_in_use)
	{
		if ((qubit_val > qgate_used_qubits.front()) && (qubit_val < qgate_used_qubits.back()))
		{
			stride_over_qubits.push_back(qubit_val);
		}
	}
}

void QprogToMatrix::MatrixOfOneLayer::mergeToCalcUnit(std::vector<int>& qubits, QStat& gate_mat, calcUnitVec_t &calc_unit_vec, gateQubitInfo_t &single_qubit_gates)
{
	//auto &qubits = curGareItem.second;
	std::sort(qubits.begin(), qubits.end(), [](int &a, int &b) {return a < b; });
	std::vector<int> stride_over_qubits;
	getStrideOverQubits(qubits, stride_over_qubits);
	if (stride_over_qubits.empty())
	{
		//serial qubits
		calc_unit_vec.insert(calc_unit_vec.begin(), std::pair<QStat, std::vector<int>>(gate_mat, qubits));
	}
	else
	{
		//get crossed CalcUnits;
		calcUnitVec_t crossed_calc_units;
		for (auto itr_calc_unit = calc_unit_vec.begin(); itr_calc_unit < calc_unit_vec.end();)
		{
			if ((qubits[0] < itr_calc_unit->second.front()) && (qubits[1] > itr_calc_unit->second.back()))
			{
				/*if the current double qubit gate has crossed the itr_calc_unit, 
				  calc two crossed matrix, and replease the current itr_calc_unit
				*/
				check_cross_calc_unit(crossed_calc_units, itr_calc_unit);
				crossed_calc_units.push_back(*itr_calc_unit);

				itr_calc_unit = calc_unit_vec.erase(itr_calc_unit);
				continue;

			}

			++itr_calc_unit;
		}

		//get crossed SingleQubitGates;
		gateQubitInfo_t crossed_single_qubit_gates;
		for (auto itr_single_gate = single_qubit_gates.begin(); itr_single_gate < single_qubit_gates.end();)
		{
			const int qubit_val = itr_single_gate->second.front();
			if ((qubit_val > qubits[0]) && (qubit_val < qubits[1]))
			{
				crossed_single_qubit_gates.push_back(*itr_single_gate);

				itr_single_gate = single_qubit_gates.erase(itr_single_gate);
				continue;
			}

			++itr_single_gate;
		}

		//zhang multiply
		QStat filled_matrix;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			if (in_used_qubit_val > qubits[0])
			{
				if (in_used_qubit_val >= qubits[1])
				{
					break;
				}

				bool b_no_qGate_on_this_qubit = true;

				//find current qubit_val in crossed_single_qubit_gates
				std::shared_ptr<AbstractQGateNode> pGate;
				for (auto itr_crossed_single_gate = crossed_single_qubit_gates.begin();
					itr_crossed_single_gate != crossed_single_qubit_gates.end(); itr_crossed_single_gate++)
				{
					if (in_used_qubit_val == itr_crossed_single_gate->second.front())
					{
						b_no_qGate_on_this_qubit = false;

						tensorByQGate(filled_matrix, itr_crossed_single_gate->first);
						crossed_single_qubit_gates.erase(itr_crossed_single_gate);
						break;
					}
				}

				//find current qubit_val in crossed_calc_units
				for (auto itr_crossed_calc_unit = crossed_calc_units.begin();
					itr_crossed_calc_unit != crossed_calc_units.end(); itr_crossed_calc_unit++)
				{
					if ((in_used_qubit_val >= itr_crossed_calc_unit->second.front()) && (in_used_qubit_val <= itr_crossed_calc_unit->second.back()))
					{
						b_no_qGate_on_this_qubit = false;

						if (in_used_qubit_val == itr_crossed_calc_unit->second.front())
						{
							tensorByMatrix(filled_matrix, itr_crossed_calc_unit->first);
							//just break, CANN'T erase itr_crossed_calc_unit here
							break;
						}
					}
				}

				//No handle on this qubit
				if (b_no_qGate_on_this_qubit)
				{
					tensorByMatrix(filled_matrix, m_mat_I);
				}
			}
		}

		//blockMultip
		QStat filled_double_gate_matrix;
		blockedMatrix_t blocked_mat;
		partition(gate_mat, 2, 2, blocked_mat);
		blockMultip(filled_matrix, blocked_mat, filled_double_gate_matrix);

		//insert into calc_unit_vec
		calc_unit_vec.insert(calc_unit_vec.begin(), std::pair<QStat, std::vector<int>>(filled_double_gate_matrix, qubits));
	}
}

void  QprogToMatrix::MatrixOfOneLayer::merge_double_gate()
{
	GateType gate_T = GATE_UNDEFINED;
	for (auto &double_gate : m_double_qubit_gates)
	{
		QStat gate_mat;
		gate_T = (GateType)(double_gate.first->getQGate()->getGateType());
		if (2 == double_gate.second.size())
		{
			auto &qubits = double_gate.second;

			double_gate.first->getQGate()->getMatrix(gate_mat);

			if (qubits[0] > qubits[1])
			{
				if (CNOT_GATE == gate_T)
				{
					// transf base matrix
					auto transformed_mat = reverseCtrlGateMatrixCX(gate_mat);
					gate_mat.swap(transformed_mat);
				}
				else if (CU_GATE == gate_T)
				{
					auto transformed_mat = reverseCtrlGateMatrixCU(gate_mat);
					gate_mat.swap(transformed_mat);
				}
			}

			if (double_gate.first->isDagger())
			{
				dagger(gate_mat);
			}
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Qubits number error.");
		}

		mergeToCalcUnit(double_gate.second, gate_mat, m_calc_unit_vec, m_single_qubit_gates);
	}
}

void  QprogToMatrix::MatrixOfOneLayer::merge_calc_unit()
{
	for (auto &itr_calc_unit_vec : m_calc_unit_vec)
	{
		//calc all the qubits to get the final matrix
		QStat final_mat_of_one_calc_unit;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			bool b_no_gate_on_this_qubit = true;
			for (auto itr_single_gate = m_single_qubit_gates.begin(); itr_single_gate < m_single_qubit_gates.end();)
			{
				const int qubit_val = itr_single_gate->second.front();
				if (qubit_val == in_used_qubit_val)
				{
					b_no_gate_on_this_qubit = false;
					tensorByQGate(final_mat_of_one_calc_unit, itr_single_gate->first);

					itr_single_gate = m_single_qubit_gates.erase(itr_single_gate);
					continue;
				}

				++itr_single_gate;
			}

			if (itr_calc_unit_vec.second.front() == in_used_qubit_val)
			{
				b_no_gate_on_this_qubit = false;
				tensorByMatrix(final_mat_of_one_calc_unit, itr_calc_unit_vec.first);
			}

			if ((itr_calc_unit_vec.second.front() <= in_used_qubit_val) && (itr_calc_unit_vec.second.back() >= in_used_qubit_val))
			{
				continue;
			}

			if (b_no_gate_on_this_qubit)
			{
				tensorByMatrix(final_mat_of_one_calc_unit, m_mat_I);
			}
		}

		//Multiply, NOT tensor
		if (m_current_layer_mat.empty())
		{
			m_current_layer_mat = final_mat_of_one_calc_unit;
		}
		else
		{
			m_current_layer_mat = (m_current_layer_mat * final_mat_of_one_calc_unit);
		}
	}
}

void QprogToMatrix::MatrixOfOneLayer::reverse_ctrl_gate_matrix(QStat& src_mat, const GateType &gate_T)
{
	QStat result;
	switch (gate_T)
	{
	case CNOT_GATE:
		result = reverseCtrlGateMatrixCX(src_mat);
		break;

	case CU_GATE:
		result = reverseCtrlGateMatrixCU(src_mat);
		break;

	default:
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: reverse_ctrl_gate_matrix error, unsupport type.");
		break;
	}

	src_mat.swap(result);
}

void  QprogToMatrix::MatrixOfOneLayer::merge_controled_gate()
{
	if (m_controled_gates.size() == 0)
	{
		return;
	}

	GateType gate_T = GATE_UNDEFINED;
	for (auto& controled_gare : m_controled_gates)
	{
		gate_T = (GateType)(controled_gare.first->getQGate()->getGateType());
		QVec gate_qubits;
		controled_gare.first->getQuBitVector(gate_qubits);
		QVec control_gate_qubits;
		controled_gare.first->getControlVector(control_gate_qubits);
		
		//get base matrix
		QStat base_gate_mat;
		controled_gare.first->getQGate()->getMatrix(base_gate_mat);

		//build standard controled gate matrix
		std::vector<int> all_gate_qubits_vec;
		for (auto& itr : gate_qubits)
		{
			all_gate_qubits_vec.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
		}

		for (auto& itr : control_gate_qubits)
		{
			all_gate_qubits_vec.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
		}

		sort(all_gate_qubits_vec.begin(), all_gate_qubits_vec.end(), [](const int &a, const int &b) {return a < b; });
		all_gate_qubits_vec.erase(unique(all_gate_qubits_vec.begin(), all_gate_qubits_vec.end()), all_gate_qubits_vec.end());

		int all_control_gate_qubits = all_gate_qubits_vec.size();
		QStat standard_mat;
		build_standard_control_gate_matrix(base_gate_mat, all_control_gate_qubits, standard_mat);

		//tensor
		//int all_used_qubits = controled_gare.second[1] - controled_gare.second[0] + 1;
		int idle_qubits = m_qubits_in_use.size() - all_control_gate_qubits;
		QStat tmp_idle_mat;
		if (idle_qubits > 0)
		{
			for (size_t i = 0; i < idle_qubits; i++)
			{
				tensorByMatrix(tmp_idle_mat, m_mat_I);
			}
			standard_mat = tensor(tmp_idle_mat, standard_mat);
		}

#if PRINT_TRACE
		cout << "tmp_idle_mat:" << endl;
		cout << tmp_idle_mat << endl;

		cout << "tensored standard matrix:" << endl;
		cout << standard_mat << endl;
#endif // PRINT_TRACE

		//swap
		auto used_qubit_iter = m_qubits_in_use.begin() + idle_qubits;
		auto gate_qubit_item = all_gate_qubits_vec.begin();
		for (; gate_qubit_item != all_gate_qubits_vec.end(); ++gate_qubit_item, ++used_qubit_iter)
		{
			const auto& gate_qubit_tmp = *gate_qubit_item;
			const auto& maped_gate_qubit = *used_qubit_iter;
			if (gate_qubit_tmp != maped_gate_qubit)
			{
				swap_two_qubit_on_matrix(standard_mat, controled_gare.second[0], controled_gare.second[1], gate_qubit_tmp, maped_gate_qubit);
			}
		}

		//swap target qubit by gate type
		if ((SWAP_GATE == gate_T) || (SQISWAP_GATE == gate_T)|| (ISWAP_GATE == gate_T)|| (ISWAP_THETA_GATE == gate_T))
		{
			swap_two_qubit_on_matrix(standard_mat, controled_gare.second[0], controled_gare.second[1], 
				gate_qubits.front()->getPhysicalQubitPtr()->getQubitAddr(), all_gate_qubits_vec.back() -1);
		}
		
		swap_two_qubit_on_matrix(standard_mat, controled_gare.second[0], controled_gare.second[1],
			gate_qubits.back()->getPhysicalQubitPtr()->getQubitAddr(), all_gate_qubits_vec.back());

		//merge to current layer matrix directly
		if (m_current_layer_mat.empty())
		{
			m_current_layer_mat = standard_mat;
		}
		else
		{
			m_current_layer_mat = (m_current_layer_mat * standard_mat);
		}
	}
}

void QprogToMatrix::MatrixOfOneLayer::swap_two_qubit_on_matrix(QStat& src_mat, const int mat_qubit_start, const int mat_qubit_end, const int qubit_1, const int qubit_2)
{
	if (qubit_1 == qubit_2)
	{
		return;
	}

	auto machine = initQuantumMachine(QMachineType::CPU);
	auto q = machine->allocateQubits(4);
	auto c = machine->allocateCBits(4);
	auto swap_gate = SWAP(q[0], q[1]);
	QStat swap_gate_matrix;
	swap_gate.getQGate()->getMatrix(swap_gate_matrix);
	destroyQuantumMachine(machine);

	QStat tmp_tensor_mat;
	int tensor_start_qubit = qubit_1 < qubit_2 ? qubit_1 : qubit_2;
	int tensor_end_qubit = qubit_1 < qubit_2 ? qubit_2 : qubit_1;

	for (auto &used_qubit_item : m_qubits_in_use)
	{
		if ((used_qubit_item > tensor_start_qubit) && (used_qubit_item < tensor_end_qubit))
		{
			tensorByMatrix(tmp_tensor_mat, m_mat_I);
		}
	}

#if PRINT_TRACE
	cout << "tmp_tensor_mat:" << endl;
	cout << tmp_tensor_mat << endl;
#endif // PRINT_TRACE

	//blockMultip
	QStat tensored_swap_gate_matrix;
	blockedMatrix_t blocked_mat;
	partition(swap_gate_matrix, 2, 2, blocked_mat);
	blockMultip(tmp_tensor_mat, blocked_mat, tensored_swap_gate_matrix);

	QStat tmp_mat;
	for (auto used_qubit_itr = m_qubits_in_use.begin(); used_qubit_itr != m_qubits_in_use.end(); ++used_qubit_itr)
	{
		const auto& qubit_tmp = *used_qubit_itr;
		if ((qubit_tmp < qubit_1) || (qubit_tmp > qubit_2))
		{
			tensorByMatrix(tmp_mat, m_mat_I);
		}
		else if (qubit_1 == qubit_tmp)
		{
			tensorByMatrix(tmp_mat, tensored_swap_gate_matrix);
		}
	}

	src_mat = src_mat * tmp_mat;
}

void QprogToMatrix::MatrixOfOneLayer::build_standard_control_gate_matrix(const QStat& src_mat, const int qubit_number, QStat& result_mat)
{
	size_t rows = 1; // rows of the standard matrix
	size_t columns = 1;// columns of the standard matrix
	for (size_t i = 0; i < qubit_number; i++)
	{
		rows *= 2;
	}
	columns = rows;

	result_mat.resize(rows * columns);

	size_t src_mat_colums = sqrt(src_mat.size());
	size_t src_mat_rows = src_mat_colums;
	size_t item_index = 0;
	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			item_index = i * rows + j;
			if (((rows - i) <= src_mat_rows) && ((columns - j) <= src_mat_colums))
			{
				result_mat[item_index] = src_mat[(src_mat_rows - (rows - i)) * src_mat_rows + src_mat_colums - (columns - j)];
			}
			else if (i == j)
			{
				result_mat[item_index] = 1;
			}
			else
			{
				result_mat[item_index] = 0;
			}
		}
	}

#if PRINT_TRACE
	cout << result_mat << endl;
#endif // PRINT_TRACE
}

void QprogToMatrix::MatrixOfOneLayer::merge_sing_gate()
{
	if (m_single_qubit_gates.size() > 0)
	{
		QStat all_single_gate_matrix;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			bool b_no_gate_on_this_qubit = true;
			for (auto itr_single_gate = m_single_qubit_gates.begin(); itr_single_gate != m_single_qubit_gates.end();)
			{
				const int qubit_val = itr_single_gate->second.front();
				if (qubit_val == in_used_qubit_val)
				{
					b_no_gate_on_this_qubit = false;
					tensorByQGate(all_single_gate_matrix, itr_single_gate->first);

					itr_single_gate = m_single_qubit_gates.erase(itr_single_gate);
					continue;
				}

				++itr_single_gate;
			}

			if (b_no_gate_on_this_qubit)
			{
				tensorByMatrix(all_single_gate_matrix, m_mat_I);
			}
		}

		if (m_current_layer_mat.empty())
		{
			m_current_layer_mat = all_single_gate_matrix;
		}
		else
		{
			m_current_layer_mat = (m_current_layer_mat * all_single_gate_matrix);
		}
	}
}

QStat QprogToMatrix::getMatrix()
{
	QStat result_matrix;

	//get quantumBits number
	get_all_used_qubits(m_prog, m_qubits_in_use);

	//layer
	GraphMatch match;
	TopologicalSequence seq;
	match.get_topological_sequence(m_prog, seq);
	const QProgDAG& prog_dag = match.getProgDAG();
	for (auto &seqItem : seq)
	{
		//each layer
		if (result_matrix.size() == 0)
		{
			result_matrix = getMatrixOfOneLayer(seqItem, prog_dag);
		}
		else
		{
			result_matrix = result_matrix * (getMatrixOfOneLayer(seqItem, prog_dag));
		}
	}

	return result_matrix;
}

QStat QprogToMatrix::getMatrixOfOneLayer(SequenceLayer& layer, const QProgDAG& prog_dag)
{
	MatrixOfOneLayer get_one_layer_matrix(layer, prog_dag, m_qubits_in_use);

	get_one_layer_matrix.merge_controled_gate();
	
	get_one_layer_matrix.merge_double_gate();	

	get_one_layer_matrix.merge_calc_unit();

	get_one_layer_matrix.merge_sing_gate();

	return get_one_layer_matrix.m_current_layer_mat;
}

/*******************************************************************
*                      public interface
********************************************************************/
QStat QPanda::getCircuitMatrix(QProg srcProg, const NodeIter nodeItrStart, const NodeIter nodeItrEnd)
{
	QProg tmp_prog;

	pickUpNode(tmp_prog, srcProg, nodeItrStart == NodeIter() ? srcProg.getFirstNodeIter() : nodeItrStart,
		nodeItrEnd == NodeIter() ? srcProg.getEndNodeIter() : nodeItrEnd);

#if PRINT_TRACE
	cout << "got the target tmp-prog:" << endl;
	printAllNodeType(tmp_prog);
#endif

	QprogToMatrix calc_matrix(tmp_prog);

	return calc_matrix.getMatrix();
}

std::string QPanda::getAdjacentQGateType(QProg &prog, NodeIter &nodeItr, std::vector<NodeIter>& frontAndBackIter)
{
	std::shared_ptr<AdjacentQGates> p_adjacent_QGates = std::make_shared<AdjacentQGates>(prog, nodeItr);
	if (nullptr == p_adjacent_QGates)
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Failed to create adjacent object, memory error.");
		return std::string("Error");
	}

	//Judging whether the target nodeItr is Qgate or no
	if ((GATE_UNDEFINED == p_adjacent_QGates->getItrNodeType(nodeItr)))
	{
		// target node type error
		QCERR_AND_THROW_ERRSTR(runtime_error, "The target node is not a Qgate.");
		return std::string("Error");
	}

	p_adjacent_QGates->traverse_qprog();

	frontAndBackIter.clear();
	frontAndBackIter.push_back(p_adjacent_QGates->getFrontIter());
	frontAndBackIter.push_back(p_adjacent_QGates->getBackIter());

	std::string ret = std::string("frontNodeType = ") + p_adjacent_QGates->getFrontIterNodeTypeStr()
		+ std::string(", backNodeType = ") + p_adjacent_QGates->getBackIterNodeTypeStr();

	return ret;
}

bool QPanda::isSwappable(QProg &prog, NodeIter &nodeItr1, NodeIter &nodeItr2)
{
	if (nodeItr1 == nodeItr2)
	{
		QCERR("Error: the two nodeIter is equivalent.");
		return false;
	}

	std::shared_ptr<JudgeTwoNodeIterIsSwappable> p_judge_node_iters = std::make_shared<JudgeTwoNodeIterIsSwappable>(prog, nodeItr1, nodeItr2);
	if (nullptr == p_judge_node_iters.get())
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Failed to create JudgeNodeIter object, memory error.");
		return false;
	}

	p_judge_node_iters->traverse_qprog();

	return p_judge_node_iters->getResult();
}

bool QPanda::isMatchTopology(const QGate& gate, const std::vector<std::vector<int>>& vecTopoSt)
{
	if (0 == vecTopoSt.size())
	{
		return false;
	}
	QVec vec_qubits;
	gate.getQuBitVector(vec_qubits);

	size_t first_qubit_pos = vec_qubits.front()->getPhysicalQubitPtr()->getQubitAddr();
	if (vecTopoSt.size() <= first_qubit_pos)
	{
		return false;
	}

	int pos_in_topology = first_qubit_pos; //the index of qubits in topological structure is start from 1.
	std::vector<int> vec_topology = vecTopoSt[pos_in_topology];
	for (auto iter = ++(vec_qubits.begin()); iter != vec_qubits.end(); ++iter)
	{
		auto target_qubit = (*iter)->getPhysicalQubitPtr()->getQubitAddr();
		if (vecTopoSt.size() <= target_qubit)
		{
			//cout << target_qubit << endl;
			return false;
		}
		if (0 == vec_topology[target_qubit])
		{
			return false;
		}
	}
	return true;
}

bool QPanda::isSupportedGateType(const NodeIter &nodeItr)
{
	//read meta data
	QuantumMetadata meta_data;
	std::vector<std::string> vec_single_gates;
	std::vector<string> vec_double_gates;
	meta_data.getQGate(vec_single_gates, vec_double_gates);

	//judge
	string gate_type_str;
	NodeType tmp_node_type = (*nodeItr)->getNodeType();
	if (GATE_NODE == tmp_node_type)
	{
		std::shared_ptr<OriginQGate> gate = std::dynamic_pointer_cast<OriginQGate>(*nodeItr);
		gate_type_str = TransformQGateType::getInstance()[(GateType)(gate->getQGate()->getGateType())];
	}
	else
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: The target node is NOT a QGate.");
		return false;
	}

	std::transform(gate_type_str.begin(), gate_type_str.end(), gate_type_str.begin(), ::tolower);
	for (auto itr : vec_single_gates)
	{
		std::transform(itr.begin(), itr.end(), itr.begin(), ::tolower);
		if (0 == strcmp(gate_type_str.c_str(), itr.c_str()))
		{
			return true;
		}
	}

	for (auto itr : vec_double_gates)
	{
		std::transform(itr.begin(), itr.end(), itr.begin(), ::tolower);
		if (0 == strcmp(gate_type_str.c_str(), itr.c_str()))
		{
			return true;
		}
	}

	return false;
}

void QPanda::pickUpNode(QProg &outPutProg, QProg &srcProg, const NodeIter nodeItrStart/* = NodeIter()*/, const NodeIter nodeItrEnd /*= NodeIter()*/,
	bool bPickMeasure/* = false*/, bool bDagger/* = false*/)
{
	//fill the prog through traversal 
	PickUpNodes pick_handle(outPutProg, srcProg,
		nodeItrStart == NodeIter() ? srcProg.getFirstNodeIter() : nodeItrStart,
		nodeItrEnd == NodeIter() ? srcProg.getEndNodeIter() : nodeItrEnd);

	pick_handle.setPickUpMeasureNode(bPickMeasure);
	pick_handle.setDaggerFlag(bDagger);

	pick_handle.traverse_qprog();

	if (bDagger)
	{
		//reverse outPutProg
		pick_handle.reverse_dagger_circuit();
	}
}

void QPanda::get_all_used_qubits(QProg &prog, QVec &vecQuBitsInUse)
{
	vecQuBitsInUse.clear();
	NodeIter itr = prog.getFirstNodeIter();
	NodeIter itr_end = prog.getEndNodeIter();
	if (itr == itr_end)
	{
		return;
	}

	QVec qubits_vector;
	QVec vec_control_qubits;
	NodeType type = NODE_UNDEFINED;
	do
	{
		type = (*itr)->getNodeType();
		if (GATE_NODE != type)
		{
			if (MEASURE_GATE == type)
			{
				std::shared_ptr<AbstractQuantumMeasure> p_QMeasure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*itr);
				vecQuBitsInUse.push_back(p_QMeasure->getQuBit());
			}
			continue;
		}

		std::shared_ptr<AbstractQGateNode> p_QGate = std::dynamic_pointer_cast<AbstractQGateNode>(*itr);
		qubits_vector.clear();
		p_QGate->getQuBitVector(qubits_vector);

		vec_control_qubits.clear();
		p_QGate->getControlVector(vec_control_qubits);
		for (auto _val : qubits_vector)
		{
			vecQuBitsInUse.push_back(_val);
		}

		for (auto _val : vec_control_qubits)
		{
			vecQuBitsInUse.push_back(_val);
		}

	} while ((++itr) != itr_end);
	sort(vecQuBitsInUse.begin(), vecQuBitsInUse.end());
	vecQuBitsInUse.erase(unique(vecQuBitsInUse.begin(), vecQuBitsInUse.end()), vecQuBitsInUse.end());
}

void QPanda::get_all_used_qubits(QProg &prog, std::vector<int> &vecQuBitsInUse)
{
	vecQuBitsInUse.clear();
	QVec vec_all_qubits;
	get_all_used_qubits(prog, vec_all_qubits);
	for (auto &itr : vec_all_qubits)
	{
		vecQuBitsInUse.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
	}

	sort(vecQuBitsInUse.begin(), vecQuBitsInUse.end(), [](const int &a, const int& b) {return a < b; });
	vecQuBitsInUse.erase(unique(vecQuBitsInUse.begin(), vecQuBitsInUse.end()), vecQuBitsInUse.end());
}

void QPanda::get_all_used_class_bits(QProg &prog, std::vector<int> &vecClBitsInUse)
{
	vecClBitsInUse.clear();
	NodeIter itr = prog.getFirstNodeIter();
	NodeIter itr_end = prog.getEndNodeIter();
	if (itr == itr_end)
	{
		return;
	}

	QVec qubits_vector;
	NodeType type = NODE_UNDEFINED;

	do
	{
		type = (*itr)->getNodeType();
		if (MEASURE_GATE == type)
		{
			std::shared_ptr<AbstractQuantumMeasure> p_QMeasure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*itr);
			vecClBitsInUse.push_back(p_QMeasure->getCBit()->getValue());
		}
	} while ((++itr) != itr_end);

	sort(vecClBitsInUse.begin(), vecClBitsInUse.end());
	vecClBitsInUse.erase(unique(vecClBitsInUse.begin(), vecClBitsInUse.end()), vecClBitsInUse.end());
}

string QPanda::printAllNodeType(QProg &prog)
{
	GetAllNodeType print_node_type(prog);
	print_node_type.traverse_qprog();
	return print_node_type.printNodesType();
}

void QPanda::get_gate_parameter(std::shared_ptr<AbstractQGateNode> pGate, std::string& para_str)
{
	GateType tmpType = (GateType)(pGate->getQGate()->getGateType());

	switch (tmpType)
	{
	case PAULI_X_GATE:
	case PAULI_Y_GATE:
	case PAULI_Z_GATE:
	case X_HALF_PI:
	case Y_HALF_PI:
	case Z_HALF_PI:
	case HADAMARD_GATE:
	case T_GATE:
	case S_GATE:
	case I_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case ISWAP_GATE:
	case SQISWAP_GATE:
	case SWAP_GATE:
	case TWO_QUBIT_GATE:
		break;

	case U4_GATE:
	{
		QGATE_SPACE::U4 *u4gate = dynamic_cast<QGATE_SPACE::U4*>(pGate->getQGate());
		para_str.append(string("(") + to_string(u4gate->getAlpha())
			+ "," + to_string(u4gate->getBeta())
			+ "," + to_string(u4gate->getGamma())
			+ "," + to_string(u4gate->getDelta())
			+ ")");
	}
	break;

	case U1_GATE:
	case RX_GATE:
	case RY_GATE:
	case RZ_GATE:
	case CPHASE_GATE:
	{
		auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(pGate->getQGate());
		string  gate_angle = to_string(gate_parameter->getParameter());
		para_str.append(string("(") + gate_angle + ")");
	}
	break;

	case CU_GATE:
	{
		QuantumGate * gate_parameter = dynamic_cast<QuantumGate *>(pGate->getQGate());
		auto angle = dynamic_cast<QGATE_SPACE::AbstractAngleParameter *>(gate_parameter);
		string gate_four_theta = to_string(angle->getAlpha()) + ',' +
			to_string(angle->getBeta()) + ',' +
			to_string(angle->getGamma()) + ',' +
			to_string(angle->getDelta());
		para_str.append(string("(") + gate_four_theta + ")");
	}
	break;

	default:
		QCERR("Unsupported GateNode");
		break;
	}
}