/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgInfo/JudgeTwoNodeIterIsSwappable.h"
#include "Core/Utilities/QProgInfo/GetAdjacentNodes.h"
#include "Core/Utilities/QProgInfo/QProgToMatrix.h"
#include "Core/Utilities/QProgInfo/GetAllUsedQubitAndCBit.h"

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
*                      class NodeInfo
********************************************************************/
void NodeInfo::init(const int type, const QVec& target_qubits, const QVec& control_qubits)
{
	if (type < 0)
	{
		switch (type)
		{
		case -1:
		{
			auto p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*m_iter);
			m_cbits.push_back(p_measure->getCBit()->getValue());
			m_node_type = MEASURE_GATE;
		}
		break;

		case -2:
			m_node_type = RESET_NODE;
			break;

		default:
			std::cerr << "Node-tpye:" << type << std::endl;
			QCERR_AND_THROW_ERRSTR(init_fail, "Error: Node-type error.");
			break;
		}
	}
	else
	{
		m_gate_type = (GateType)type;
		m_name = TransformQGateType::getInstance()[m_gate_type];
		if (m_is_dagger)
		{
			m_name += ".dag";
		}
		m_params = get_gate_parameter(std::dynamic_pointer_cast<AbstractQGateNode>(*m_iter));
	}
}

void NodeInfo::reset() 
{
	m_iter = NodeIter();
	m_node_type = NODE_UNDEFINED;
	m_gate_type = GATE_UNDEFINED;
	m_is_dagger = false;
	m_target_qubits.clear();
	m_control_qubits.clear();
	m_params.clear();
	m_name = "";
}

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
		QCERR_AND_THROW_ERRSTR(init_fail, "Current prog-node is empty.");
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

void TraverseByNodeIter::traverse_qprog(QProg prog)
{
	NodeIter itr = NodeIter();
	auto param = std::make_shared<QCircuitParam>();
	execute(prog.getImplementationPtr(), nullptr, *param, itr);
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

void GetAllNodeType::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle reset node
	sub_circuit_indent();

	//std::string gateTypeStr;
	char reset_buf[258] = "";
	snprintf(reset_buf, 256, "<<Reset(q[%d])", cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
	m_output_str.append(reset_buf);
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

void PickUpNodes::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle reset node
	pickUp(cur_node_iter, std::mem_fn(&PickUpNodes::pickQResetNode), this, cur_node_iter);
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
		//QCERR("Unsupported GateNode");
		break;
	}
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

	//handle control gate
	switch ((GateType)(gate.getQGate()->getGateType()))
	{
	case CU_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case CPHASE_GATE:
	{
		QVec gate_qubits;
		gate.getQuBitVector(gate_qubits);
		const auto self_control_qubit = gate_qubits.front()->getPhysicalQubitPtr()->getQubitAddr();
		for (auto itr = increased_control_qubits.begin(); itr != increased_control_qubits.end(); ++itr)
		{
			if (self_control_qubit == (*itr)->getPhysicalQubitPtr()->getQubitAddr())
			{
				increased_control_qubits.erase(itr);
				break;
			}
		}
	}
	break;

	default:
		break;
	}

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

	for (auto &itr : gate_target_qubits)
	{
		target_qubits_val.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
	}

	switch ((GateType)(gate_type))
	{
	case CU_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case CPHASE_GATE:
	{
		target_qubits_val.front() = target_qubits_val.back();
		target_qubits_val.pop_back();
	}
	break;

	default:
		break;
	}

	std::sort(target_qubits_val.begin(), target_qubits_val.end());
	std::sort(control_qubits_val.begin(), control_qubits_val.end());

	std::vector<int> result_vec;
	set_intersection(target_qubits_val.begin(), target_qubits_val.end(), control_qubits_val.begin(), control_qubits_val.end(), std::back_inserter(result_vec));
	return (result_vec.size() == 0);
}

void PickUpNodes::pickQMeasureNode(const NodeIter cur_node_iter)
{
	if (is_valid_pick_up_node_type(MEASURE_GATE))
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

void PickUpNodes::pickQResetNode(const NodeIter cur_node_iter)
{
	if (is_valid_pick_up_node_type(RESET_NODE))
	{
		auto reset = QReset(std::dynamic_pointer_cast<AbstractQuantumReset>(*cur_node_iter));
		m_output_prog.pushBackNode(dynamic_pointer_cast<QNode>(deepCopy(reset).getImplementationPtr()));
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
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Illegal reset nodes.");
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
*                      public interface
********************************************************************/
QStat QPanda::getCircuitMatrix(QProg srcProg, const bool b_bid_endian /*= false*/, const NodeIter nodeItrStart, const NodeIter nodeItrEnd)
{
	QProg tmp_prog;

	pickUpNode(tmp_prog, srcProg, {MEASURE_GATE, RESET_NODE}, nodeItrStart == NodeIter() ? srcProg.getFirstNodeIter() : nodeItrStart,
		nodeItrEnd == NodeIter() ? srcProg.getEndNodeIter() : nodeItrEnd);

#if PRINT_TRACE
	cout << "got the target tmp-prog:" << endl;
	printAllNodeType(tmp_prog);
#endif

	QProgToMatrix calc_matrix(tmp_prog, b_bid_endian);

	return calc_matrix.get_matrix();
}

std::string QPanda::getAdjacentQGateType(QProg prog, NodeIter &nodeItr, std::vector<NodeInfo>& adjacentNodes)
{
	std::shared_ptr<AdjacentQGates> p_adjacent_QGates = std::make_shared<AdjacentQGates>(prog, nodeItr);
	if (nullptr == p_adjacent_QGates)
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Failed to create adjacent object, memory error.");
		return std::string("Error");
	}

	//Judging whether the target nodeItr is Qgate or no
	if ((GATE_UNDEFINED == p_adjacent_QGates->get_node_ype(nodeItr)))
	{
		// target node type error
		QCERR_AND_THROW_ERRSTR(runtime_error, "The target node is not a Qgate.");
		return std::string("Error");
	}

	p_adjacent_QGates->traverse_qprog();

	adjacentNodes.clear();
	adjacentNodes.push_back(p_adjacent_QGates->get_front_node());
	adjacentNodes.push_back(p_adjacent_QGates->get_back_node());

	std::string ret = std::string("frontNodeType = ") + p_adjacent_QGates->get_front_node_type_str()
		+ std::string(", backNodeType = ") + p_adjacent_QGates->get_back_node_type_str();

	return ret;
}

bool QPanda::isSwappable(QProg prog, NodeIter &nodeItr1, NodeIter &nodeItr2)
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

	// judge node type
	if (!p_judge_node_iters->judge_node_type())
	{
		return false;
	}

	p_judge_node_iters->traverse_qprog();

	return p_judge_node_iters->get_result();
}

bool QPanda::isMatchTopology(const QGate& gate, const std::vector<std::vector<double>>& vecTopoSt)
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
	std::vector<double> vec_topology = vecTopoSt[pos_in_topology];
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

void QPanda::pickUpNode(QProg &outPutProg, QProg srcProg, const std::vector<NodeType> reject_node_types, 
	const NodeIter nodeItrStart/* = NodeIter()*/, const NodeIter nodeItrEnd /*= NodeIter()*/,
	bool bDagger/* = false*/)
{
	//fill the prog through traversal 
	PickUpNodes pick_handle(outPutProg, srcProg, reject_node_types,
		nodeItrStart == NodeIter() ? srcProg.getFirstNodeIter() : nodeItrStart,
		nodeItrEnd == NodeIter() ? srcProg.getEndNodeIter() : nodeItrEnd);

	pick_handle.setDaggerFlag(bDagger);

	pick_handle.traverse_qprog();

	if (bDagger)
	{
		//reverse outPutProg
		pick_handle.reverse_dagger_circuit();
	}
}

size_t QPanda::get_all_used_qubits(QProg prog, QVec &vecQuBitsInUse)
{
	vecQuBitsInUse.clear();

	GetAllUsedQubitAndCBit get_qubit_object;
	get_qubit_object.traversal(prog);

	auto qubit_vec = get_qubit_object.get_used_qubits();
	for (auto &i : qubit_vec)
	{
		vecQuBitsInUse.push_back(i);
	}

	sort(vecQuBitsInUse.begin(), vecQuBitsInUse.end(), [](Qubit* a, Qubit* b) { 
		return a->getPhysicalQubitPtr()->getQubitAddr() < b->getPhysicalQubitPtr()->getQubitAddr(); });
	vecQuBitsInUse.erase(unique(vecQuBitsInUse.begin(), vecQuBitsInUse.end()), vecQuBitsInUse.end());

	return vecQuBitsInUse.size();
}

size_t QPanda::get_all_used_qubits(QProg prog, std::vector<int> &vecQuBitsInUse)
{
	vecQuBitsInUse.clear();
	QVec vec_all_qubits;
	get_all_used_qubits(prog, vec_all_qubits);
	for (auto &itr : vec_all_qubits)
	{
		vecQuBitsInUse.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
	}

	return vecQuBitsInUse.size();
}

size_t QPanda::get_all_used_class_bits(QProg prog, std::vector<int> &vecClBitsInUse)
{
	vecClBitsInUse.clear();

	GetAllUsedQubitAndCBit get_cbit_object;
	get_cbit_object.traversal(prog);

	auto cbit_vec = get_cbit_object.get_used_cbits();
	for (auto i : cbit_vec)
	{
		vecClBitsInUse.push_back(i);
	}

	sort(vecClBitsInUse.begin(), vecClBitsInUse.end(), [](const int& a, const int& b) { return a < b; });
	vecClBitsInUse.erase(unique(vecClBitsInUse.begin(), vecClBitsInUse.end()), vecClBitsInUse.end());

	return vecClBitsInUse.size();
}

string QPanda::printAllNodeType(QProg prog)
{
	GetAllNodeType print_node_type;
	print_node_type.traverse_qprog(prog);
	return print_node_type.printNodesType();
}

std::vector<double> QPanda::get_gate_parameter(std::shared_ptr<AbstractQGateNode> pGate)
{
	GateType tmpType = (GateType)(pGate->getQGate()->getGateType());
	std::vector<double> params;
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
	case ECHO_GATE:
	case BARRIER_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case SWAP_GATE:
	case TWO_QUBIT_GATE:
	case ISWAP_GATE:
	case SQISWAP_GATE:
		break;

	case ISWAP_THETA_GATE:
	{
		auto single_angle_gate = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(pGate->getQGate());
		params.push_back(single_angle_gate->getParameter());
	}
		break;

	case U4_GATE:
	{
		QGATE_SPACE::U4 *u4gate = dynamic_cast<QGATE_SPACE::U4*>(pGate->getQGate());
		params.push_back(u4gate->getAlpha());
		params.push_back(u4gate->getBeta());
		params.push_back(u4gate->getGamma());
		params.push_back(u4gate->getDelta());
	}
	break;

	case RPHI_GATE:
	{
		QGATE_SPACE::RPhi *rphi_gate = dynamic_cast<QGATE_SPACE::RPhi*>(pGate->getQGate());
		params.push_back(rphi_gate->getBeta());
		params.push_back(rphi_gate->get_phi());
	}
	break;

	case U1_GATE:
	case RX_GATE:
	case RY_GATE:
	case RZ_GATE:
	case CPHASE_GATE:
	{
		auto single_angle_gate = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(pGate->getQGate());
		params.push_back(single_angle_gate->getParameter());
	}
	break;

	case U2_GATE:
	{
		QGATE_SPACE::U2 *u2_gate = dynamic_cast<QGATE_SPACE::U2*>(pGate->getQGate());
		params.push_back(u2_gate->get_phi());
		params.push_back(u2_gate->get_lambda());
	}
	break;

	case U3_GATE:
	{
		QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>(pGate->getQGate());
		params.push_back(u3_gate->get_theta());
		params.push_back(u3_gate->get_phi());
		params.push_back(u3_gate->get_lambda());
	}
	break;

	case CU_GATE:
	{
		QuantumGate * gate_parameter = dynamic_cast<QuantumGate *>(pGate->getQGate());
		auto angle = dynamic_cast<QGATE_SPACE::AbstractAngleParameter *>(gate_parameter);
		params.push_back(angle->getAlpha());
		params.push_back(angle->getBeta());
		params.push_back(angle->getGamma());
		params.push_back(angle->getDelta());
	}
	break;

	default:
		QCERR("Unsupported GateNode");
		std::cerr << "unsupport gate node : " << tmpType << std::endl;
		break;
	}

	return params;
}

void QPanda::get_gate_parameter(std::shared_ptr<AbstractQGateNode> pGate, std::string& para_str)
{
	std::vector<double> param_vec = get_gate_parameter(pGate);
	if (param_vec.size() == 0)
	{
		return;
	}

	for (size_t i = 0; i < param_vec.size(); ++i)
	{
		if (0 == i)
		{
			para_str.append(string("(") + to_string(param_vec[i]));
		}
		else
		{
			para_str.append("," + to_string(param_vec[i]));
		}
	}
	para_str += ")";
}

bool QPanda::check_dagger(std::shared_ptr<AbstractQGateNode> p_gate, const bool& b_dagger)
{
	switch (p_gate->getQGate()->getGateType())
	{
	case PAULI_X_GATE:
	case PAULI_Y_GATE:
	case PAULI_Z_GATE:
	case HADAMARD_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case SWAP_GATE:
	case I_GATE:
		return false;
		break;

	default:
		break;
	}

	return b_dagger;
}
