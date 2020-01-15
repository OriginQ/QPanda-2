#include "Core/Utilities/QProgInfo/JudgeTwoNodeIterIsSwappable.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

USING_QPANDA
using namespace std;

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#else
#define PTrace
#endif

/**
* @brief Judge whether the prog is related to the target qubits
* @ingroup QProgInfo
*/
class JudgeProgOperateQubts : public TraverseByNodeIter
{
public:
	JudgeProgOperateQubts(QProg &prog, const std::vector<int>& qubits)
		:TraverseByNodeIter(prog), m_qubits(qubits), m_is_related_to_qubits(false)
	{}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		//handle QGate node
		if (m_is_related_to_qubits) return;

		QVec gate_qubits;
		cur_node->getQuBitVector(gate_qubits);
		cur_node->getControlVector(gate_qubits);
		gate_qubits.insert(gate_qubits.end(), cir_param.m_control_qubits.begin(), cir_param.m_control_qubits.end());
		int qubit_val = -1;
		for (auto& gate_qubit_iter : gate_qubits)
		{
			for (const auto& item : m_qubits)
			{
				if (gate_qubit_iter->getPhysicalQubitPtr()->getQubitAddr() == item)
				{
					m_is_related_to_qubits = true;
					return;
				}
			}
		}
	}

	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		//handle measure node
		if (m_is_related_to_qubits) return;

		for (const auto& item : m_qubits)
		{
			if (cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr() == item)
			{
				m_is_related_to_qubits = true;
				return;
			}
		}
	}

	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		//handle reset node
		if (m_is_related_to_qubits) return;

		for (const auto& item : m_qubits)
		{
			if (cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr() == item)
			{
				m_is_related_to_qubits = true;
				return;
			}
		}
	}

	bool prog_is_related_to_qubits() { return m_is_related_to_qubits; }

private:
	const std::vector<int>& m_qubits;
	bool m_is_related_to_qubits;
};

bool QPanda::judge_prog_operate_target_qubts(QProg prog, const QCircuitParam &cir_param, const std::vector<int>& qubits_vec)
{
	for (const auto& cir_control_qubit : cir_param.m_control_qubits)
	{
		for (const auto& item : qubits_vec)
		{
			if (cir_control_qubit->getPhysicalQubitPtr()->getQubitAddr() == item)
			{
				return true;
			}
		}
	}

	JudgeProgOperateQubts judge_operate(prog, qubits_vec);
	judge_operate.traverse_qprog();
	return judge_operate.prog_is_related_to_qubits();
}

void JudgeTwoNodeIterIsSwappable::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
	std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	m_judge_statue->on_enter_circuit(cur_node, cir_param);
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	m_judge_statue->on_leave_circuit(cur_node, cir_param);
}

void JudgeTwoNodeIterIsSwappable::execute(std::shared_ptr<AbstractQuantumProgram> cur_node, 
	std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	m_judge_statue->on_enter_prog(cur_node, parent_node, cir_param);
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	m_judge_statue->on_leave_prog(cur_node, parent_node, cir_param);
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
		m_judge_statue->enter_flow_ctrl_node();
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		m_judge_statue->leave_flow_ctrl_node();
	}
	else if (QIF_START_NODE == iNodeType)
	{
		m_judge_statue->enter_flow_ctrl_node();
		auto true_branch_node = cur_node->getTrueBranch();
		Traversal::traversalByType(true_branch_node, pNode, *this, cir_param, cur_node_iter);
		m_judge_statue->leave_flow_ctrl_node();

		auto false_branch_node = cur_node->getFalseBranch();

		if (nullptr != false_branch_node)
		{
			m_judge_statue->enter_flow_ctrl_node();
			Traversal::traversalByType(false_branch_node, pNode, *this, cir_param, cur_node_iter);
			m_judge_statue->leave_flow_ctrl_node();
		}
	}
}

bool JudgeTwoNodeIterIsSwappable::get_result()
{
	return (COULD_BE_EXCHANGED == m_result);
}

bool JudgeTwoNodeIterIsSwappable::judge_node_type()
{
	const NodeType type_node1 = (*m_nodeItr1)->getNodeType();
	const NodeType type_node2 = (*m_nodeItr2)->getNodeType();
	if ((type_node1 == GATE_NODE) && (type_node2 == GATE_NODE))
	{
		return true;
	}

	return false;
}

void JudgeTwoNodeIterIsSwappable::traverse_qprog()
{
	//get the correlated qubits
	auto get_gate_qubit_func = [&](NodeIter itr, std::vector<int>& vec) {
		QVec tmp_vec;
		switch ((*itr)->getNodeType())
		{
		case GATE_NODE:
		{
			auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*itr);
			p_gate->getQuBitVector(tmp_vec);
			p_gate->getControlVector(tmp_vec);
			for (auto& qubit_item : tmp_vec)
			{
				vec.push_back(qubit_item->getPhysicalQubitPtr()->getQubitAddr());
			}
		}
			break;

		case MEASURE_GATE:
		case RESET_NODE:
			break;

		default:
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Node type error.");
			break;
		}
	};

	std::vector<int> tmp_vec1;
	std::vector<int> tmp_vec2;
	get_gate_qubit_func(m_nodeItr1, tmp_vec1);
	get_gate_qubit_func(m_nodeItr2, tmp_vec2);
	if ((tmp_vec1.size() == 0) || (tmp_vec2.size() == 0))
	{
		m_result = CAN_NOT_BE_EXCHANGED;
		return;
	}

	sort(tmp_vec1.begin(), tmp_vec1.end());
	sort(tmp_vec2.begin(), tmp_vec2.end());
	std::vector<int> same_qubit_vec;
	set_intersection(tmp_vec1.begin(), tmp_vec1.end(), tmp_vec2.begin(), tmp_vec2.end(), std::back_inserter(same_qubit_vec));
	if (same_qubit_vec.size() == 0)
	{
		m_result = CAN_NOT_BE_EXCHANGED;
		return;
	}

	m_correlated_qubits.insert(m_correlated_qubits.end(), tmp_vec1.begin(), tmp_vec1.end());
	m_correlated_qubits.insert(m_correlated_qubits.end(), tmp_vec2.begin(), tmp_vec2.end());
	sort(m_correlated_qubits.begin(), m_correlated_qubits.end());
	m_correlated_qubits.erase(unique(m_correlated_qubits.begin(), m_correlated_qubits.end()), m_correlated_qubits.end());

	//start traverse
	m_judge_statue = (new(std::nothrow) OnInitStatue(*this, INIT));
	if (nullptr == m_judge_statue)
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Memery error, failed to new traversal-statue obj.");
	}
	else
	{
		TraverseByNodeIter::traverse_qprog();
		m_judge_statue->on_traversal_end();
	}
}

void JudgeTwoNodeIterIsSwappable::_pick_node(const NodeIter iter, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param)
{
	QVec tmp_vec;
	switch ((*iter)->getNodeType())
	{
	case GATE_NODE:
	{
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*iter);
		auto gate = QGate(p_gate);
		auto tmp_gate = deepCopy(gate);
		tmp_gate.setDagger(gate.isDagger() ^ (cir_param.m_is_dagger));
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
		m_pick_prog.pushBackNode(dynamic_pointer_cast<QNode>(tmp_gate.getImplementationPtr()));
	}
	break;

	case MEASURE_GATE:
	case RESET_NODE:
	{
		_change_statue(new CanNotBeExchange(*this, CAN_NOT_BE_EXCHANGED));
		return;
	}
	break;

	default:
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Node type error.");
		break;
	}

	if (iter == m_nodeItr1)
	{
		m_nodeItr1 = m_pick_prog.getLastNodeIter();
		m_node_circuit_info.front().m_dagger = cir_param.m_is_dagger;
		m_node_circuit_info.front().m_in_circuit = (CIRCUIT_NODE == parent_node->getNodeType());
	}
	else if (iter == m_nodeItr2)
	{
		m_nodeItr2 = m_pick_prog.getLastNodeIter();
		m_node_circuit_info.back().m_dagger = cir_param.m_is_dagger;
		m_node_circuit_info.back().m_in_circuit = (CIRCUIT_NODE == parent_node->getNodeType());
	}
}

void JudgeTwoNodeIterIsSwappable::_check_picked_prog_matrix()
{
	//deal with dagger
	auto tmp_prog = deepCopy(m_pick_prog);
	auto first_node_iter = tmp_prog.getFirstNodeIter();
	auto last_node_iter = tmp_prog.getLastNodeIter();
	if ((GATE_NODE != (*first_node_iter)->getNodeType()) || (GATE_NODE != (*last_node_iter)->getNodeType()))
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Node type error.");
	}

	if (m_node_circuit_info.front().m_dagger && (!(m_node_circuit_info.back().m_dagger)))
	{
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*first_node_iter);
		p_gate->setDagger(p_gate->isDagger() ^ m_node_circuit_info.front().m_dagger);

		p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(last_node_iter));
		p_gate->setDagger(p_gate->isDagger() ^ m_node_circuit_info.front().m_dagger);
	}
	else if ((!m_node_circuit_info.front().m_dagger) && m_node_circuit_info.back().m_dagger)
	{
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*first_node_iter);
		p_gate->setDagger(p_gate->isDagger() ^ m_node_circuit_info.back().m_dagger);

		p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(last_node_iter));
		p_gate->setDagger(p_gate->isDagger() ^ m_node_circuit_info.back().m_dagger);
	}

#if PRINT_TRACE
	cout << "test the pick prog:" << endl;
	printAllNodeType(tmp_prog);
	cout << tmp_prog << endl;
#endif

	//get the matrix of m_pick_prog
	QStat mat1 = getCircuitMatrix(tmp_prog);

	//get the matrix of the m_pick_prog after exchanged the first node and the last node
	QProg reversed_prog;
	reversed_prog.pushBackNode(*(last_node_iter));
	for (auto iter = ++tmp_prog.getFirstNodeIter(); iter != last_node_iter; ++iter)
	{
		reversed_prog.pushBackNode(*iter);
	}
	reversed_prog.pushBackNode(*(tmp_prog.getFirstNodeIter()));

#if PRINT_TRACE
	cout << mat1 << endl;
	cout << "test the reversed prog:" << endl;
	printAllNodeType(reversed_prog);
	cout << reversed_prog << endl;
#endif
	QStat mat2 = getCircuitMatrix(reversed_prog);

#if PRINT_TRACE
	cout << mat2 << endl;
#endif

	if (0 == mat_compare(mat1, mat2))
	{
		_change_statue(new CoubleBeExchange(*this, COULD_BE_EXCHANGED));
	}
	else
	{
		_change_statue(new CanNotBeExchange(*this, CAN_NOT_BE_EXCHANGED));
	}
}

void JudgeTwoNodeIterIsSwappable::_change_statue(AbstractJudgeStatueInterface* s)
{
	if (nullptr != m_judge_statue)
	{
		SAFE_DELETE_PTR(m_last_statue);
		m_last_statue = m_judge_statue;
	}
	m_judge_statue = s;

	if ((CAN_NOT_BE_EXCHANGED == m_result) || (COULD_BE_EXCHANGED == m_result))
	{
		return;
	}

	if (FOUND_ALL_NODES == m_judge_statue->get_statue())
	{
		_check_picked_prog_matrix();
	}
	else if (JUDGE_MATRIX == m_judge_statue->get_statue())
	{
		//judge matrix
		_check_picked_prog_matrix();
	}

	m_result = m_judge_statue->get_statue();
}