#include "Core/Utilities/Tools/QProgFlattening.h"
#include <set>
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

USING_QPANDA

QProgFlattening::QProgFlattening(bool is_full_faltten)
{
	m_full_flatten = is_full_faltten;
}

QProgFlattening::~QProgFlattening()
{
}

void QProgFlattening::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	auto type = parent_node->getNodeType();
	if (type == NodeType::CIRCUIT_NODE)
	{
		//flatten the quantum gates in the circuit
		QVec parent_qv_ctrl;
		QVec curnode_qv_ctrl;
		
		auto parent_circuit = std::dynamic_pointer_cast<AbstractQuantumCircuit>(parent_node);
		parent_circuit->getControlVector(parent_qv_ctrl);
		cur_node->getControlVector(curnode_qv_ctrl);
		bool is_dagger = cur_node->isDagger() ^ parent_circuit->isDagger();

		QGate cur_node_qgate = QGate(cur_node);
		QGate deep_copy_qgate = deepCopy(cur_node_qgate);

		// parent_qv_ctrl - curnode_qv_ctrl
		auto sort_fun = [](Qubit*a, Qubit* b) {return a->getPhysicalQubitPtr()->getQubitAddr() < b->getPhysicalQubitPtr()->getQubitAddr(); };
		std::sort(parent_qv_ctrl.begin(), parent_qv_ctrl.end(), sort_fun);

		auto _earse_check_fun = [](Qubit*a, Qubit* b) {return a->getPhysicalQubitPtr()->getQubitAddr() == b->getPhysicalQubitPtr()->getQubitAddr(); };
		parent_qv_ctrl.erase(unique(parent_qv_ctrl.begin(),
			parent_qv_ctrl.end(), _earse_check_fun),
			parent_qv_ctrl.end());

		std::sort(curnode_qv_ctrl.begin(), curnode_qv_ctrl.end(), sort_fun);
		curnode_qv_ctrl.erase(unique(curnode_qv_ctrl.begin(),
			curnode_qv_ctrl.end(), _earse_check_fun),
			curnode_qv_ctrl.end());

		QVec result_vec;
		set_difference(parent_qv_ctrl.begin(), parent_qv_ctrl.end(), curnode_qv_ctrl.begin(), curnode_qv_ctrl.end(), std::back_inserter(result_vec), sort_fun);
		
		deep_copy_qgate.setControl(result_vec);
		deep_copy_qgate.setDagger(is_dagger);

		prog.pushBackNode(std::dynamic_pointer_cast<QNode>(deep_copy_qgate.getImplementationPtr()));
	}
	else if (type == NodeType::PROG_NODE)
	{
		prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
	}
	else
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}
}

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	auto type = parent_node->getNodeType();
	if (type != NodeType::PROG_NODE)
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}
	prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
}

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	auto type = parent_node->getNodeType();
	if (type != NodeType::PROG_NODE)
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}
	prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
}


void QProgFlattening::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
}

void QProgFlattening::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	auto type = parent_node->getNodeType();
	if (type != NodeType::PROG_NODE)
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}
	auto qnode = std::dynamic_pointer_cast<QNode>(cur_node);

	switch (qnode->getNodeType())
	{
	case NodeType::WHILE_START_NODE:
	{
		auto while_true_branch = cur_node->getTrueBranch();
		if (nullptr == while_true_branch)
		{
			QCERR("while_branch_node error");
			throw std::invalid_argument("while_branch_node error");
		}
		QProg while_true_branch_prog;
		Traversal::traversalByType(while_true_branch, nullptr, *this, while_true_branch_prog);

		auto qwhile = createWhileProg(cur_node->getCExpr(), while_true_branch_prog);
		prog.pushBackNode(std::dynamic_pointer_cast<QNode>(qwhile.getImplementationPtr()));

	}
	break;

	case NodeType::QIF_START_NODE:
	{
		QProg if_true_branch_prog;
		QProg if_false_branch_prog;
		auto if_true_branch = cur_node->getTrueBranch();
		if (nullptr == if_true_branch)
		{
			QCERR("if_true_branch error");
			throw std::invalid_argument("if_true_branch error");
		}

		Traversal::traversalByType(if_true_branch, nullptr, *this, if_true_branch_prog);

		auto if_false_branch = cur_node->getFalseBranch();
		if (nullptr != if_false_branch)
		{
			Traversal::traversalByType(if_false_branch, nullptr, *this, if_false_branch_prog);
			auto qif = createIfProg(cur_node->getCExpr(), if_true_branch_prog, if_false_branch_prog);
			prog.pushBackNode(std::dynamic_pointer_cast<QNode>(qif.getImplementationPtr()));
		}
		else
		{
			auto qif = createIfProg(cur_node->getCExpr(), if_true_branch_prog);
			prog.pushBackNode(std::dynamic_pointer_cast<QNode>(qif.getImplementationPtr()));
		}
	}
	break;
	default:
		throw std::invalid_argument("control flow node error");
	break;
	}
}

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
    QVec qv_ctrl;
	bool identify_dagger = false;
	bool is_dagger = false;
	if (parent_node != nullptr)
	{
		if (parent_node->getNodeType() == NodeType::CIRCUIT_NODE)
		{
			//flatten the nested circuit
			auto parent_abstract_circuit = std::dynamic_pointer_cast<AbstractQuantumCircuit>(parent_node);

			parent_abstract_circuit->getControlVector(qv_ctrl);
			is_dagger = cur_node->isDagger() ^ parent_abstract_circuit->isDagger();

			QCircuit cur_node_cir = QCircuit(cur_node);
			QCircuit deep_copy_cir = deepCopy(cur_node_cir);
			auto new_cir_node = deep_copy_cir.getImplementationPtr();
			new_cir_node->setControl(qv_ctrl);
			new_cir_node->setDagger(is_dagger);

			Traversal::traversal(new_cir_node, is_dagger, *this, prog);
		}
		else 	if (parent_node->getNodeType() == NodeType::PROG_NODE)
		{
			if (m_full_flatten == true)
			{
				bool is_dagger = cur_node->isDagger() ^ identify_dagger;
				Traversal::traversal(cur_node, is_dagger, *this, prog);
			}
			else
			{
				cur_node->getControlVector(qv_ctrl);
				is_dagger = cur_node->isDagger();

				//not flatten  first circuit 
				QCircuit cur_node_cir = QCircuit(cur_node);
				QCircuit deep_copy_cir = deepCopy(cur_node_cir);
				auto new_abstract_circuit = deep_copy_cir.getImplementationPtr();
				new_abstract_circuit->clearControl();
				new_abstract_circuit->setDagger(false);

				QProg new_out_prog;
				Traversal::traversal(new_abstract_circuit, identify_dagger, *this, new_out_prog);

				QCircuit new_out_circuit = prog_flatten_to_cir(new_out_prog);
				new_out_circuit.setDagger(is_dagger);
				new_out_circuit.setControl(qv_ctrl);

				prog.pushBackNode(std::dynamic_pointer_cast<QNode>(new_out_circuit.getImplementationPtr()));
			}
		}
		else
		{
			QCERR("node error");
			throw std::invalid_argument("node error");
		}
	}
	else
	{
		cur_node->getControlVector(m_global_ctrl_qubits);
		m_global_dagger = cur_node->isDagger();

		//not flatten  first circuit 
		QCircuit cur_node_cir = QCircuit(cur_node);
		QCircuit deep_copy_cir = deepCopy(cur_node_cir);
		auto new_cir_node = deep_copy_cir.getImplementationPtr();
		new_cir_node->clearControl();
		new_cir_node->setDagger(false);

		Traversal::traversal(new_cir_node, identify_dagger, *this, prog);
	}
}

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	Traversal::traversal(cur_node, *this, prog);
}

void QProgFlattening::execute(std::shared_ptr<AbstractQNoiseNode> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
}

void QProgFlattening::execute(std::shared_ptr<AbstractQDebugNode> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)
{
	prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
}

QVec QProgFlattening::get_two_qvec_union(QVec qv_1, QVec qv_2)
{
	QVec out_qv;
	std::set<Qubit *, QubitPointerCmp> qubits_set;
	for (auto iter : qv_1)
	{
		qubits_set.insert(iter);
	}
	for (auto iter : qv_2)
	{
		qubits_set.insert(iter);
	}

	for (auto iter : qubits_set)
	{
		out_qv.push_back(iter);
	}

	return out_qv;
}

void QProgFlattening::flatten_by_type(std::shared_ptr<QNode> node, QProg& flattened_prog)
{
	if (node == nullptr)
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}

	Traversal::traversalByType(node, nullptr, *this, flattened_prog);
}

QCircuit QProgFlattening::prog_flatten_to_cir(QProg& prog)
{
	QCircuit ret_cir;
	flatten(prog);
	for (auto gate_itr = prog.getFirstNodeIter(); gate_itr != prog.getEndNodeIter(); ++gate_itr)
	{
		if (GATE_NODE != (*gate_itr)->getNodeType())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: can't transfer current prog to circuit.");
		}
		ret_cir.pushBackNode(*gate_itr);
	}

	return ret_cir;
}

void QProgFlattening::flatten_circuit(QCircuit &src_cir)
{
	QProg out_prog;
	flatten_by_type(std::dynamic_pointer_cast<QNode>(src_cir.getImplementationPtr()), out_prog);
	QCircuit tmp_cir = prog_flatten_to_cir(out_prog);
	tmp_cir.setControl(m_global_ctrl_qubits);
	tmp_cir.setDagger(m_global_dagger);
	src_cir = tmp_cir;
}

void QProgFlattening::flatten_prog(QProg &src_prog)
{
	QProg out_prog;
	flatten_by_type(std::dynamic_pointer_cast<QNode>(src_prog.getImplementationPtr()), out_prog);
	src_prog = out_prog;
}

void QPanda::flatten(QCircuit& src_cir)
{
	QProgFlattening flattener;
	flattener.flatten_circuit(src_cir);
}

void QPanda::flatten(QProg& src_prog, bool b_full_flatten /*= true*/)
{
	QProgFlattening flattener(b_full_flatten);
	flattener.flatten_prog(src_prog);
}

