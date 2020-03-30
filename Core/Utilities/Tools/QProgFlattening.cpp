#include "Core/Utilities/Tools/QProgFlattening.h"
#include <set>

USING_QPANDA

QProgFlattening::QProgFlattening(bool is_full_faltten)
{
	m_full_flatten = is_full_faltten;
}

QProgFlattening::~QProgFlattening()
{
}

void QProgFlattening::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog, QCircuit &circuit)
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
		deep_copy_qgate.setControl(get_two_qvec_union(curnode_qv_ctrl, parent_qv_ctrl));
		deep_copy_qgate.setDagger(is_dagger);

		if (m_full_flatten == true)
		{
			prog.pushBackNode(std::dynamic_pointer_cast<QNode>(deep_copy_qgate.getImplementationPtr()));
		}
		else
		{
			circuit.pushBackNode(std::dynamic_pointer_cast<QNode>(deep_copy_qgate.getImplementationPtr()));
		}
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

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog, QCircuit &circuit)
{
	auto type = parent_node->getNodeType();
	if (type != NodeType::PROG_NODE)
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}
	prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
}

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog, QCircuit &circuit)
{
	auto type = parent_node->getNodeType();
	if (type != NodeType::PROG_NODE)
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}
	prog.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
}


void QProgFlattening::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog, QCircuit &circuit)
{
	auto type = parent_node->getNodeType();
	if (type == NodeType::CIRCUIT_NODE)
	{
		circuit.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
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

void QProgFlattening::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog, QCircuit &circuit)
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
		Traversal::traversalByType(while_true_branch, nullptr, *this, while_true_branch_prog, circuit);

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

		Traversal::traversalByType(if_true_branch, nullptr, *this, if_true_branch_prog, circuit);

		auto if_false_branch = cur_node->getFalseBranch();
		if (nullptr != if_false_branch)
		{
			Traversal::traversalByType(if_false_branch, nullptr, *this, if_false_branch_prog, circuit);
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

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog, QCircuit &circuit)
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

			Traversal::traversal(new_cir_node, is_dagger, *this, prog, circuit);
		}
		else 	if (parent_node->getNodeType() == NodeType::PROG_NODE)
		{
			if (m_full_flatten == true)
			{
				bool is_dagger = cur_node->isDagger() ^ identify_dagger;
				Traversal::traversal(cur_node, is_dagger, *this, prog, circuit);
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

				QCircuit new_out_circuit;
				Traversal::traversal(new_abstract_circuit, identify_dagger, *this, prog, new_out_circuit);

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
		cur_node->getControlVector(qv_ctrl);
		is_dagger = cur_node->isDagger();

		//not flatten  first circuit 
		QCircuit cur_node_cir = QCircuit(cur_node);
		QCircuit deep_copy_cir = deepCopy(cur_node_cir);
		auto new_cir_node = deep_copy_cir.getImplementationPtr();
		new_cir_node->clearControl();
		new_cir_node->setDagger(false);

		Traversal::traversal(new_cir_node, identify_dagger, *this, prog, circuit);

		circuit.setControl(qv_ctrl);
		circuit.setDagger(is_dagger);
	}
}

void QProgFlattening::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog, QCircuit &circuit)
{
	Traversal::traversal(cur_node, *this, prog, circuit);
}

void QProgFlattening::flatten_by_type(std::shared_ptr<QNode> node, QProg &out_prog, QCircuit &out_circuit)
{
	if (node == nullptr)
	{
		QCERR("node error");
		throw std::invalid_argument("node error");
	}
	
	Traversal::traversalByType(node, nullptr, *this, out_prog, out_circuit);
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

void QPanda::flatten(QProg &prog)
{
	QCircuit out_circuit;
	QProg out_prog;
	QProgFlattening flatten_qprog;
	flatten_qprog.flatten_by_type(std::dynamic_pointer_cast<QNode>(prog.getImplementationPtr()), out_prog, out_circuit);
	prog = out_prog;
}

void QPanda::flatten(QCircuit &circuit)
{
	QProg out_prog;
	QCircuit out_circuit;
	QProgFlattening flatten_qprog;
	flatten_qprog.flatten_by_type(std::dynamic_pointer_cast<QNode>(circuit.getImplementationPtr()), out_prog, out_circuit);

	circuit = out_circuit;
}

void QPanda::full_flatten(QProg &prog)
{
	QCircuit out_circuit;
	QProg out_prog;
	QProgFlattening flatten_qprog(true);
	flatten_qprog.flatten_by_type(std::dynamic_pointer_cast<QNode>(prog.getImplementationPtr()), out_prog, out_circuit);
	prog = out_prog;
}




