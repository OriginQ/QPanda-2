#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/QuantumCircuit/QuantumGate.h"
using namespace std;
USING_QPANDA

QGate QNodeDeepCopy::copy_node(shared_ptr<AbstractQGateNode>cur_node)
{
    QVec qubit_vector;
    cur_node->getQuBitVector(qubit_vector);
    QVec control_qubit_vector;
    cur_node->getControlVector(control_qubit_vector);

	QGate temp_gate = copy_qgate(cur_node->getQGate(), qubit_vector);
	temp_gate.setControl(control_qubit_vector);
	temp_gate.setDagger(cur_node->isDagger());
	return temp_gate;
}

QMeasure QNodeDeepCopy::copy_node(shared_ptr<AbstractQuantumMeasure> cur_node)
{
    auto measure_node = QMeasure(cur_node->getQuBit(), cur_node->getCBit());
    return measure_node;
}

QReset QNodeDeepCopy::copy_node(std::shared_ptr<AbstractQuantumReset> cur_node)
{
	auto reset_node = QReset(cur_node->getQuBit());
	return reset_node;
}

QCircuit QNodeDeepCopy::copy_node(shared_ptr<AbstractQuantumCircuit> cur_node)
{
    QVec control_vec;
    cur_node->getControlVector(control_vec);

    auto temp_cir = QCircuit();
    for (auto iter = cur_node->getFirstNodeIter(); iter != cur_node->getEndNodeIter(); ++iter)
    {
        Traversal::traversalByType(*iter, dynamic_pointer_cast<QNode>(temp_cir.getImplementationPtr()), *this);
    }

    temp_cir.setDagger(cur_node->isDagger());
    temp_cir.setControl(control_vec);

    return temp_cir;
}

QProg QNodeDeepCopy::copy_node(shared_ptr<AbstractQuantumProgram> cur_node)
{
    auto temp_prog = QProg();
    for (auto iter = cur_node->getFirstNodeIter(); iter != cur_node->getEndNodeIter(); ++iter)
    {
        Traversal::traversalByType((*iter), dynamic_pointer_cast<QNode>(temp_prog.getImplementationPtr()), *this);
    }
    return temp_prog;
}


std::shared_ptr<QNode> QNodeDeepCopy::executeQNode(std::shared_ptr<QNode> node)
{
	if (nullptr == node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	int iNodeType = node->getNodeType();
	switch (iNodeType)
	{
	case NodeType::CIRCUIT_NODE:
	{
		auto qcircuit_node = std::dynamic_pointer_cast<AbstractQuantumCircuit>(node);
		if (nullptr == qcircuit_node)
		{
			QCERR("Unknown internal error");
			throw runtime_error("Unknown internal error");
		}
		auto temp = copy_node(qcircuit_node);
		auto result = std::dynamic_pointer_cast<QNode>(temp.getImplementationPtr());

		return result;
	}
	break;

	case NodeType::GATE_NODE:
	{
		auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(node);
		if (nullptr == gate_node)
		{
			QCERR("Unknown internal error");
			throw runtime_error("Unknown internal error");
		}

		auto temp = copy_node(gate_node);
		auto result = std::dynamic_pointer_cast<QNode>(temp.getImplementationPtr());
	}
	break;

	case NodeType::MEASURE_GATE:
	{
		auto measure_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(node);

		if (nullptr == measure_node)
		{
			QCERR("Unknown internal error");
			throw runtime_error("Unknown internal error");
		}

		auto temp = copy_node(measure_node);
		auto result = std::dynamic_pointer_cast<QNode>(temp.getImplementationPtr());
	}
	break;

	case NodeType::QIF_START_NODE:
	case NodeType::WHILE_START_NODE:
	{
		auto control_flow_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(node);

		if (nullptr == control_flow_node)
		{
			QCERR("Unknown internal error");
			throw runtime_error("Unknown internal error");
		}

		auto temp = copy_node(control_flow_node);
		auto result = std::dynamic_pointer_cast<QNode>(temp);
	}
	break;

	case NodeType::PROG_NODE:
	{
		auto qprog_node = std::dynamic_pointer_cast<AbstractQuantumProgram>(node);

		if (nullptr == qprog_node)
		{
			QCERR("Unknown internal error");
			throw runtime_error("Unknown internal error");
		}

		auto temp = copy_node(qprog_node);
		auto result = std::dynamic_pointer_cast<QNode>(temp.getImplementationPtr());
	}
	break;

	case NodeType::CLASS_COND_NODE:
	{
		auto cprog_node = std::dynamic_pointer_cast<AbstractClassicalProg>(node);

		if (nullptr == cprog_node)
		{
			QCERR("Unknown internal error");
			throw runtime_error("Unknown internal error");
		}

		auto temp = copy_node(cprog_node);
		auto result = std::dynamic_pointer_cast<QNode>(temp.getImplementationPtr());
	}
	break;


	case NodeType::NODE_UNDEFINED:
	default:
	{
		QCERR("NodeType error");
		throw undefine_error("NodeType");
	}
	break;
	}
}


std::shared_ptr<AbstractControlFlowNode> QNodeDeepCopy::copy_node(shared_ptr<AbstractControlFlowNode> cur_node)
{
    if (nullptr == cur_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pNode = dynamic_pointer_cast<QNode>(cur_node);
    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto Cexpr = cur_node->getCExpr();

    auto expr = Cexpr.getExprPtr()->deepcopy();
    ClassicalCondition cbit = ClassicalCondition(expr);

    switch (pNode->getNodeType())
    {
    case NodeType::WHILE_START_NODE:
    {
        auto true_branch_node = executeQNode(cur_node->getTrueBranch());
        auto while_node = QWhileProg(cbit, true_branch_node);
        return while_node.getImplementationPtr();
    }
    break;

    case NodeType::QIF_START_NODE:
    {
        auto true_branch_node = executeQNode(cur_node->getTrueBranch());

        auto false_branch_node = cur_node->getFalseBranch();
        if (nullptr != false_branch_node)
        {
            auto false_branch_node = executeQNode(cur_node->getFalseBranch());
            auto if_node = QIfProg(cbit, true_branch_node, false_branch_node);
            return dynamic_pointer_cast<AbstractControlFlowNode> (if_node.getImplementationPtr());;
        }
        else
        {
            auto if_node = QIfProg(cbit, true_branch_node);
			return if_node.getImplementationPtr();
        }
    }
    break;

    default:
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }
    break;
    }
}

void QNodeDeepCopy::execute(std::shared_ptr<AbstractControlFlowNode> cur_node,  std::shared_ptr<QNode>  parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pControlFlow = copy_node(cur_node);
    insert(dynamic_pointer_cast<QNode>(pControlFlow), parent_node);
}

void QNodeDeepCopy::execute(shared_ptr<AbstractQuantumProgram> cur_node, shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pQProg = copy_node(cur_node);
    insert(dynamic_pointer_cast<QNode>(pQProg.getImplementationPtr()), parent_node);
}

void QNodeDeepCopy::execute(shared_ptr<AbstractQGateNode>  cur_node, shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pGate = copy_node(cur_node);
    insert(dynamic_pointer_cast<QNode>(pGate.getImplementationPtr()), parent_node);
}

void QNodeDeepCopy::execute(  shared_ptr<AbstractQuantumMeasure> cur_node, shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pMeasure = copy_node(cur_node);
    insert(dynamic_pointer_cast<QNode>(pMeasure.getImplementationPtr()), parent_node);
}

void QNodeDeepCopy::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
{
	if (nullptr == cur_node || nullptr == parent_node)
	{
		QCERR("node is nullptr");
		throw invalid_argument("node is nullptr");
	}

	auto pReset = copy_node(cur_node);
	insert(dynamic_pointer_cast<QNode>(pReset.getImplementationPtr()), parent_node);
}

void QNodeDeepCopy::execute( shared_ptr<AbstractQuantumCircuit> cur_node, shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pCircuit = copy_node(cur_node);
    insert(dynamic_pointer_cast<QNode>(pCircuit.getImplementationPtr()), parent_node);
}


void QNodeDeepCopy::insert(std::shared_ptr<QNode> cur_node, shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    int parent_node_type = parent_node->getNodeType();
    switch (parent_node_type)
    {
        case NodeType::CIRCUIT_NODE:
            {
                auto qcircuit_node = dynamic_pointer_cast<AbstractQuantumCircuit>(parent_node);
                if (nullptr == qcircuit_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                int cur_node_type = cur_node->getNodeType();
                if (NodeType::CIRCUIT_NODE == cur_node_type ||
                    NodeType::GATE_NODE == cur_node_type)
                {
                    qcircuit_node->pushBackNode(cur_node);
                }
                else
                {
                    QCERR("cur_node_type error");
                    throw qprog_syntax_error("cur_node_type");
                }
            }
            break;

        case NodeType::PROG_NODE:
            {
                auto qprog_node = dynamic_pointer_cast<AbstractQuantumProgram>(parent_node);
                if (nullptr == qprog_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                qprog_node->pushBackNode(cur_node);
            }
            break;

        default:
            {
                QCERR("parent_node_type error");
                throw runtime_error("parent_node_type error");
            }
        break;
    }
}

ClassicalProg QNodeDeepCopy::copy_node(std::shared_ptr<AbstractClassicalProg>  cur_node)
{
    auto expr = cur_node->getExpr();
    if (nullptr == expr)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    auto expr_node = expr.get()->deepcopy();
    ClassicalCondition cbit = ClassicalCondition(expr_node);

    auto classical_node = ClassicalProg(cbit);
    return classical_node;
}



void QNodeDeepCopy::execute(shared_ptr<AbstractClassicalProg> cur_node, shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pControlProg = copy_node(cur_node);
    insert(dynamic_pointer_cast<QNode>(pControlProg.getImplementationPtr()), parent_node);
}