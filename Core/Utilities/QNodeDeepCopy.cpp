#include "QNodeDeepCopy.h"
#include "Core/QuantumCircuit/QuantumGate.h"
using namespace std;
USING_QPANDA

std::shared_ptr<QNode> QNodeDeepCopy::executeQGate(AbstractQGateNode *cur_node)
{
    QVec qubit_vector;
    cur_node->getQuBitVector(qubit_vector);
    QVec control_qubit_vector;
    cur_node->getControlVector(control_qubit_vector);
    QStat matrix;
    cur_node->getQGate()->getMatrix(matrix);

    auto gate_fac = QGateNodeFactory::getInstance();
    if (qubit_vector.size() == 1)
    {
        QuantumGate * quantum_gate = new QGATE_SPACE::U4(matrix);
        QGate * temp_gate =new QGate(qubit_vector[0], quantum_gate);
        temp_gate->setControl(control_qubit_vector);
        temp_gate->setDagger(cur_node->isDagger());
        shared_ptr<QNode> temp(temp_gate);
        return temp;
    }
    else
    {
        QuantumGate * quantum_gate = new QGATE_SPACE::QDoubleGate(matrix);
        QGate * temp_gate = new QGate(qubit_vector[0],qubit_vector[1], quantum_gate);
        temp_gate->setControl(control_qubit_vector);
        temp_gate->setDagger(cur_node->isDagger());
        shared_ptr<QNode> temp(temp_gate);
        return temp;
    }
}

std::shared_ptr<QNode> QNodeDeepCopy::executeQMeasure(AbstractQuantumMeasure *cur_node)
{
    auto measure_node = new QMeasure(cur_node->getQuBit(), cur_node->getCBit());
    shared_ptr<QNode> temp(measure_node);
    return temp;
}

std::shared_ptr<QNode> QNodeDeepCopy::executeQCircuit(AbstractQuantumCircuit *cur_node)
{
    QVec control_vec;
    cur_node->getControlVector(control_vec);

    auto temp_cir = new QCircuit();
    for (auto iter = cur_node->getFirstNodeIter(); iter != cur_node->getEndNodeIter(); ++iter)
    {
        Traversal::traversalByType((*iter).get(), dynamic_cast<QNode *>(temp_cir), this);
    }

    temp_cir->setDagger(cur_node->isDagger());
    temp_cir->setControl(control_vec);

    shared_ptr<QNode> temp(temp_cir);
    return temp;
}

std::shared_ptr<QNode> QNodeDeepCopy::executeQProg(AbstractQuantumProgram *cur_node)
{
    auto temp_prog = new QProg();
    for (auto iter = cur_node->getFirstNodeIter(); iter != cur_node->getEndNodeIter(); ++iter)
    {
        Traversal::traversalByType((*iter).get(), dynamic_cast<QNode *>(temp_prog), this);
    }

    shared_ptr<QNode> temp(temp_prog);
    return temp;
}

std::shared_ptr<QNode> QNodeDeepCopy::executeQNode(QNode *node)
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
                auto qcircuit_node = dynamic_cast<AbstractQuantumCircuit *>(node);
                if (nullptr == qcircuit_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }
                return executeQCircuit(qcircuit_node);
            }
            break;

        case NodeType::GATE_NODE:
            {
                auto gate_node = dynamic_cast<AbstractQGateNode *>(node);
                if (nullptr == gate_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                return executeQGate(gate_node);
            }
            break;

        case NodeType::MEASURE_GATE:
            {
                auto measure_node = dynamic_cast<AbstractQuantumMeasure *>(node);

                if (nullptr == measure_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                return executeQMeasure(measure_node);
            }
            break;

        case NodeType::QIF_START_NODE:
        case NodeType::WHILE_START_NODE:
            {
                auto control_flow_node = dynamic_cast<AbstractControlFlowNode *>(node);

                if (nullptr == control_flow_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                return executeControlFlow(control_flow_node);
            }
            break;

        case NodeType::PROG_NODE:
            {
                auto qprog_node = dynamic_cast<AbstractQuantumProgram *>(node);

                if (nullptr == qprog_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                return executeQProg(qprog_node);
            }
            break;

        case NodeType::CLASS_COND_NODE:
            {
                auto cprog_node = dynamic_cast<AbstractClassicalProg *>(node);

                if (nullptr == cprog_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                return executeClassicalProg(cprog_node);
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

std::shared_ptr<QNode> QNodeDeepCopy::executeControlFlow(AbstractControlFlowNode *cur_node)
{
    if (nullptr == cur_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pNode = dynamic_cast<QNode *>(cur_node);
    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto Cexpr = cur_node->getCExpr();
    if (nullptr == Cexpr)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    auto expr = Cexpr->getExprPtr()->deepcopy();
    ClassicalCondition cbit = ClassicalCondition(expr);

    switch (pNode->getNodeType())
    {
    case NodeType::WHILE_START_NODE:
    {
        auto true_branch_node = executeQNode(cur_node->getTrueBranch());
        auto while_node = new QWhileProg(cbit, true_branch_node.get());
        shared_ptr<QNode> temp(while_node);
        return temp;
    }
    break;

    case NodeType::QIF_START_NODE:
    {
        auto true_branch_node = executeQNode(cur_node->getTrueBranch());

        auto false_branch_node = cur_node->getFalseBranch();
        if (nullptr != false_branch_node)
        {
            auto false_branch_node = executeQNode(cur_node->getFalseBranch());
            auto if_node = new QIfProg(cbit, true_branch_node.get(), false_branch_node.get());
            shared_ptr<QNode> temp(if_node);
            return temp;
        }
        else
        {
            auto if_node = new QIfProg(cbit, true_branch_node.get());
            shared_ptr<QNode> temp(if_node);
            return temp;
        }
    }
    break;

    default:
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    break;
    }
}

void QNodeDeepCopy::execute(AbstractControlFlowNode *cur_node, QNode *parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pControlFlow = executeControlFlow(cur_node);
    insert(pControlFlow, parent_node);
}

void QNodeDeepCopy::execute(AbstractQuantumProgram *cur_node, QNode *parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pQProg = executeQProg(cur_node);
    insert(pQProg, parent_node);
}

void QNodeDeepCopy::execute(AbstractQGateNode *cur_node, QNode *parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pGate = executeQGate(cur_node);
    insert(pGate, parent_node);
}

void QNodeDeepCopy::execute(AbstractQuantumMeasure *cur_node, QNode *parent_node)                                                                
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pMeasure = executeQMeasure(cur_node);
    insert(pMeasure, parent_node);
}

void QNodeDeepCopy::execute(AbstractQuantumCircuit *cur_node, QNode *parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pCircuit = executeQCircuit(cur_node);
    insert(pCircuit, parent_node);
}


void QNodeDeepCopy::insert(std::shared_ptr<QNode> cur_node, QNode * parent_node)
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
                auto qcircuit_node = dynamic_cast<AbstractQuantumCircuit *>(parent_node);
                if (nullptr == qcircuit_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                int cur_node_type = cur_node->getNodeType();
                if (NodeType::CIRCUIT_NODE == cur_node_type ||
                    NodeType::GATE_NODE == cur_node_type)
                {
                    qcircuit_node->pushBackNode(cur_node->getImplementationPtr());
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
                auto qprog_node = dynamic_cast<AbstractQuantumProgram *>(parent_node);
                if (nullptr == qprog_node)
                {
                    QCERR("Unknown internal error");
                    throw runtime_error("Unknown internal error");
                }

                qprog_node->pushBackNode(cur_node->getImplementationPtr());
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

std::shared_ptr<QNode> QNodeDeepCopy::executeClassicalProg(AbstractClassicalProg * cur_node)
{
    auto expr = cur_node->getExpr();
    if (nullptr == expr)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    auto expr_node = expr.get()->deepcopy();
    ClassicalCondition cbit = ClassicalCondition(expr_node);

    auto classical_node = new ClassicalProg(cbit);
    shared_ptr<QNode> temp(classical_node);
    return temp;
}



void QNodeDeepCopy::execute(AbstractClassicalProg *cur_node, QNode *parent_node)
{
    if (nullptr == cur_node || nullptr == parent_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    auto pControlProg = executeClassicalProg(cur_node);
    insert(pControlProg, parent_node);
}