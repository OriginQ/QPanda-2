#include "QProgClockCycle.h"
#include "Utilities/TranformQGateTypeStringAndEnum.h"
using namespace std;
USING_QPANDA
QProgClockCycle::QProgClockCycle(map<int, size_t> gate_time)
    :m_gate_time(gate_time)
{ }

QProgClockCycle::~QProgClockCycle()
{ }

size_t QProgClockCycle::countQProgClockCycle(AbstractQuantumProgram *prog)
{
    if (nullptr == prog)
    {
        throw invalid_argument("prog is a nullptr");
    }

    size_t clock_cycle = 0;
    for (auto iter = prog->getFirstNodeIter(); iter != prog->getEndNodeIter(); iter++)
    {
        QNode * node = (*iter).get();
        clock_cycle += countQNodeClockCycle(node);
    }

    return clock_cycle;
}

size_t QProgClockCycle::countQCircuitClockCycle(AbstractQuantumCircuit *circuit)
{
    if (nullptr == circuit)
    {
        QCERR("circuit is a nullptr");
        throw invalid_argument("circuit is a nullptr");
    }

    size_t clock_cycle = 0;
    for (auto iter = circuit->getFirstNodeIter(); iter != circuit->getEndNodeIter(); iter++)
    {
        QNode * node = (*iter).get();
        clock_cycle += countQNodeClockCycle(node);
    }

    return clock_cycle;
}

size_t QProgClockCycle::countQWhileClockCycle(AbstractControlFlowNode *qwhile)
{
    if (nullptr == qwhile)
    {
        QCERR("qwhile is a nullptr");
        throw invalid_argument("qwhile is a nullptr");
    }

    QNode *pNode = dynamic_cast<QNode *>(qwhile);
    if (nullptr == pNode)
    {
        QCERR("node type error");
        throw runtime_error("node type error");
    }

    size_t clock_cycle = 0;
    QNode *true_branch_node = qwhile->getTrueBranch();

    if (nullptr != true_branch_node)
    {
        clock_cycle += countQNodeClockCycle(true_branch_node);
    }

    return clock_cycle;
}

size_t QProgClockCycle::countQIfClockCycle(AbstractControlFlowNode *qif)
{
    if (nullptr == qif)
    {
        QCERR("qif is a nullptr");
        throw invalid_argument("qif is a nullptr");
    }

    QNode *pNode = dynamic_cast<QNode *>(qif);
    if (nullptr == pNode)
    {
        QCERR("node type error");
        throw runtime_error("node type error");
    }

    size_t true_branch_clock_cycle = 0;
    size_t false_branch_clock_cycle = 0;
    QNode *true_branch_node = qif->getTrueBranch();

    if (nullptr != true_branch_node)
    {
        true_branch_clock_cycle += countQNodeClockCycle(true_branch_node);
    }

    QNode *false_branch_node = qif->getFalseBranch();
    if (nullptr != false_branch_node)
    {
        false_branch_clock_cycle += countQNodeClockCycle(false_branch_node);
    }

    size_t clock_cycle = (true_branch_clock_cycle > false_branch_clock_cycle)
                         ? true_branch_clock_cycle : false_branch_clock_cycle;
    return clock_cycle;
}

size_t QProgClockCycle::getQGateTime(AbstractQGateNode *gate)
{
    if (nullptr == gate)
    {
        QCERR("gate is a nullptr");
        throw invalid_argument("gate is a nullptr");
    }

    int gate_type = gate->getQGate()->getGateType();
    auto iter = m_gate_time.find(gate_type);
    size_t gate_time_value = 0;

    if (m_gate_time.end() == iter)
    {
        gate_time_value = getDefalutQGateTime(gate_type);
        m_gate_time.insert({gate_type, gate_time_value});
        std::cout << "warning: "
                  << QGateTypeEnumToString::getInstance()[gate_type]
                  + " gate is not Configured, will be set a default value "
                  + std::to_string(gate_time_value) + "\n";
    }
    else
    {
        gate_time_value = iter->second;
    }

    return gate_time_value;
}

size_t QProgClockCycle::countQNodeClockCycle(QNode *node)
{
    if (nullptr == node)
    {
        QCERR("node is a nullptr");
        throw invalid_argument("node is a nullptr");
    }

    size_t clock_cycle = 0;
    int type = node->getNodeType();
    switch (type)
    {
    case NodeType::GATE_NODE :
        clock_cycle += getQGateTime(dynamic_cast<AbstractQGateNode *>(node));
        break;
    case NodeType::CIRCUIT_NODE:
        clock_cycle += countQCircuitClockCycle(dynamic_cast<AbstractQuantumCircuit *>(node));
        break;
    case NodeType::PROG_NODE:
        clock_cycle += countQProgClockCycle(dynamic_cast<AbstractQuantumProgram *>(node));
        break;
    case NodeType::QIF_START_NODE:
        clock_cycle += countQIfClockCycle(dynamic_cast<AbstractControlFlowNode *>(node));
        break;
    case NodeType::WHILE_START_NODE:
        clock_cycle += countQWhileClockCycle(dynamic_cast<AbstractControlFlowNode *>(node));
        break;
    case NodeType::MEASURE_GATE:
        break;
    default:
        QCERR("Bad nodeType");
        throw runtime_error("Bad nodeType");
    }

    return clock_cycle;
}

size_t QProgClockCycle::getDefalutQGateTime(int gate_type)
{
    const size_t kSingleGateDefaultTime = 2;
    const size_t kDoubleGateDefaultTime = 5;

    switch (gate_type)
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
    case RX_GATE:
    case RY_GATE:
    case RZ_GATE:
    case U1_GATE:
    case U2_GATE:
    case U3_GATE:
    case U4_GATE:
        return kSingleGateDefaultTime;
        break;
    case CU_GATE:
    case CNOT_GATE:
    case CZ_GATE:
    case CPHASE_GATE:
    case ISWAP_THETA_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
    case TWO_QUBIT_GATE:
        return kDoubleGateDefaultTime;
    default:
        QCERR("Bad nodeType");
        throw runtime_error("Bad nodeType");
    }

    return 0;
}
