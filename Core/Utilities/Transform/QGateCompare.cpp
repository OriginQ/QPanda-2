#include "QGateCompare.h"
USING_QPANDA
using namespace std;
static map<int, string>  s_gateTypeInt_map_gateTypeStr = {
    { PAULI_X_GATE,       "X" },
    { PAULI_Y_GATE,       "Y" },
    { PAULI_Z_GATE,       "Z" },
    { X_HALF_PI,          "X1"},
    { Y_HALF_PI,          "Y1"},
    { Z_HALF_PI,          "Z1"},
    { HADAMARD_GATE,      "H" },
    { T_GATE,             "T" },
    { S_GATE,             "S" },
    { RX_GATE,            "RX" },
    { RY_GATE,            "RY" },
    { RZ_GATE,            "RZ" },
    { U1_GATE,            "U1" },
    { U2_GATE,            "U2" },
    { U3_GATE,            "U3" },
    { U4_GATE,            "U4" },
    { CU_GATE,            "CU" },
    { CNOT_GATE,          "CNOT" },
    { CZ_GATE,            "CZ" },
    { CPHASE_GATE,        "CPHASE" },
    { ISWAP_THETA_GATE,   "ISWAP_THETA" },
    { ISWAP_GATE,         "ISWAP" },
    { SQISWAP_GATE,       "SQISWAP" },
    { TWO_QUBIT_GATE,     "TWO_QUBIT" }
};

QGateCompare::QGateCompare()
{
    
}

QGateCompare::~QGateCompare()
{
}

size_t QGateCompare::countQGateNotSupport(AbstractQGateNode * p_gata, const vector<vector<string>>& instructions)
{
    if (nullptr == p_gata)
    {
        QCERR("p_gate node is null");
        throw invalid_argument("p_gate node is null");
    }

    size_t count = 0;
    int qop_num = p_gata->getQGate()->getOperationNum();
    int gate_type = p_gata->getQGate()->getGateType();


    auto iter = s_gateTypeInt_map_gateTypeStr.find(gate_type);
    if (iter == s_gateTypeInt_map_gateTypeStr.end())
    {
        QCERR("gate is error");
        throw invalid_argument("gate is error");
    }

    string item = iter->second;
    if (!isItemExist(item, instructions))
    {
        count++;
    }

    return count;
}

size_t QGateCompare::countQGateNotSupport(AbstractQuantumProgram *p_prog, const vector<vector<string>>& instructions)
{
    if (nullptr == p_prog)
    {
        QCERR("p_prog is null");
        throw invalid_argument("p_prog is null");
    }

    size_t count = 0;
    for (auto iter = p_prog->getFirstNodeIter(); iter != p_prog->getEndNodeIter(); iter++)
    {
        QNode * p_node = (*iter).get();
        count += countQGateNotSupport(p_node, instructions);
    }

    return count;
}


size_t QGateCompare::countQGateNotSupport(AbstractControlFlowNode * p_controlflow, const vector<vector<string>>& instructions)
{
    if (nullptr == p_controlflow)
    {
        QCERR("p_controlflow is null");
        throw invalid_argument("p_controlflow is null");
    }

    QNode *pNode = dynamic_cast<QNode *>(p_controlflow);
    if (nullptr == pNode)
    {
        QCERR("p_node is null");
        throw invalid_argument("p_node is null");
    }

    size_t count = 0;
    QNode *p_true_branch_node = p_controlflow->getTrueBranch();

    if (nullptr != p_true_branch_node)
    {
        count += countQGateNotSupport(p_true_branch_node, instructions);
    }


    if (NodeType::QIF_START_NODE == pNode->getNodeType())
    {
        QNode *p_false_branch_node = p_controlflow->getFalseBranch();
        if (nullptr != p_false_branch_node)
        {
            count += countQGateNotSupport(p_false_branch_node, instructions);
        }
    }

    return count;
}


size_t QGateCompare::countQGateNotSupport(AbstractQuantumCircuit * p_circuit, const vector<vector<string>>& instructions)
{
    if (nullptr == p_circuit)
    {
        QCERR("p_circuit is null");
        throw invalid_argument("p_circuit is null");
    }

    size_t count = 0;
    for (auto iter = p_circuit->getFirstNodeIter(); iter != p_circuit->getEndNodeIter(); iter++)
    {
        QNode * p_node = (*iter).get();
        count += countQGateNotSupport(p_node, instructions);
    }

    return count;
}

size_t QGateCompare::countQGateNotSupport(QNode * p_node, const vector<vector<string>>& instructions)
{
    if (nullptr == p_node)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }

    size_t count = 0;
    int type = p_node->getNodeType();
    switch (type)
    {
    case NodeType::GATE_NODE :
        count += countQGateNotSupport(dynamic_cast<AbstractQGateNode *>(p_node), instructions);
        break;
    case NodeType::CIRCUIT_NODE:
        count += countQGateNotSupport(dynamic_cast<AbstractQuantumCircuit *>(p_node), instructions);
        break;
    case NodeType::PROG_NODE:
        count += countQGateNotSupport(dynamic_cast<AbstractQuantumProgram *>(p_node), instructions);
        break;
    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        count += countQGateNotSupport(dynamic_cast<AbstractControlFlowNode *>(p_node), instructions);
        break;
    case NodeType::MEASURE_GATE:
        break;
    case NodeType::NODE_UNDEFINED:
        break;
    default:
        QCERR("node type error");
        throw invalid_argument("node type error");
        break;
    }

    return count;
}


bool QGateCompare::isItemExist(const string & item, const vector<vector<string>>& instructions)
{
    for (auto &vec : instructions)
    {
        for (auto val : vec)
        {
            if (item == val)
            {
                return true;
            }
        }
    }

    return false;
}

