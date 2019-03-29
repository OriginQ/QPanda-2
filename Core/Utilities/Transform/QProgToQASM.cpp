#include "Core/Utilities/MetadataValidity.h"
#include "Core/Utilities/Transform/QProgToQASM.h"
#include "Core/Utilities/Transform/TransformDecomposition.h"
#include "QPanda.h"
using namespace std;
USING_QPANDA

QProgToQASM::QProgToQASM()
{
    m_gatetype.insert(pair<int, string>(PAULI_X_GATE, "X"));
    m_gatetype.insert(pair<int, string>(PAULI_Y_GATE, "Y"));
    m_gatetype.insert(pair<int, string>(PAULI_Z_GATE, "Z"));

    m_gatetype.insert(pair<int, string>(X_HALF_PI, "X1"));
    m_gatetype.insert(pair<int, string>(Y_HALF_PI, "Y1"));
    m_gatetype.insert(pair<int, string>(Z_HALF_PI, "Z1"));

    m_gatetype.insert(pair<int, string>(HADAMARD_GATE, "H"));
    m_gatetype.insert(pair<int, string>(T_GATE, "T"));
    m_gatetype.insert(pair<int, string>(S_GATE, "S"));

    m_gatetype.insert(pair<int, string>(RX_GATE, "RX"));
    m_gatetype.insert(pair<int, string>(RY_GATE, "RY"));
    m_gatetype.insert(pair<int, string>(RZ_GATE, "RZ"));
    m_gatetype.insert(pair<int, string>(U1_GATE, "U1"));

    m_gatetype.insert(pair<int, string>(CU_GATE, "CU"));
    m_gatetype.insert(pair<int, string>(CNOT_GATE, "CNOT"));
    m_gatetype.insert(pair<int, string>(CZ_GATE, "CZ"));
    m_gatetype.insert(pair<int, string>(CPHASE_GATE, "CR"));
    m_gatetype.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_gatetype.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));

    m_qasm.clear();
}

QProgToQASM::~QProgToQASM()
{}

void QProgToQASM::progToQASM(AbstractQuantumProgram *pQProg)
{
    m_qasm.emplace_back("OPENQASM 2.0;");
    m_qasm.emplace_back("qreg q[" + to_string(getAllocateQubitNum()) + "];");
    m_qasm.emplace_back("creg c[" + to_string(getAllocateCMem()) + "];");

    const int KMETADATA_GATE_TYPE_COUNT = 2;
    vector<vector<string>> ValidQGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
    vector<vector<string>> QGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
    vector<vector<int>> vAdjacentMatrix;

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[PAULI_X_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[PAULI_Y_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[PAULI_Z_GATE]);

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[HADAMARD_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[T_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[S_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[RX_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[RY_GATE]);

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[RZ_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[U1_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CU_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CNOT_GATE]);

    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CZ_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CPHASE_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[ISWAP_GATE]);

    SingleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
    DoubleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */
    TransformDecomposition traversal_vector(ValidQGateMatrix, QGateMatrix, vAdjacentMatrix);
    traversal_vector.TraversalOptimizationMerge(dynamic_cast<QNode *>(pQProg));

    QProgToQASM::qProgToQASM(pQProg);
}


void QProgToQASM::qProgToQASM(AbstractQGateNode * pQGate)
{
    if (nullptr == pQGate || nullptr == pQGate->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    QVec qubits_vector;
    pQGate->getQuBitVector(qubits_vector);
    auto iter = m_gatetype.find(pQGate->getQGate()->getGateType());

    string tarQubit = to_string(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
    string all_qubits;
    for (auto _val : qubits_vector)
    {
        all_qubits = all_qubits + "q[" + to_string(_val->getPhysicalQubitPtr()->getQubitAddr()) + "]" + ",";
    }
    all_qubits = all_qubits.substr(0, all_qubits.length() - 1);

    string sTemp = iter->second;
    switch (iter->first)
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
            {
                sTemp.append(pQGate->isDagger() ? "dg q[" + tarQubit + "];" : " q[" + tarQubit + "];");
            }
            break;

        case U1_GATE:
        case RX_GATE:
        case RY_GATE:
        case RZ_GATE: 
            {
                string  gate_angle = to_string(dynamic_cast<angleParameter *>(pQGate->getQGate())->getParameter());
                sTemp.append(pQGate->isDagger() ? "dg(" + gate_angle + ")" : "(" + gate_angle + ")");
                sTemp.append(" q[" + tarQubit + "];");
            }
            break;

        case CNOT_GATE:
        case CZ_GATE:
        case ISWAP_GATE:
        case SQISWAP_GATE:
            {
                sTemp.append(pQGate->isDagger() ? "dg " + all_qubits + ";" : " " + all_qubits + ";");
            }
            break;

        case CPHASE_GATE: 
            {
                string  gate_parameter = to_string(dynamic_cast<angleParameter *>(pQGate->getQGate())->getParameter());
                sTemp.append(pQGate->isDagger() ? "dg(" : "(");
                sTemp.append(gate_parameter + ") " + all_qubits + ";");
            }
            break;

        case CU_GATE: 
            {
                QuantumGate * gate_parameter = dynamic_cast<QuantumGate *>(pQGate->getQGate());
                string gate_four_theta = to_string(gate_parameter->getAlpha()) + ',' + 
                                         to_string(gate_parameter->getBeta())  + ',' + 
                                         to_string(gate_parameter->getDelta()) + ',' + 
                                         to_string(gate_parameter->getGamma());

                sTemp.append(pQGate->isDagger() ? "dg(" : "(");
                sTemp.append(gate_four_theta + ") " + all_qubits + ";");
            }
            break;

        default:sTemp = "UnSupportedQuantumGate;";
            break;
    }
    m_qasm.emplace_back(sTemp);
}

void QProgToQASM::qProgToQASM(AbstractQuantumMeasure *pMeasure)
{
    if (nullptr == pMeasure)
    {
        QCERR("pMeasure is null");
        throw invalid_argument("pMeasure is null");
    }
    if (nullptr == pMeasure->getQuBit()->getPhysicalQubitPtr())
    {
        QCERR("PhysicalQubitPtr is null");
        throw invalid_argument("PhysicalQubitPtr is null");
    }

    std::string tar_qubit = to_string(pMeasure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
    std::string creg_name = pMeasure->getCBit()->getName().substr(1);
    m_qasm.emplace_back("measure q[" + tar_qubit + "]" +" -> "+ "c[" + creg_name + "];");
}


void QProgToQASM::qProgToQASM(AbstractQuantumProgram *pQProg)
{
    if (nullptr == pQProg)
    {
        QCERR("pQProg is null");
        throw invalid_argument("pQProg is null");
    }
    for (auto aiter = pQProg->getFirstNodeIter(); aiter != pQProg->getEndNodeIter(); aiter++)
    {
        QNode * pNode = (*aiter).get();
        qProgToQASM(pNode);
    }
}

void QProgToQASM::qProgToQASM(QNode * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }

    switch (pNode->getNodeType())
    {
    case NodeType::GATE_NODE:
        qProgToQASM(dynamic_cast<AbstractQGateNode *>(pNode));
        break;

    case NodeType::CIRCUIT_NODE:
        qProgToQASM(dynamic_cast<AbstractQuantumCircuit *>(pNode));
        break;

    case NodeType::PROG_NODE:
        qProgToQASM(dynamic_cast<AbstractQuantumProgram *>(pNode));
        break;

    case NodeType::MEASURE_GATE:
        qProgToQASM(dynamic_cast<AbstractQuantumMeasure *>(pNode));
        break;

    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
    case NodeType::NODE_UNDEFINED:
    default:m_qasm.emplace_back("UnSupported ProgNode");
        break;
    }
}

void QProgToQASM::handleDaggerNode(QNode * pNode,int nodetype)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }
    if (GATE_NODE == nodetype)
    {
        AbstractQGateNode * pGATE = dynamic_cast<AbstractQGateNode *>(pNode);
        pGATE->setDagger(!pGATE->isDagger());
        qProgToQASM(pGATE);
    }
    else if (CIRCUIT_NODE == nodetype)
    {
        AbstractQuantumCircuit * qCircuit = dynamic_cast<AbstractQuantumCircuit *>(pNode);
        qCircuit->setDagger(!qCircuit->isDagger());
        qProgToQASM(qCircuit);
    }
    else
    {
        QCERR("node type error");
        throw invalid_argument("node type error");
    }
}

void QProgToQASM::handleDaggerCir(QNode * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }
    switch (pNode->getNodeType())
    {
    case NodeType::GATE_NODE:
        QProgToQASM::handleDaggerNode(pNode, GATE_NODE);
        break;

    case NodeType::CIRCUIT_NODE: 
        QProgToQASM::handleDaggerNode(pNode, CIRCUIT_NODE);
        break;

    case NodeType::PROG_NODE:
        QProgToQASM::qProgToQASM(dynamic_cast<AbstractQuantumProgram *>(pNode));
        break;

    case NodeType::MEASURE_GATE:
        QProgToQASM::qProgToQASM(dynamic_cast<AbstractQuantumMeasure *>(pNode));
        break;

    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
    default:m_qasm.emplace_back("UnSupported Prog Node");
    }
}

static void traversalInOrderPCtr(const CExpr* pCtrFlow, string &ctr_statement)
{
    if (nullptr != pCtrFlow)
    {
        traversalInOrderPCtr(pCtrFlow->getLeftExpr(), ctr_statement);
        ctr_statement = ctr_statement + pCtrFlow->getName();
        traversalInOrderPCtr(pCtrFlow->getRightExpr(), ctr_statement);
    }
}

void QProgToQASM::handleIfWhileQASM(AbstractControlFlowNode * pCtrFlow,string ctrflowtype)
{
    if (nullptr == pCtrFlow)
    {
        QCERR("pCtrFlow is null");
        throw invalid_argument("pCtrFlow is null");
    }
    string exper ="";
    auto expr = pCtrFlow->getCExpr()->getExprPtr().get();

    traversalInOrderPCtr(expr, exper);
    m_qasm.emplace_back(ctrflowtype + " " + exper );

    QNode *truth_branch_node = pCtrFlow->getTrueBranch();
    if (nullptr != truth_branch_node)
    {
        qProgToQASM(truth_branch_node);
    }

}

void QProgToQASM::qProgToQASM(AbstractControlFlowNode * pCtrFlow)
{
    if (nullptr == pCtrFlow)
    {
        QCERR("pCtrFlow is null");
        throw invalid_argument("pCtrFlow is null");
    }

    QNode *pNode = dynamic_cast<QNode *>(pCtrFlow);
    switch (pNode->getNodeType())
    {
        case NodeType::WHILE_START_NODE:
            {
                QProgToQASM::handleIfWhileQASM(pCtrFlow, "while");
            }
            break;

        case NodeType::QIF_START_NODE:
            {
                QProgToQASM::handleIfWhileQASM(pCtrFlow, "if");
                m_qasm.emplace_back("else");
                QNode *false_branch_node = pCtrFlow->getFalseBranch();
                if (nullptr != false_branch_node)
                {
                    qProgToQASM(false_branch_node);
                }
            }
            break;
  }
}

void QProgToQASM::qProgToQASM(AbstractQuantumCircuit * pCircuit)
{
    if (nullptr == pCircuit)
    {
        QCERR("pCircuit is null");
        throw invalid_argument("pCircuit is null");
    }
    if (pCircuit->isDagger())
    {
        for (auto aiter = pCircuit->getLastNodeIter(); aiter != pCircuit->getHeadNodeIter(); aiter--)
        {
            QNode * pNode = (*aiter).get();
            handleDaggerCir(pNode);
        }
    }
    else
    {
        for (auto aiter = pCircuit->getFirstNodeIter(); aiter != pCircuit->getEndNodeIter(); aiter++)
        {
            QNode * pNode = (*aiter).get();
            qProgToQASM(pNode);
        }
    }
}

string QProgToQASM::insturctionsQASM()
{
    string instructions;

    for (auto &val : m_qasm)
    {
        transform(val.begin(), val.end(), val.begin(), ::tolower);
        instructions.append(val).append("\n");
    }
    instructions.erase(instructions.size() - 1);
    return instructions;
}

ostream & QPanda::operator<<(ostream & out, const QProgToQASM &qasm_qprog)
{
    for (auto instruct_out : qasm_qprog.m_qasm)
    {
        out << instruct_out << "\n";
    }
    return out;
}
