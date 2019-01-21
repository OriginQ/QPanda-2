#include <iostream>
#include "QProgToQASM.h"
#include "QPanda.h"
#include "TransformDecomposition.h"
#include "Utilities/MetadataValidity.h"
using QGATE_SPACE::angleParameter;
using namespace std;
USING_QPANDA

QProgToQASM::QProgToQASM()
{
    m_gatetype_majuscule.insert(pair<int, string>(PAULI_X_GATE, "X"));
    m_gatetype_majuscule.insert(pair<int, string>(PAULI_Y_GATE, "Y"));
    m_gatetype_majuscule.insert(pair<int, string>(PAULI_Z_GATE, "Z"));

    m_gatetype_majuscule.insert(pair<int, string>(X_HALF_PI, "X1"));
    m_gatetype_majuscule.insert(pair<int, string>(Y_HALF_PI, "Y1"));
    m_gatetype_majuscule.insert(pair<int, string>(Z_HALF_PI, "Z1"));

    m_gatetype_majuscule.insert(pair<int, string>(HADAMARD_GATE, "H"));
    m_gatetype_majuscule.insert(pair<int, string>(T_GATE, "T"));
    m_gatetype_majuscule.insert(pair<int, string>(S_GATE, "S"));

    m_gatetype_majuscule.insert(pair<int, string>(RX_GATE, "RX"));
    m_gatetype_majuscule.insert(pair<int, string>(RY_GATE, "RY"));
    m_gatetype_majuscule.insert(pair<int, string>(RZ_GATE, "RZ"));

    m_gatetype_majuscule.insert(pair<int, string>(U1_GATE, "U1"));
    m_gatetype_majuscule.insert(pair<int, string>(U2_GATE, "U2"));
    m_gatetype_majuscule.insert(pair<int, string>(U3_GATE, "U3"));
    m_gatetype_majuscule.insert(pair<int, string>(U4_GATE, "U4"));

    m_gatetype_majuscule.insert(pair<int, string>(CU_GATE, "CU"));
    m_gatetype_majuscule.insert(pair<int, string>(CNOT_GATE, "CNOT"));
    m_gatetype_majuscule.insert(pair<int, string>(CZ_GATE, "CZ"));
    m_gatetype_majuscule.insert(pair<int, string>(CPHASE_GATE, "CPHASE"));
    m_gatetype_majuscule.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_gatetype_majuscule.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));




    m_gatetype_qasm.insert(pair<int, string>(PAULI_X_GATE, "x"));
    m_gatetype_qasm.insert(pair<int, string>(PAULI_Y_GATE, "y"));
    m_gatetype_qasm.insert(pair<int, string>(PAULI_Z_GATE, "u"));

    m_gatetype_qasm.insert(pair<int, string>(X_HALF_PI, "x1"));
    m_gatetype_qasm.insert(pair<int, string>(Y_HALF_PI, "y1"));
    m_gatetype_qasm.insert(pair<int, string>(Z_HALF_PI, "z1"));

    m_gatetype_qasm.insert(pair<int, string>(HADAMARD_GATE, "h"));
    m_gatetype_qasm.insert(pair<int, string>(T_GATE, "t"));
    m_gatetype_qasm.insert(pair<int, string>(S_GATE, "s"));

    m_gatetype_qasm.insert(pair<int, string>(RX_GATE, "rx"));
    m_gatetype_qasm.insert(pair<int, string>(RY_GATE, "ry"));
    m_gatetype_qasm.insert(pair<int, string>(RZ_GATE, "rz"));

    m_gatetype_qasm.insert(pair<int, string>(U1_GATE, "u1"));
    m_gatetype_qasm.insert(pair<int, string>(U2_GATE, "u2"));
    m_gatetype_qasm.insert(pair<int, string>(U3_GATE, "u3"));
    m_gatetype_qasm.insert(pair<int, string>(U4_GATE, "u4"));

    m_gatetype_qasm.insert(pair<int, string>(CU_GATE, "cu"));
    m_gatetype_qasm.insert(pair<int, string>(CNOT_GATE, "cnot"));
    m_gatetype_qasm.insert(pair<int, string>(CZ_GATE, "cz"));
    m_gatetype_qasm.insert(pair<int, string>(CPHASE_GATE, "cphase"));
    m_gatetype_qasm.insert(pair<int, string>(ISWAP_GATE, "iswap"));
    m_gatetype_qasm.insert(pair<int, string>(SQISWAP_GATE, "sqiswap"));

    m_qasm.clear();

}

QProgToQASM::~QProgToQASM()
{
}

void QProgToQASM::qProgToQasm(AbstractQuantumProgram *pQpro)
{
    m_qasm.emplace_back("OPENQASM 2.0;");
    m_qasm.emplace_back("qreg q[" + to_string(getAllocateQubitNum()) + "];");
    m_qasm.emplace_back("creg c[" + to_string(getAllocateCMem()) + "];");


    const int KMETADATA_GATE_TYPE_COUNT = 2;
    vector<vector<string>> ValidQGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
    vector<vector<string>> QGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
    vector<vector<int>> vAdjacentMatrix;

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[PAULI_X_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[PAULI_Y_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[PAULI_Z_GATE]);

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[HADAMARD_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[T_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[S_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[RX_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[RY_GATE]);

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[RZ_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype_majuscule[U1_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype_majuscule[CU_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype_majuscule[CNOT_GATE]);

    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype_majuscule[CZ_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype_majuscule[CPHASE_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype_majuscule[ISWAP_GATE]);

    SingleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
    DoubleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */
    TransformDecomposition traversal_vector(ValidQGateMatrix, QGateMatrix, vAdjacentMatrix);
    traversal_vector.TraversalOptimizationMerge(dynamic_cast<QNode *>(pQpro));

    QProgToQASM::qProgToQASM(pQpro);
}


void QProgToQASM::qProgToQASM(AbstractQGateNode * pQGata)
{
    if (nullptr == pQGata)
    {
        QCERR("pQGata is null");
        throw invalid_argument("pQGata is null");
    }
    if (nullptr == pQGata->getQGate())
    {
        QCERR("pQGata->getQGate() is null");
        throw invalid_argument("pQGata->getQGate() is null");
    }

    int gate_type = pQGata->getQGate()->getGateType();

    vector<Qubit*> qubits_vector;

    pQGata->getQuBitVector(qubits_vector);
    auto iter = m_gatetype_qasm.find(gate_type);

    string item = iter->second;
    string first_qubit = to_string(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
    string all_qubits;

    for (auto _val : qubits_vector)
    {
        all_qubits = all_qubits + "q["+to_string(_val->getPhysicalQubitPtr()->getQubitAddr())+"]" + ",";
    }

    all_qubits = all_qubits.substr(0, all_qubits.length() - 1);
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
            if (pQGata->isDagger())
            {
                m_qasm.emplace_back(item + "dg q[" + first_qubit + "];");
            }
            else
            {
                m_qasm.emplace_back(item + " q[" + first_qubit + "];");
            }
        }
        break;

    case U1_GATE:
    case RX_GATE:
    case RY_GATE:
    case RZ_GATE: 
        {
            string  gate_angle = to_string(dynamic_cast<angleParameter *>(pQGata->getQGate())->getParameter());
            if (pQGata->isDagger())
            {
             m_qasm.emplace_back(item + "dg(" + gate_angle + ")" + " q[" + first_qubit + "];");
            }
            else
            {
             m_qasm.emplace_back(item + "(" + gate_angle + ")" + " q[" + first_qubit + "];");
            }
        }
        break;

    case CNOT_GATE:
    case CZ_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
        {
            if (pQGata->isDagger())
            {
                m_qasm.emplace_back(item + "dg " + all_qubits + ";");
            }
            else
            {
                m_qasm.emplace_back(item + " " + all_qubits + ";");
            }
        }
        break;

    case CPHASE_GATE: 
        {
            string  gate_parameter = to_string(dynamic_cast<angleParameter *>(pQGata->getQGate())->getParameter());
            if (pQGata->isDagger())
            {
                m_qasm.emplace_back(item + "dg(" + gate_parameter + ") " + all_qubits + ";");
            }
            else
            {
                m_qasm.emplace_back(item + "(" + gate_parameter + ") " + all_qubits + ";");
            }
        }
        break;

    case CU_GATE: 
        {
            QuantumGate * gate_parameter = dynamic_cast<QuantumGate *>(pQGata->getQGate());
            if (nullptr == gate_parameter)
            {
                QCERR("gate_parameter is null");
                throw invalid_argument("gate_parameter is null");
            }

            string gate_four_theta = to_string(gate_parameter->getAlpha()) + ',' + to_string(gate_parameter->getBeta())+ ','
                              + to_string(gate_parameter->getDelta()) + ',' + to_string(gate_parameter->getGamma());

            if (pQGata->isDagger())
            {
                m_qasm.emplace_back(item + "dg(" + gate_four_theta + ") " + all_qubits + ";");
            }
            else
            {
                m_qasm.emplace_back(item + "(" + gate_four_theta + ") " + all_qubits + ";");
            }
        }
        break;

    default:m_qasm.emplace_back("NoGateSupport;"); 
        break;
    }

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
    std::string creg_name = pMeasure->getCBit()->getName();

    creg_name = creg_name.substr(1);
    m_qasm.emplace_back("measure q[" + tar_qubit + "]" +" -> "+ "c[" + creg_name + "];");
}


void QProgToQASM::qProgToQASM(AbstractQuantumProgram *pQPro)
{
    if (nullptr == pQPro)
    {
        QCERR("pQPro is null");
        throw invalid_argument("pQPro is null");
    }
    for (auto aiter = pQPro->getFirstNodeIter(); aiter != pQPro->getEndNodeIter(); aiter++)
    {
        QNode * pNode = *aiter;
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

    int type = pNode->getNodeType();

    switch (type)
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

    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        QCERR("Un Support QNode");
        throw invalid_argument("Un Support QNode");
        break;

    case NodeType::MEASURE_GATE:
        qProgToQASM(dynamic_cast<AbstractQuantumMeasure *>(pNode));
        break;

    case NodeType::NODE_UNDEFINED:
        break;

    default:m_qasm.emplace_back("UnSupported ProgNode");
        break;
    }
}

void QProgToQASM::handleDaggerQASM(QNode * pNode,int nodetype)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }
    if (GATE_NODE == nodetype)
    {
        AbstractQGateNode * pGATE = dynamic_cast<AbstractQGateNode *>(pNode);
        if (pGATE->isDagger())
        {
            pGATE->setDagger(false);
            qProgToQASM(pGATE);
        }
        else
        {
            pGATE->setDagger(true);
            qProgToQASM(pGATE);
        }
    }
    else if (CIRCUIT_NODE == nodetype)
    {
        AbstractQuantumCircuit * quantum_circuit = dynamic_cast<AbstractQuantumCircuit *>(pNode);
        if (quantum_circuit->isDagger())
        {
            quantum_circuit->setDagger(false);
            qProgToQASM(quantum_circuit);
        }
        else
        {
            quantum_circuit->setDagger(true);
            qProgToQASM(quantum_circuit);
        }
    }
    else
    {
        QCERR("node type error");
        throw invalid_argument("node type error");
    }
}

void QProgToQASM::qDaggerCirToQASM(QNode * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }
    int type = pNode->getNodeType();
    switch (type)
    {
    case NodeType::GATE_NODE:
        QProgToQASM::handleDaggerQASM(pNode, GATE_NODE);
        break;

    case NodeType::CIRCUIT_NODE: 
        QProgToQASM::handleDaggerQASM(pNode, CIRCUIT_NODE);
        break;

    case NodeType::PROG_NODE:
        qProgToQASM(dynamic_cast<AbstractQuantumProgram *>(pNode));
        break;

    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        QCERR("Error");
        throw invalid_argument("Error");
        break;

    case NodeType::MEASURE_GATE:
        qProgToQASM(dynamic_cast<AbstractQuantumMeasure *>(pNode));
        break;

    case NodeType::NODE_UNDEFINED:
        break;

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
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }

    int node_type = pNode->getNodeType();
    switch (node_type)
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
            QNode * pNode = *aiter;
            qDaggerCirToQASM(pNode);
        }
    }
    else
    {
        for (auto aiter = pCircuit->getFirstNodeIter(); aiter != pCircuit->getEndNodeIter(); aiter++)
        {
            QNode * pNode = *aiter;
            qProgToQASM(pNode);
        }
    }
    
}


string QProgToQASM::insturctionsQASM()
{
    string instructions;
    for (auto &instruct_out : m_qasm)
    {
        instructions.append(instruct_out).append("\n");
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
