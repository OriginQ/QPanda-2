#include "Core/Utilities/Transform/QProgToQRunes.h"
#include "QPanda.h"
using namespace std;
USING_QPANDA

QProgToQRunes::QProgToQRunes(QuantumMachine * quantum_machine)
{
    m_gatetype.insert(pair<int, string>(PAULI_X_GATE, "X"));
    m_gatetype.insert(pair<int, string>(PAULI_Y_GATE, "Y"));
    m_gatetype.insert(pair<int, string>(PAULI_Z_GATE, "Z"));

    m_gatetype.insert(pair<int, string>(X_HALF_PI, "X1"));
    m_gatetype.insert(pair<int, string>(Y_HALF_PI, "Y1"));
    m_gatetype.insert(pair<int, string>(Z_HALF_PI, "Z1"));

    m_gatetype.insert(pair<int, string>(HADAMARD_GATE, "H"));
    m_gatetype.insert(pair<int, string>(T_GATE,        "T"));
    m_gatetype.insert(pair<int, string>(S_GATE,        "S"));

    m_gatetype.insert(pair<int, string>(RX_GATE, "RX"));
    m_gatetype.insert(pair<int, string>(RY_GATE, "RY"));
    m_gatetype.insert(pair<int, string>(RZ_GATE, "RZ"));

    m_gatetype.insert(pair<int, string>(U1_GATE, "U1"));
    m_gatetype.insert(pair<int, string>(U2_GATE, "U2"));
    m_gatetype.insert(pair<int, string>(U3_GATE, "U3"));
    m_gatetype.insert(pair<int, string>(U4_GATE, "U4"));

    m_gatetype.insert(pair<int, string>(CU_GATE, "CU"));
    m_gatetype.insert(pair<int, string>(CNOT_GATE, "CNOT"));
    m_gatetype.insert(pair<int, string>(CZ_GATE, "CZ"));
    m_gatetype.insert(pair<int, string>(CPHASE_GATE, "CR"));
    m_gatetype.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_gatetype.insert(pair<int, string>(SWAP_GATE, "SWAP"));
    m_gatetype.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));
    m_gatetype.insert(pair<int, string>(TWO_QUBIT_GATE, "QDoubleGate"));
    m_QRunes.clear();

    m_quantum_machine = quantum_machine;
}

void QProgToQRunes::transformQGate(AbstractQGateNode * pQGate)
{
    if (nullptr == pQGate || nullptr == pQGate->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    QVec qubits_vector;
    QVec ctr_qubits_vector;

    std::string all_ctr_qubits;
    pQGate->getQuBitVector(qubits_vector);
    pQGate->getControlVector(ctr_qubits_vector);

    if (pQGate->isDagger())
    {
        m_QRunes.emplace_back("DAGGER");
    }
    if (!ctr_qubits_vector.empty())
    {
        for (auto val : ctr_qubits_vector)
        {
            all_ctr_qubits = all_ctr_qubits + to_string(val->getPhysicalQubitPtr()->getQubitAddr()) + ",";
        }
        m_QRunes.emplace_back("CONTROL " + all_ctr_qubits.substr(0, all_ctr_qubits.length() - 1));
    }

    auto iter = m_gatetype.find(pQGate->getQGate()->getGateType());
    if (iter == m_gatetype.end())
    {
        QCERR("unknow error");
        throw runtime_error("unknow error");
    }

    string first_qubit = to_string(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());

    string all_qubits;
    for (auto _val : qubits_vector)
    {
        all_qubits = all_qubits + to_string(_val->getPhysicalQubitPtr()->getQubitAddr()) + ",";
    }
    all_qubits = all_qubits.substr(0, all_qubits.length() - 1);

    string item = iter->second;
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
    case U4_GATE:
        {
            m_QRunes.emplace_back(item + " " + first_qubit);
        }
        break;

    case U1_GATE:
    case RX_GATE:
    case RY_GATE:
    case RZ_GATE:
        {
            angleParameter * gate_parameter = dynamic_cast<angleParameter *>(pQGate->getQGate());
            string  gate_angle = to_string(gate_parameter->getParameter());
            m_QRunes.emplace_back(item + " " + first_qubit + ","+"\"" + gate_angle+"\"");
        }
        break;

    case CNOT_GATE:
    case CZ_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
    case SWAP_GATE:
    case TWO_QUBIT_GATE:
        {
            m_QRunes.emplace_back(item + " " + all_qubits);
        }
        break;

    case CPHASE_GATE: 
        {
            angleParameter * gate_parameter = dynamic_cast<angleParameter *>(pQGate->getQGate());
            string  gate_theta = to_string(gate_parameter->getParameter());
            m_QRunes.emplace_back(item + " " + all_qubits + "," + "\"" + gate_theta + "\"");
        }
        break;

    case CU_GATE: 
       { 
            QuantumGate * gate_parameter = dynamic_cast<QuantumGate *>(pQGate->getQGate());
            string gate_four_theta = to_string(gate_parameter->getAlpha()) + ',' + 
                                     to_string(gate_parameter->getBeta())  + ',' +
                                     to_string(gate_parameter->getDelta()) + ',' + 
                                     to_string(gate_parameter->getGamma());
            m_QRunes.emplace_back(item + " " + all_qubits + "," + gate_four_theta);
       }
       break;

    default:m_QRunes.emplace_back("UnSupported GateNode");
    }
    if (!ctr_qubits_vector.empty())
    {
        m_QRunes.emplace_back("ENCONTROL " + all_ctr_qubits);
    }
    if (pQGate->isDagger())
    {
        m_QRunes.emplace_back("ENDAGGER");
    }
}


void QProgToQRunes::transformQMeasure(AbstractQuantumMeasure *pMeasure)
{
    if (nullptr == pMeasure || nullptr == pMeasure->getQuBit()->getPhysicalQubitPtr())
    {
        QCERR("pMeasure is null");
        throw invalid_argument("pMeasure is null");
    }

    PhysicalQubit* qbitPtr = pMeasure->getQuBit()->getPhysicalQubitPtr();
    std::string tar_qubit = to_string(qbitPtr->getQubitAddr());
    std::string creg_name = pMeasure->getCBit()->getName();

    m_QRunes.emplace_back("MEASURE " + tar_qubit + ",$" + creg_name.substr(1));
}

void QProgToQRunes::transformQProg(AbstractQuantumProgram *pQProg)
{
    if (nullptr == pQProg)
    {
        QCERR("pQProg is null");
        throw runtime_error("pQProg is null");
    }

    for (auto aiter = pQProg->getFirstNodeIter(); aiter != pQProg->getEndNodeIter(); aiter++)
    {
        QNode * pNode = (*aiter).get();
        transformQNode(pNode);
    }
}

void QProgToQRunes::transformQNode(QNode * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw runtime_error("pNode is null");
    }

    switch (pNode->getNodeType())
    {
    case NodeType::GATE_NODE:
        QProgToQRunes::transformQGate(dynamic_cast<AbstractQGateNode *>(pNode));
        break;

    case NodeType::CIRCUIT_NODE:
        QProgToQRunes::transformQCircuit(dynamic_cast<AbstractQuantumCircuit *>(pNode));
        break;

    case NodeType::PROG_NODE:
        QProgToQRunes::transformQProg(dynamic_cast<AbstractQuantumProgram *>(pNode));
        break;

    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        QProgToQRunes::transformQControlFlow(dynamic_cast<AbstractControlFlowNode *>(pNode));
        break;

    case NodeType::MEASURE_GATE:
        QProgToQRunes::transformQMeasure(dynamic_cast<AbstractQuantumMeasure *>(pNode));
        break;

    case NodeType::NODE_UNDEFINED:
    default:m_QRunes.emplace_back("UnSupported Node");
        break;
    }
}


static void traversalInOrderPCtr(const CExpr* pCtrFlow,string &ctr_statement)
{
    if (nullptr != pCtrFlow)
    {
        traversalInOrderPCtr(pCtrFlow->getLeftExpr(), ctr_statement);
        ctr_statement = ctr_statement + pCtrFlow->getName();
        traversalInOrderPCtr(pCtrFlow->getRightExpr(), ctr_statement);
    }
}


void QProgToQRunes::transformQControlFlow(AbstractControlFlowNode * pCtrFlow)
{
    if (nullptr == pCtrFlow)
    {
        QCERR("pCtrFlow is null");
        throw runtime_error("pCtrFlow is null");
    }

    QNode *pNode = dynamic_cast<QNode *>(pCtrFlow);
    switch (pNode->getNodeType())
    {
    case NodeType::WHILE_START_NODE:
        {
            string exper;
            auto expr = pCtrFlow->getCExpr()->getExprPtr().get();

            traversalInOrderPCtr(expr, exper);
            m_QRunes.emplace_back("QWHILE " + exper);

            QNode *truth_branch_node = pCtrFlow->getTrueBranch();
            if (nullptr != truth_branch_node)
            {
                transformQNode(truth_branch_node);
            }

            m_QRunes.emplace_back("ENDQWHILE");
        }
        break;

    case NodeType::QIF_START_NODE:
        {
            string exper;
            auto expr = pCtrFlow->getCExpr()->getExprPtr().get();

            traversalInOrderPCtr(expr, exper);
            m_QRunes.emplace_back("QIF "+exper);

            QNode * truth_branch_node = pCtrFlow->getTrueBranch();
            if (nullptr != truth_branch_node)
            {
                transformQNode(truth_branch_node);
            }
            m_QRunes.emplace_back("ELSE");

            QNode *false_branch_node = pCtrFlow->getFalseBranch();
            if (nullptr != false_branch_node)
            {
                transformQNode(false_branch_node);
            }
            m_QRunes.emplace_back("ENDQIF");
        }
        break;

    }
}


void QProgToQRunes::transformQCircuit(AbstractQuantumCircuit * pCircuit)
{
    if (nullptr == pCircuit)
    {
        QCERR("pCircuit is null");
        throw runtime_error("pCircuit is null");
    }
    if (pCircuit->isDagger())
    {
        m_QRunes.emplace_back("DAGGER");
    }

    QVec circuit_ctr_qubits;
    string all_ctr_qubits;
    pCircuit->getControlVector(circuit_ctr_qubits);
    if (!circuit_ctr_qubits.empty())
    {
        for (auto val : circuit_ctr_qubits)
        {
            all_ctr_qubits = all_ctr_qubits + to_string(val->getPhysicalQubitPtr()->getQubitAddr()) + ",";
        }
        all_ctr_qubits = all_ctr_qubits.substr(0, all_ctr_qubits.length() - 1);
        m_QRunes.emplace_back("CONTROL " + all_ctr_qubits);
    }
    for (auto aiter = pCircuit->getFirstNodeIter(); aiter != pCircuit->getEndNodeIter(); aiter++)
    {
        QNode * pNode = (*aiter).get();
        transformQNode(pNode);
    }
    if (!circuit_ctr_qubits.empty())
    {
        m_QRunes.emplace_back("ENCONTROL " + all_ctr_qubits);
    }
    if (pCircuit->isDagger())
    {
        m_QRunes.emplace_back("ENDAGGER");
    }

}


string QProgToQRunes::getInsturctions()
{
    string instructions;

    for (auto &instruct_out : m_QRunes)
    {
        instructions.append(instruct_out).append("\n");
    }
    instructions.erase(instructions.size() - 1);

    return instructions;
}
