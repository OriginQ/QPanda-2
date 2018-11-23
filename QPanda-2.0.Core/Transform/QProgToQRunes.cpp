#include <iostream>
#include "QProgToQRunes.h"
#include "QPanda.h"
using QGATE_SPACE::angleParameter;

QProgToQRunes::QProgToQRunes()
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
    m_gatetype.insert(pair<int, string>(CPHASE_GATE, "CPHASE"));
    m_gatetype.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_gatetype.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));
    m_qrunes.clear();
}


QProgToQRunes::~QProgToQRunes()
{
}


void QProgToQRunes::qProgToQRunes(AbstractQuantumProgram * pQpro)
{
    if (nullptr == pQpro)
    {
        throw param_error_exception("pQPro is null", false);
    }
    m_qrunes.emplace_back("#QUBITS_NUM " + to_string(getAllocateQubitNum()));
    m_qrunes.emplace_back("#CREGS_NUM " + to_string(getAllocateCMem()));

    QProgToQRunes::progToQRunes(pQpro);

}


void QProgToQRunes::progToQRunes(AbstractQGateNode * pQGata)
{
    if (nullptr == pQGata)
    {
        throw exception();
    }
    if (nullptr == pQGata->getQGate())
    {
        throw exception();
    }

    int gate_type = pQGata->getQGate()->getGateType();

    vector<Qubit*> qubits_vector;
    vector<Qubit*> ctr_qubits_vector;

    std::string all_ctr_qubits;
    pQGata->getQuBitVector(qubits_vector);
    pQGata->getControlVector(ctr_qubits_vector);

    if (pQGata->isDagger())
    {
        m_qrunes.emplace_back("DAGGER");
    }
    if (!ctr_qubits_vector.empty())
    {
        for (auto val : ctr_qubits_vector)
        {
            all_ctr_qubits = all_ctr_qubits + to_string(val->getPhysicalQubitPtr()->getQubitAddr()) + ",";
        }
        all_ctr_qubits = all_ctr_qubits.substr(0, all_ctr_qubits.length() - 1);
        m_qrunes.emplace_back("CONTROL " + all_ctr_qubits);
    }

    auto iter = m_gatetype.find(gate_type);
    if (iter == m_gatetype.end())
    {
        throw exception();
    }

    string item = iter->second;
    string first_qubit = to_string(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
    string all_qubits;

    for (auto _val : qubits_vector)
    {
        all_qubits = all_qubits + to_string(_val->getPhysicalQubitPtr()->getQubitAddr()) + ",";
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
            m_qrunes.emplace_back(item + " " + first_qubit);
        }
        break;

    case U1_GATE:
    case RX_GATE:
    case RY_GATE:
    case RZ_GATE:
        {
            angleParameter * gate_parameter = dynamic_cast<angleParameter *>(pQGata->getQGate());
            if (nullptr == gate_parameter)
            {
              throw exception();
            }
            string  gate_angle = to_string(gate_parameter->getParameter());
            m_qrunes.emplace_back(item + " " + first_qubit + ","+"\"" + gate_angle+"\"");
        }
        break;

    case CNOT_GATE:
    case CZ_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
        {
            m_qrunes.emplace_back(item + " " + all_qubits);
        }
        break;

    case CPHASE_GATE: 
        {
            angleParameter * gate_parameter = dynamic_cast<angleParameter *>(pQGata->getQGate());
            if (nullptr == gate_parameter)
            {
                throw exception();
            }
            string  gate_theta = to_string(gate_parameter->getParameter());
            m_qrunes.emplace_back(item + " " + all_qubits + "," + gate_theta);
        }
        break;

    case CU_GATE: 
       { 
            QuantumGate * gate_parameter = dynamic_cast<QuantumGate *>(pQGata->getQGate());
            if (nullptr == gate_parameter)
                {
                    throw exception();
                }
            string gate_four_theta = to_string(gate_parameter->getAlpha()) + ',' + 
                                to_string(gate_parameter->getBeta())  + ',' +
                                to_string(gate_parameter->getDelta()) + ',' + 
                                to_string(gate_parameter->getGamma());
            m_qrunes.emplace_back(item + " " + all_qubits + "," + gate_four_theta);
       }
       break;

    default:m_qrunes.emplace_back("NoGateSupport");
    }
    if (!ctr_qubits_vector.empty())
    {
        m_qrunes.emplace_back("ENCONTROL " + all_ctr_qubits);
    }
    if (pQGata->isDagger())
    {
        m_qrunes.emplace_back("ENDAGGER");
    }
}


void QProgToQRunes::progToQRunes(AbstractQuantumMeasure *pMeasure)
{
    if (nullptr == pMeasure)
    {
        throw param_error_exception("pMeasure is null", false);
    }

    PhysicalQubit* qbitPtr = pMeasure->getQuBit()->getPhysicalQubitPtr();

    if (nullptr == qbitPtr)
    {
        throw exception();
    }

    std::string tar_qubit = to_string(qbitPtr->getQubitAddr());
    std::string creg_name = pMeasure->getCBit()->getName();

    creg_name = creg_name.substr(1);
    m_qrunes.emplace_back("MEASURE " + tar_qubit + ",$" + creg_name);
}


void QProgToQRunes::progToQRunes(AbstractQuantumProgram *pQPro)
{
    if (nullptr == pQPro)
    {
        throw param_error_exception("pQPro is null", false);
    }
    for (auto aiter = pQPro->getFirstNodeIter(); aiter != pQPro->getEndNodeIter(); aiter++)
    {
        QNode * pNode = *aiter;
        progToQRunes(pNode);
    }
}


void QProgToQRunes::progToQRunes(QNode * pNode)
{
    if (nullptr == pNode)
    {
        throw param_error_exception("pNode is null", false);
    }

    int type = pNode->getNodeType();

    switch (type)
    {
    case NodeType::GATE_NODE:
        progToQRunes(dynamic_cast<AbstractQGateNode *>(pNode));
        break;

    case NodeType::CIRCUIT_NODE:
        progToQRunes(dynamic_cast<AbstractQuantumCircuit *>(pNode));
        break;

    case NodeType::PROG_NODE:
        progToQRunes(dynamic_cast<AbstractQuantumProgram *>(pNode)); 
        break;

    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        progToQRunes(dynamic_cast<AbstractControlFlowNode *>(pNode));
        break;

    case NodeType::MEASURE_GATE:
        progToQRunes(dynamic_cast<AbstractQuantumMeasure *>(pNode));
        break;

    case NodeType::NODE_UNDEFINED:
        break;

    default:m_qrunes.emplace_back("UnSupported ProgNode");
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


void QProgToQRunes::progToQRunes(AbstractControlFlowNode * pCtrFlow)
{
    if (nullptr == pCtrFlow)
    {
        throw param_error_exception("pCtrFlow is null", false);
    }

    QNode *pNode = dynamic_cast<QNode *>(pCtrFlow);
    if (nullptr == pNode)
    {
        throw param_error_exception("pNode is null", false);
    }

    int node_type = pNode->getNodeType();

    switch (node_type)
    {
    case NodeType::WHILE_START_NODE:
        {
            string exper;
            CExpr* expr = pCtrFlow->getCExpr()->getExprPtr();
            if (nullptr == expr)
            {
                throw param_error_exception("pNode is null", false);
            }
            traversalInOrderPCtr(expr, exper);
            m_qrunes.emplace_back("QWHILE " + exper);

            QNode *truth_branch_node = pCtrFlow->getTrueBranch();
            if (nullptr != truth_branch_node)
            {
                progToQRunes(truth_branch_node);
            }

            m_qrunes.emplace_back("ENDQWHILE");
        }
        break;

    case NodeType::QIF_START_NODE:
        {
            string exper;

            CExpr* expr = pCtrFlow->getCExpr()->getExprPtr();

            if (nullptr == expr)
            {
                throw param_error_exception("pNode is null", false);
            }
            traversalInOrderPCtr(expr, exper);
            m_qrunes.emplace_back("QIF "+exper);

            QNode * truth_branch_node = pCtrFlow->getTrueBranch();
            if (nullptr != truth_branch_node)
            {
                progToQRunes(truth_branch_node);
            }
            m_qrunes.emplace_back("ELSE");

            QNode *false_branch_node = pCtrFlow->getFalseBranch();
            if (nullptr != false_branch_node)
            {
                progToQRunes(false_branch_node);
            }
            m_qrunes.emplace_back("ENDQIF");
        }
        break;

    }
}


void QProgToQRunes::progToQRunes(AbstractQuantumCircuit * pCircuit)
{
    if (nullptr == pCircuit)
    {
        throw param_error_exception("pCircuit is null", false);
    }
    if (pCircuit->isDagger())
    {
        m_qrunes.emplace_back("DAGGER");
    }

    std::vector<Qubit*> circuit_ctr_qubits;

    string all_ctr_qubits;

    pCircuit->getControlVector(circuit_ctr_qubits);
    if (!circuit_ctr_qubits.empty())
    {
        for (auto val : circuit_ctr_qubits)
        {
            all_ctr_qubits = all_ctr_qubits + to_string(val->getPhysicalQubitPtr()->getQubitAddr()) + ",";

        }
        all_ctr_qubits = all_ctr_qubits.substr(0, all_ctr_qubits.length() - 1);
        m_qrunes.emplace_back("CONTROL " + all_ctr_qubits);
    }
    for (auto aiter = pCircuit->getFirstNodeIter(); aiter != pCircuit->getEndNodeIter(); aiter++)
    {
        QNode * pNode = *aiter;
        progToQRunes(pNode);
    }
    if (!circuit_ctr_qubits.empty())
    {
        m_qrunes.emplace_back("ENCONTROL " + all_ctr_qubits);
    }
    if (pCircuit->isDagger())
    {
        m_qrunes.emplace_back("ENDAGGER");
    }

}


string QProgToQRunes::insturctionsQRunes()
{
    string instructions;

    for (auto &instruct_out : m_qrunes)
    {
        instructions.append(instruct_out).append("\n");
    }
    instructions.erase(instructions.size() - 1);

    return instructions;
}


ostream & operator<<(ostream & out, const QProgToQRunes &qrunes_prog)
{
    for (auto instruct_out : qrunes_prog.m_qrunes)
    {
        out << instruct_out << "\n";
    }

    return out;
}
