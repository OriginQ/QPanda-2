#include "Core/VirtualQuantumProcessor/PartialAmplitude/MergeMap.h"
using namespace std;
USING_QPANDA

SINGLE_GATE(Hadamard);
SINGLE_GATE(X);
SINGLE_GATE(Y);
SINGLE_GATE(Z);
SINGLE_GATE(T);
SINGLE_GATE(S);
SINGLE_GATE(P0);
SINGLE_GATE(P1);

SINGLE_ANGLE_GATE(RX_GATE);
SINGLE_ANGLE_GATE(RY_GATE);
SINGLE_ANGLE_GATE(RZ_GATE);

DOUBLE_GATE(CZ);
DOUBLE_GATE(CNOT);
DOUBLE_GATE(iSWAP);
DOUBLE_GATE(SqiSWAP);

DOUBLE_ANGLE_GATE(CR);

MergeMap::MergeMap()
{
    m_GateFunc.insert(make_pair((unsigned short)GateType::PAULI_X_GATE, X_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::PAULI_Y_GATE, Y_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::PAULI_Z_GATE, Z_Gate));

    m_GateFunc.insert(make_pair((unsigned short)GateType::HADAMARD_GATE, Hadamard_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::T_GATE, T_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::S_GATE, S_Gate));

    m_GateFunc.insert(make_pair((unsigned short)GateType::P0_GATE, P0_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::P1_GATE, P1_Gate));

    m_GateFunc.insert(make_pair((unsigned short)GateType::RX_GATE, RX_GATE_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::RY_GATE, RY_GATE_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::RZ_GATE, RZ_GATE_Gate));

    m_GateFunc.insert(make_pair((unsigned short)GateType::CZ_GATE, CZ_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::CNOT_GATE, CNOT_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::ISWAP_GATE, iSWAP_Gate));
    m_GateFunc.insert(make_pair((unsigned short)GateType::SQISWAP_GATE, SqiSWAP_Gate));

    m_GateFunc.insert(make_pair((unsigned short)GateType::CPHASE_GATE, CR_Gate));

    m_key_map.insert(make_pair(GateType::CNOT_GATE, GateType::PAULI_X_GATE));
    m_key_map.insert(make_pair(GateType::CZ_GATE, GateType::PAULI_Z_GATE));
}

void MergeMap::traversalMap(std::vector<QGateNode> &prog_map, QPUImpl *pQGate, QuantumGateParam* pGateParam)
{
    if (nullptr == pQGate || nullptr == pGateParam)
    {
        QCERR("Error");
        throw invalid_argument("Error");
    }
    for (auto val : prog_map)
    {
        auto iter = m_GateFunc.find(val.gate_type);
        if (iter == m_GateFunc.end())
        {
            QCERR("Error");
            throw invalid_argument("Error");
        }
        else
        {
            iter->second(val, pQGate);
        }
    }
}

void MergeMap::traversalAll(AbstractQuantumProgram *pQProg)
{
    m_qubit_num = getAllocateQubitNum();
    if (nullptr == pQProg || m_qubit_num <= 0 || m_qubit_num % 2 != 0)
    {
        QCERR("Error");
        throw invalid_argument("Error");
    }

    TraversalQProg::traversal(pQProg);
    MergeMap::traversalQlist(m_circuit);
}

void MergeMap::traversal(AbstractQGateNode *pQGate)
{
    if (nullptr == pQGate || nullptr == pQGate->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    QVec qubits_vector;
    pQGate->getQuBitVector(qubits_vector);

    auto gate_type = (unsigned short)pQGate->getQGate()->getGateType();
    QGateNode node = { gate_type,pQGate->isDagger() };
    switch (gate_type)
    {
    case GateType::P0_GATE:
    case GateType::P1_GATE:
    case GateType::PAULI_X_GATE:
    case GateType::PAULI_Y_GATE:
    case GateType::PAULI_Z_GATE:
    case GateType::X_HALF_PI:
    case GateType::Y_HALF_PI:
    case GateType::Z_HALF_PI:
    case GateType::HADAMARD_GATE:
    case GateType::T_GATE:
    case GateType::S_GATE:
    {
        node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
    }
    break;

    case GateType::U1_GATE:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::RZ_GATE:
    {
        node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        node.gate_parm = dynamic_cast<angleParameter *>(pQGate->getQGate())->getParameter();
    }
    break;

    case GateType::ISWAP_GATE:
    case GateType::SQISWAP_GATE:
    {
        auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
        if (ctr_qubit == tar_qubit || isCorssNode(ctr_qubit, tar_qubit))
        {
            QCERR("Error");
            throw qprog_syntax_error();
        }
        else
        {
            node.ctr_qubit = ctr_qubit;
            node.tar_qubit = tar_qubit;
        }
    }
    break;

    case GateType::CNOT_GATE:
    case GateType::CZ_GATE:
    {
        node.ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        node.tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
    }
    break;

    case GateType::CPHASE_GATE:
    {
        auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
        if (ctr_qubit == tar_qubit || isCorssNode(ctr_qubit, tar_qubit))
        {
            QCERR("Error");
            throw qprog_syntax_error();
        }
        else
        {
            node.ctr_qubit = ctr_qubit;
            node.tar_qubit = tar_qubit;
            node.gate_parm = dynamic_cast<angleParameter *>(pQGate->getQGate())->getParameter();
        }
    }
    break;

    default:
        {
            QCERR("UnSupported QGate Node");
            throw undefine_error("QGate");
        }
    break;
    }
    m_circuit.emplace_back(node);
}

void MergeMap::traversalQlist(std::vector<QGateNode> &QCir)
{
    for (size_t i = 0; i < QCir.size(); ++i)
    {
        if (QCir[i].gate_type == GateType::CZ_GATE ||
            QCir[i].gate_type == GateType::CNOT_GATE)
        {
            if (isCorssNode(QCir[i].ctr_qubit, QCir[i].tar_qubit))
            {
                vector<QGateNode> P0_Cir = QCir;
                vector<QGateNode> P1_Cir = QCir;

                QGateNode P0_Node;
                P0_Node.gate_type = P0_GATE;
                P0_Node.tar_qubit = QCir[i].ctr_qubit;
                P0_Cir[i] = P0_Node;

                QGateNode P1_Node1;
                P1_Node1.gate_type = P1_GATE;
                P1_Node1.tar_qubit = QCir[i].ctr_qubit;

                QGateNode P1_Node2;
                P1_Node2.gate_type = m_key_map.find(QCir[i].gate_type)->second;
                P1_Node2.tar_qubit = QCir[i].tar_qubit;

                P1_Cir[i] = P1_Node2;
                P1_Cir.emplace(P1_Cir.begin() + i, P1_Node1);

                traversalQlist(P0_Cir);
                traversalQlist(P1_Cir);

                splitQlist(P0_Cir);
                splitQlist(P1_Cir);
                break;
            }
        }
    }
}

void MergeMap::splitQlist(std::vector<QGateNode> &QCir)
{
    bool is_operater{ true };
    for (auto val : QCir)
    {
        auto iter = m_key_map.find(val.gate_type);
        if (iter != m_key_map.end() && isCorssNode(val.tar_qubit, val.ctr_qubit))
        {
            is_operater = false;
            break;
        }
    }

    if (is_operater)
    {
        vector<QGateNode> Cir0, Cir1;
        for (auto val : QCir)
        {
            QGateNode node = { val.gate_type,val.isConjugate };
            switch (val.gate_type)
            {
            case GateType::P0_GATE:
            case GateType::P1_GATE:
            case GateType::HADAMARD_GATE:
            case GateType::T_GATE:
            case GateType::S_GATE:
            case GateType::PAULI_X_GATE:
            case GateType::PAULI_Y_GATE:
            case GateType::PAULI_Z_GATE:
            case GateType::X_HALF_PI:
            case GateType::Y_HALF_PI:
            case GateType::Z_HALF_PI:
            {
                if (val.tar_qubit < (m_qubit_num / 2))
                {
                    node.tar_qubit = val.tar_qubit;
                    Cir0.emplace_back(node);
                }
                else
                {
                    node.tar_qubit = val.tar_qubit - (m_qubit_num / 2);
                    Cir1.emplace_back(node);
                }
            }
            break;

            case GateType::U1_GATE:
            case GateType::RX_GATE:
            case GateType::RY_GATE:
            case GateType::RZ_GATE:
            {
                if (val.tar_qubit < (m_qubit_num / 2))
                {
                    node.tar_qubit = val.tar_qubit;
                    node.gate_parm = val.gate_parm;
                    Cir0.emplace_back(node);
                }
                else
                {
                    node.gate_parm = val.gate_parm;
                    node.tar_qubit = val.tar_qubit - (m_qubit_num / 2);
                    Cir1.emplace_back(node);
                }
            }
            break;

            case GateType::CNOT_GATE:
            case GateType::CZ_GATE:
            case GateType::ISWAP_GATE:
            case GateType::SQISWAP_GATE:
            {
                if ((val.tar_qubit <= (m_qubit_num / 2)) &&
                    (val.ctr_qubit <= (m_qubit_num / 2)))
                {
                    node.tar_qubit = val.tar_qubit;
                    node.ctr_qubit = val.ctr_qubit;
                    Cir0.emplace_back(node);
                }
                else
                {
                    node.tar_qubit = val.tar_qubit - (m_qubit_num / 2),
                        node.ctr_qubit = val.ctr_qubit - (m_qubit_num / 2);
                    Cir1.emplace_back(node);
                }
            }
            break;

            case GateType::CPHASE_GATE:
            {
                if ((val.tar_qubit <= (m_qubit_num / 2)) &&
                    (val.ctr_qubit <= (m_qubit_num / 2)))
                {
                    node.tar_qubit = val.tar_qubit;
                    node.ctr_qubit = val.ctr_qubit;
                    node.gate_parm = val.gate_parm;
                    Cir0.emplace_back(node);
                }
                else
                {
                    node.tar_qubit = val.tar_qubit - (m_qubit_num / 2),
                        node.ctr_qubit = val.ctr_qubit - (m_qubit_num / 2);
                    node.gate_parm = val.gate_parm;
                    Cir1.emplace_back(node);
                }
            }
            break;

            default:
                {
                    QCERR("UnSupported QGate Node");
                    throw undefine_error("QGate");
                }
                break;
            }
        }

        map<bool, vector<QGateNode>> QlistMap;
        if (!Cir0.empty())
        {
            QlistMap.insert(make_pair(false, Cir0));
        }
        if (!Cir1.empty())
        {
            QlistMap.insert(make_pair(true, Cir1));
        }
        m_circuit_vec.emplace_back(QlistMap);
    }
}

bool MergeMap::isCorssNode(size_t ctr, size_t tar)
{
    return ((ctr >= m_qubit_num / 2) && (tar < m_qubit_num / 2)) ||
        ((tar >= m_qubit_num / 2) && (ctr < m_qubit_num / 2));
}

