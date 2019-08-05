#include "Core/VirtualQuantumProcessor/PartialAmplitude/MergeMap.h"
using namespace std;
USING_QPANDA

_SINGLE_GATE(Hadamard);
_SINGLE_GATE(X);
_SINGLE_GATE(Y);
_SINGLE_GATE(Z);
_SINGLE_GATE(T);
_SINGLE_GATE(S);
_SINGLE_GATE(P0);
_SINGLE_GATE(P1);

_SINGLE_ANGLE_GATE(RX_GATE);
_SINGLE_ANGLE_GATE(RY_GATE);
_SINGLE_ANGLE_GATE(RZ_GATE);
_SINGLE_ANGLE_GATE(U1_GATE);

_DOUBLE_GATE(CZ);
_DOUBLE_GATE(CNOT);
_DOUBLE_GATE(iSWAP);
_DOUBLE_GATE(SqiSWAP);

_DOUBLE_ANGLE_GATE(CR);

MergeMap::MergeMap()
{
    m_GateFunc.insert(make_pair((unsigned short)GateType::PAULI_X_GATE, _X));
    m_GateFunc.insert(make_pair((unsigned short)GateType::PAULI_Y_GATE, _Y));
    m_GateFunc.insert(make_pair((unsigned short)GateType::PAULI_Z_GATE, _Z));

    m_GateFunc.insert(make_pair((unsigned short)GateType::HADAMARD_GATE, _Hadamard));
    m_GateFunc.insert(make_pair((unsigned short)GateType::T_GATE, _T));
    m_GateFunc.insert(make_pair((unsigned short)GateType::S_GATE, _S));

    m_GateFunc.insert(make_pair((unsigned short)GateType::P0_GATE, _P0));
    m_GateFunc.insert(make_pair((unsigned short)GateType::P1_GATE, _P1));

    m_GateFunc.insert(make_pair((unsigned short)GateType::RX_GATE, _RX_GATE));
    m_GateFunc.insert(make_pair((unsigned short)GateType::RY_GATE, _RY_GATE));
    m_GateFunc.insert(make_pair((unsigned short)GateType::RZ_GATE, _RZ_GATE));
    m_GateFunc.insert(make_pair((unsigned short)GateType::U1_GATE, _U1_GATE));

    m_GateFunc.insert(make_pair((unsigned short)GateType::CZ_GATE, _CZ));
    m_GateFunc.insert(make_pair((unsigned short)GateType::CNOT_GATE, _CNOT));
    m_GateFunc.insert(make_pair((unsigned short)GateType::ISWAP_GATE, _iSWAP));
    m_GateFunc.insert(make_pair((unsigned short)GateType::SQISWAP_GATE, _SqiSWAP));

    m_GateFunc.insert(make_pair((unsigned short)GateType::CPHASE_GATE, _CR));

    m_key_map.insert(make_pair(GateType::CNOT_GATE, GateType::PAULI_X_GATE));
    m_key_map.insert(make_pair(GateType::CZ_GATE, GateType::PAULI_Z_GATE));
    m_key_map.insert(make_pair(GateType::CPHASE_GATE, GateType::U1_GATE));
}

void MergeMap::traversalMap
(std::vector<QGateNode> &prog_map, QPUImpl *pQGate, QuantumGateParam* pGateParam)
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

void MergeMap::traversalQlist(std::vector<QGateNode> &QCir)
{
    for (size_t i = 0; i < QCir.size(); ++i)
    {
        auto iter = m_key_map.find(QCir[i].gate_type);
        if (m_key_map.end() != iter)
        {
            if (isCorssNode(QCir[i].ctr_qubit, QCir[i].tar_qubit))
            {
                vector<QGateNode> P0_Cir = QCir;
                vector<QGateNode> P1_Cir = QCir;

                QGateNode P0_Node = { P0_GATE , QCir[i].isConjugate };
                P0_Node.tar_qubit = QCir[i].ctr_qubit;
                P0_Cir[i] = P0_Node;

                QGateNode P1_Node1 = { P1_GATE , QCir[i].isConjugate };
                P1_Node1.tar_qubit = QCir[i].ctr_qubit;

                QGateNode P1_Node2 = { iter->second ,QCir[i].isConjugate };
                P1_Node2.tar_qubit = QCir[i].tar_qubit;
                P1_Node2.gate_parm = QCir[i].gate_parm;

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

        QlistMap.insert(make_pair(false, Cir0));
        QlistMap.insert(make_pair(true, Cir1));

        m_circuit_vec.emplace_back(QlistMap);
    }
}

bool MergeMap::isCorssNode(size_t ctr, size_t tar)
{
    return ((ctr >= m_qubit_num / 2) && 
           (tar < m_qubit_num / 2)) ||
           ((tar >= m_qubit_num / 2) && 
           (ctr < m_qubit_num / 2));
}
