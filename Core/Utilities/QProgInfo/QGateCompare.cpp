#include "Core/Utilities/QProgInfo/QGateCompare.h"
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

QGateCompare::QGateCompare(const std::vector<std::vector<string> > &gates) :
    m_gates(gates), m_count(0)
{

}

size_t QGateCompare::count()
{
    return m_count;
}

void QGateCompare::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
    int gate_type = cur_node->getQGate()->getGateType();
    auto iter = s_gateTypeInt_map_gateTypeStr.find(gate_type);
    if (iter == s_gateTypeInt_map_gateTypeStr.end())
    {
        QCERR("gate is error");
        throw invalid_argument("gate is error");
    }

    string item = iter->second;
    for (auto &val : m_gates)
    {
        auto iter_gate = std::find(val.begin(), val.end(), item);
        if (iter_gate != val.end())
        {
            return ;
        }
    }

    m_count++;
    return ;
}

