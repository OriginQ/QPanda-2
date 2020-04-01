/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

TranformQGateTypeStringAndEnum.cpp
Author: Wangjing
Created in 2018-10-15

Classes for tranform gate type enum and string

*/
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
using namespace std;

USING_QPANDA

TransformQGateType::TransformQGateType()
{
    m_qgate_type_map.insert({ "X", PAULI_X_GATE });
    m_qgate_type_map.insert({ "Y", PAULI_Y_GATE });
    m_qgate_type_map.insert({ "Z", PAULI_Z_GATE });

    m_qgate_type_map.insert({ "X1", X_HALF_PI });
    m_qgate_type_map.insert({ "Y1", Y_HALF_PI });
    m_qgate_type_map.insert({ "Z1", Z_HALF_PI });

    m_qgate_type_map.insert({ "H", HADAMARD_GATE });
    m_qgate_type_map.insert({ "T", T_GATE });
    m_qgate_type_map.insert({ "S", S_GATE });

    m_qgate_type_map.insert({ "RX", RX_GATE });
    m_qgate_type_map.insert({ "RY", RY_GATE });
    m_qgate_type_map.insert({ "RZ", RZ_GATE });

    m_qgate_type_map.insert({ "U1", U1_GATE });
    m_qgate_type_map.insert({ "U2", U2_GATE });
    m_qgate_type_map.insert({ "U3", U3_GATE });
    m_qgate_type_map.insert({ "U4", U4_GATE });

    m_qgate_type_map.insert({ "CU", CU_GATE });
    m_qgate_type_map.insert({ "CNOT", CNOT_GATE });
    m_qgate_type_map.insert({ "CZ", CZ_GATE });
    m_qgate_type_map.insert({ "CPHASE", CPHASE_GATE });

	m_qgate_type_map.insert({ "SWAP", SWAP_GATE });
    m_qgate_type_map.insert({ "ISWAPTheta", ISWAP_THETA_GATE });
    m_qgate_type_map.insert({ "ISWAP", ISWAP_GATE }); 
    m_qgate_type_map.insert({ "SQISWAP", SQISWAP_GATE });
    m_qgate_type_map.insert({ "QDoubleGate", TWO_QUBIT_GATE });

	m_qgate_type_map.insert({ "I", I_GATE });
}

TransformQGateType & TransformQGateType::getInstance()
{
    static TransformQGateType gate_map;
    return gate_map;
}

std::string TransformQGateType::operator[](GateType type)
{
    for (auto & aiter : m_qgate_type_map)
    {
        if (type == aiter.second)
        {
            return aiter.first;
        }
    }

    QCERR("gate name is not support");
    throw invalid_argument("gate name is not support");
}

GateType TransformQGateType::operator[](std::string gate_name)
{
    auto iter = m_qgate_type_map.find(gate_name);
    if (m_qgate_type_map.end() == iter)
    {
        QCERR("gate name is not support");
        throw invalid_argument("gate name is not support");
    }

    return iter->second;
}
