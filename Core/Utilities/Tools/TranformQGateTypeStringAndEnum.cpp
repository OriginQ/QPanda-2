/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

TranformQGateTypeStringAndEnum.cpp
Author: Wangjing
Created in 2018-10-15

Classes for tranform gate type enum and string

*/
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
using namespace std;

USING_QPANDA

bool QPanda::is_single_gate(const GateType gt_type) {
    if (gt_type < 0)
    {
        return false;
    }

    switch (gt_type)
    {
    case P0_GATE:
    case P1_GATE:
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
    case RPHI_GATE:
    case U1_GATE:
    case U2_GATE:
    case U3_GATE:
    case U4_GATE:
    case I_GATE:
    case ECHO_GATE:
        return true;
    }

    return false;
}

TransformQGateType::TransformQGateType()
{
    m_qgate_type_map.insert({ GATE_X, PAULI_X_GATE });
    m_qgate_type_map.insert({ GATE_Y, PAULI_Y_GATE });
    m_qgate_type_map.insert({ GATE_Z, PAULI_Z_GATE });

    m_qgate_type_map.insert({ GATE_X1, X_HALF_PI });
    m_qgate_type_map.insert({ GATE_Y1, Y_HALF_PI });
    m_qgate_type_map.insert({ GATE_Z1, Z_HALF_PI });

    m_qgate_type_map.insert({ GATE_P, P_GATE });

    m_qgate_type_map.insert({ GATE_H, HADAMARD_GATE });
    m_qgate_type_map.insert({ GATE_T, T_GATE });
    m_qgate_type_map.insert({ GATE_S, S_GATE });

    m_qgate_type_map.insert({ GATE_RX, RX_GATE });
    m_qgate_type_map.insert({ GATE_RY, RY_GATE });
    m_qgate_type_map.insert({ GATE_RZ, RZ_GATE });
    m_qgate_type_map.insert({ GATE_RPHI, RPHI_GATE });

    m_qgate_type_map.insert({ GATE_U1, U1_GATE });
    m_qgate_type_map.insert({ GATE_U2, U2_GATE });
    m_qgate_type_map.insert({ GATE_U3, U3_GATE });
    m_qgate_type_map.insert({ GATE_U4, U4_GATE });

    m_qgate_type_map.insert({ GATE_CU, CU_GATE });
    m_qgate_type_map.insert({ GATE_CP, CP_GATE });
	m_qgate_type_map.insert({ GATE_RYY, RYY_GATE });
	m_qgate_type_map.insert({ GATE_RXX, RXX_GATE });
	m_qgate_type_map.insert({ GATE_RZZ, RZZ_GATE });
	m_qgate_type_map.insert({ GATE_RZX, RZX_GATE });

    m_qgate_type_map.insert({ GATE_CNOT, CNOT_GATE });
    m_qgate_type_map.insert({ GATE_CZ, CZ_GATE });
    m_qgate_type_map.insert({ GATE_CPHASE, CPHASE_GATE });

	m_qgate_type_map.insert({ GATE_SWAP, SWAP_GATE });
    m_qgate_type_map.insert({ GATE_ISWAPTheta, ISWAP_THETA_GATE });
    m_qgate_type_map.insert({ GATE_ISWAP, ISWAP_GATE }); 
    m_qgate_type_map.insert({ GATE_SQISWAP, SQISWAP_GATE });
    m_qgate_type_map.insert({ GATE_QDoubleGate, TWO_QUBIT_GATE });

	m_qgate_type_map.insert({ GATE_I, I_GATE });
    m_qgate_type_map.insert({ GATE_ECHO, ECHO_GATE });
    m_qgate_type_map.insert({ GATE_BARRIER, BARRIER_GATE });
	m_qgate_type_map.insert({ GATE_ORACLE, ORACLE_GATE });
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
