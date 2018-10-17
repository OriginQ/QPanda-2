/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

TranformQGateTypeStringAndEnum.cpp
Author: Wangjing
Created in 2018-10-15

Classes for tranform gate type enum and string

*/
#include "TranformQGateTypeStringAndEnum.h"
#include "QPandaException.h"


QGateTypeEnumToString::QGateTypeEnumToString()
{

    m_QGate_type_map.insert({PAULI_X_GATE, "X"});
    m_QGate_type_map.insert({PAULI_Y_GATE, "Y"});
    m_QGate_type_map.insert({PAULI_Z_GATE, "Z"});

    m_QGate_type_map.insert({X_HALF_PI, "X1"});
    m_QGate_type_map.insert({Y_HALF_PI, "Y1"});
    m_QGate_type_map.insert({Z_HALF_PI, "Z1"});

    m_QGate_type_map.insert({HADAMARD_GATE, "H"});
    m_QGate_type_map.insert({T_GATE, "T"});
    m_QGate_type_map.insert({S_GATE, "S"});

    m_QGate_type_map.insert({RX_GATE, "RX"});
    m_QGate_type_map.insert({RY_GATE, "RY"});
    m_QGate_type_map.insert({RZ_GATE, "RZ"});

    m_QGate_type_map.insert({U1_GATE, "U1"});
    m_QGate_type_map.insert({U2_GATE, "U2"});
    m_QGate_type_map.insert({U3_GATE, "U3"});
    m_QGate_type_map.insert({U4_GATE, "U4"});

    m_QGate_type_map.insert({CU_GATE, "CU"});
    m_QGate_type_map.insert({CNOT_GATE, "CNOT"});
    m_QGate_type_map.insert({CZ_GATE, "CZ"});
    m_QGate_type_map.insert({CPHASE_GATE, "CPHASE"});

    m_QGate_type_map.insert({ISWAP_THETA_GATE, "ISWAP_THETA"});
    m_QGate_type_map.insert({ISWAP_GATE, "ISWAP"});
    m_QGate_type_map.insert({SQISWAP_GATE, "SQISWAP"});
    m_QGate_type_map.insert({TWO_QUBIT_GATE, "TWO_QUBIT"});
}

QGateTypeEnumToString &QGateTypeEnumToString::getInstance()
{
    static QGateTypeEnumToString gate_map;
    return gate_map;
}

QGateTypeEnumToString::~QGateTypeEnumToString()
{

}

string QGateTypeEnumToString::operator [](int type)
{
    auto iter = m_QGate_type_map.find(type);
    if (m_QGate_type_map.end() == iter)
    {
        throw param_error_exception("gate type is not support", false);
    }

    return iter->second;
}


QGateTypeStringToEnum::QGateTypeStringToEnum()
{
    m_QGate_type_map.insert({"X", PAULI_X_GATE});
    m_QGate_type_map.insert({"Y", PAULI_Y_GATE});
    m_QGate_type_map.insert({"Z", PAULI_Z_GATE});

    m_QGate_type_map.insert({"X1", X_HALF_PI});
    m_QGate_type_map.insert({"Y1", Y_HALF_PI});
    m_QGate_type_map.insert({"Z1", Z_HALF_PI});

    m_QGate_type_map.insert({"H", HADAMARD_GATE});
    m_QGate_type_map.insert({"T", T_GATE});
    m_QGate_type_map.insert({"S", S_GATE});

    m_QGate_type_map.insert({"RX", RX_GATE});
    m_QGate_type_map.insert({"RY", RY_GATE});
    m_QGate_type_map.insert({"RZ", RZ_GATE});

    m_QGate_type_map.insert({"U1", U1_GATE});
    m_QGate_type_map.insert({"U2", U2_GATE});
    m_QGate_type_map.insert({"U3", U3_GATE});
    m_QGate_type_map.insert({"U4", U4_GATE});

    m_QGate_type_map.insert({"CU", CU_GATE});
    m_QGate_type_map.insert({"CNOT", CNOT_GATE});
    m_QGate_type_map.insert({"CZ", CZ_GATE});
    m_QGate_type_map.insert({"CPHASE", CPHASE_GATE});

    m_QGate_type_map.insert({"ISWAP_THETA", ISWAP_THETA_GATE});
    m_QGate_type_map.insert({"ISWAP", ISWAP_GATE});
    m_QGate_type_map.insert({"SQISWAP", SQISWAP_GATE});
    m_QGate_type_map.insert({"TWO_QUBIT", TWO_QUBIT_GATE});
}

QGateTypeStringToEnum &QGateTypeStringToEnum::getInstance()
{
    static QGateTypeStringToEnum gate_map;
    return gate_map;
}

QGateTypeStringToEnum::~QGateTypeStringToEnum()
{

}

int QGateTypeStringToEnum::operator [](string gate_name)
{
    auto iter = m_QGate_type_map.find(gate_name);
    if (m_QGate_type_map.end() == iter)
    {
        throw param_error_exception("gate name is not support", false);
    }

    return iter->second;
}


