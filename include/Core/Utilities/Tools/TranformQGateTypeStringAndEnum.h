/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

TranformQGateTypeStringAndEnum.h
Author: Wangjing
Created in 2018-10-15

Classes for tranform gate type enum and std::string

*/
#ifndef TRANSFORM_QGATE_TYPE_STRING_ENUM_H
#define TRANSFORM_QGATE_TYPE_STRING_ENUM_H

#include <iostream>
#include <map>
#include <string>
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

#define GATE_X "X"
#define GATE_Y "Y"
#define GATE_Z "Z"
#define GATE_X1 "X1"
#define GATE_Y1 "Y1"
#define GATE_Z1 "Z1"
#define GATE_RX "RX"
#define GATE_RY "RY"
#define GATE_RZ "RZ"
#define GATE_H "H"
#define GATE_S "S"
#define GATE_T "T"
#define GATE_I "I"
#define GATE_RPHI "RPhi"
#define GATE_U1 "U1"
#define GATE_U2 "U2"
#define GATE_U3 "U3"
#define GATE_U4 "U4"
#define GATE_ECHO "ECHO"

#define GATE_CNOT "CNOT"
#define GATE_CZ "CZ"
#define GATE_CU "CU"
#define GATE_CPHASE "CPHASE"
#define GATE_SWAP "SWAP"
#define GATE_ISWAPTheta "ISWAPTheta"
#define GATE_ISWAP "ISWAP"
#define GATE_SQISWAP "SQISWAP"
#define GATE_BARRIER "BARRIER"
#define GATE_QDoubleGate "QDoubleGate"

/**
* @brief Classes for tranform gate type  and gate name
* @ingroup Utilities
*/
class TransformQGateType
{
public:
    static TransformQGateType &getInstance();
    ~TransformQGateType() {};
    std::string operator [](GateType);
    GateType operator [](std::string gate_name);
private:
    std::map<std::string, GateType> m_qgate_type_map;
    TransformQGateType &operator=(const TransformQGateType &);
    TransformQGateType();
    TransformQGateType(const TransformQGateType &);
};





QPANDA_END
#endif // TRANSFORM_QGATE_TYPE_STRING_ENUM_H
