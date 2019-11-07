/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
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

/*
Classes for tranform gate type enum and std::string
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
