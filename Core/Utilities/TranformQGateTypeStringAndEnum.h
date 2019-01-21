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
#include "QuantumCircuit/QGlobalVariable.h"
#include "QPandaNamespace.h"

QPANDA_BEGIN

/*
Classes for tranform gate type enum and std::string
*/
class QGateTypeEnumToString
{
public:
    static QGateTypeEnumToString &getInstance();
    ~QGateTypeEnumToString();
    std::string operator [](int type);
protected:
    QGateTypeEnumToString();
    QGateTypeEnumToString(const QGateTypeEnumToString &);
    QGateTypeEnumToString &operator=(const QGateTypeEnumToString &);
private:
    std::map<int, std::string> m_QGate_type_map;
};


class QGateTypeStringToEnum
{
public:
    static QGateTypeStringToEnum &getInstance();
    ~QGateTypeStringToEnum();
    int operator [](std::string gate_name);
protected:
    QGateTypeStringToEnum();
    QGateTypeStringToEnum(const QGateTypeStringToEnum &);
    QGateTypeStringToEnum &operator=(const QGateTypeEnumToString &);
private:
    std::map<std::string, int> m_QGate_type_map;
};





QPANDA_END
#endif // TRANSFORM_QGATE_TYPE_STRING_ENUM_H
