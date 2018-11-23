/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

TranformQGateTypeStringAndEnum.h
Author: Wangjing
Created in 2018-10-15

Classes for tranform gate type enum and string

*/
#ifndef TRANSFORM_QGATE_TYPE_STRING_ENUM_H
#define TRANSFORM_QGATE_TYPE_STRING_ENUM_H

#include <iostream>
#include <map>
#include <string>
#include "QuantumCircuit/QGlobalVariable.h"

using std::map;
using std::string;
using std::pair;

/*
Classes for tranform gate type enum and string
*/
class QGateTypeEnumToString
{
public:
    static QGateTypeEnumToString &getInstance();
    ~QGateTypeEnumToString();
    string operator [](int type);
protected:
    QGateTypeEnumToString();
    QGateTypeEnumToString(const QGateTypeEnumToString &);
    QGateTypeEnumToString &operator=(const QGateTypeEnumToString &);
private:
    map<int, string> m_QGate_type_map;
};


class QGateTypeStringToEnum
{
public:
    static QGateTypeStringToEnum &getInstance();
    ~QGateTypeStringToEnum();
    int operator [](string gate_name);
protected:
    QGateTypeStringToEnum();
    QGateTypeStringToEnum(const QGateTypeStringToEnum &);
    QGateTypeStringToEnum &operator=(const QGateTypeEnumToString &);
private:
    map<string, int> m_QGate_type_map;
};






#endif // TRANSFORM_QGATE_TYPE_STRING_ENUM_H
