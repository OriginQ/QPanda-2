/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Traversal.h
Author: doumenghan
Created in 2019-4-16

Classes for get the shortes path of graph

*/
#ifndef _UTILITIES_H
#define _UTILITIES_H

#include"QPandaNamespace.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include <iostream>
#include <map>
QPANDA_BEGIN
std::string dec2bin(unsigned n, size_t size);
double RandomNumberGenerator();
void add_up_a_map(std::map<std::string, size_t> &meas_result, std::string key);

void insertQCircuit(AbstractQGateNode * pGateNode,
    QCircuit & qCircuit,
    QNode * pParentNode);

QPANDA_END

#endif // !1

