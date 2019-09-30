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

#include"Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include <iostream>
#include <map>
QPANDA_BEGIN
std::string dec2bin(unsigned n, size_t size);
double RandomNumberGenerator();
void add_up_a_map(std::map<std::string, size_t> &meas_result, std::string key);

void insertQCircuit(AbstractQGateNode * pGateNode,
    QCircuit & qCircuit,
    QNode * pParentNode);

QProg Reset_Qubit_Circuit(Qubit *q, ClassicalCondition& cbit, bool setVal);

QProg Reset_Qubit(Qubit* q, bool setVal, QuantumMachine * qvm);

QProg Reset_All(std::vector<Qubit*> qubit_vector, bool setVal,QuantumMachine * qvm);

/*
CNOT all qubits (except last) with the last qubit

param:
qubit_vec: qubit vector
return:
QCircuit
*/
QCircuit parityCheckCircuit(std::vector<Qubit*> qubit_vec);

/*
Apply Quantum Gate on a series of Qubit

param:
qubits: qubit vector
return:
QCircuit
*/
inline QCircuit apply_QGate(QVec qubits, std::function<QGate(Qubit*)> gate) {
	QCircuit c;
	for (auto qubit : qubits) {
		c << gate(qubit);
	}
	return c;
}

inline QCircuit applyQGate(QVec qubits, std::function<QGate(Qubit*)> gate) {
	QCircuit c;
	for (auto qubit : qubits) {
		c << gate(qubit);
	}
	return c;
}

template<typename InputType, typename OutputType>

using Oracle = std::function<QCircuit(InputType, OutputType)>;

inline QGate Toffoli(Qubit* qc1, Qubit* qc2, Qubit* target) {
	auto gate = X(target);
	gate.setControl({ qc1,qc2 });
	return gate;
}


QPANDA_END

#endif // !1

