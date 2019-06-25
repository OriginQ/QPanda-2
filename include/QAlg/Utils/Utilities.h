/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef UTILITIES_H
#define UTILITIES_H
#include "Core/QPanda.h"
QPANDA_BEGIN
QProg Reset_Qubit(Qubit* q, bool setVal);

QProg Reset_Qubit_Circuit(Qubit *q, ClassicalCondition& cbit, bool setVal);

QProg Reset_All(std::vector<Qubit*> qubit_vector, bool setVal);

/*
CNOT all qubits (except last) with the last qubit

param:
    qubit_vec: qubit vector
return:
    QCircuit
*/
QCircuit parity_check_circuit(std::vector<Qubit*> qubit_vec);

/*
Apply Quantum Gate on a series of Qubit

param:
	qubits: qubit vector
return:
	QCircuit
*/
inline QCircuit apply_QGate(std::vector<Qubit*> qubits, std::function<QGate(Qubit*)> gate) {
	QCircuit c;
	for (auto qubit : qubits) {
		c << gate(qubit);
	}
	return c;
}

template<typename InputType, typename OutputType>
class Oracle : public std::function<QCircuit(InputType, OutputType)> { 
public:
	Oracle(std::function<QCircuit(InputType, OutputType)> func):
		std::function <QCircuit(InputType, OutputType)>(func) {}
};

inline QGate Toffoli(Qubit* qc1, Qubit* qc2, Qubit* target) {
	auto gate = X(target);
	return gate.control({ qc1,qc2 });
}

QPANDA_END
#endif