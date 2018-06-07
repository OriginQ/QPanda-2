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

#ifndef _QPANDA_H
#define _QPANDA_H
#include "QuantumCircuit/QGate.h"
#include "QuantumCircuit/QException.h"
#include "QuantumCircuit/QProgram.h"
#include "QuantumMachine/OriginQuantumMachine.h"

extern QNodeVector _G_QNodeVector;

// Create an empty QProg Container
extern QProg & CreateEmptyQProg();

// Create a While Program
extern QWhileNode &CreateWhileProg(
    ClassicalCondition *,
    QNode * trueNode);

// Create an If Program
extern QIfNode &CreateIfProg(
    ClassicalCondition *,
    QNode *trueNode);

// Create an If Program
extern QIfNode &CreateIfProg(
    ClassicalCondition *,
    QNode *trueNode,
    QNode *falseNode);

// Create an empty QCircuit Container
extern OriginQCircuit & CreateEmptyCircuit();

// Create a Measure operation
extern QMeasureNode& Measure(Qubit * targetQuBit, CBit * targetCbit);

// Create a X gate
extern OriginQGateNode & RX(Qubit* qbit);

// Create a X rotation
extern OriginQGateNode & RX(Qubit*, double angle);

// Create a Y gate
extern OriginQGateNode & RY(Qubit* qbit);

// Create a Y rotation
extern OriginQGateNode & RY(Qubit*, double angle);

// Create a Z gate
extern OriginQGateNode & RZ(Qubit* qbit);

// Create a Z rotation
extern OriginQGateNode & RZ(Qubit*, double angle);

// Create S gate
extern OriginQGateNode & S(Qubit* qbit);

// Create Hadamard Gate
extern OriginQGateNode & H(Qubit* qbit);

// Create an instance of CNOT gate
extern OriginQGateNode & CNOT(Qubit* targetQBit, Qubit* controlQBit);

// Create an instance of CZ gate
extern OriginQGateNode & CZ(Qubit* targetQBit, Qubit* controlQBit);

// Create an arbitrary single unitary gate
extern OriginQGateNode & QSingle(double alpha, double beta, double gamma, double delta, Qubit *);

// Create a control-U gate
extern OriginQGateNode & QDouble(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);

/* Init the environment. Use this at the beginning
* 
*/
void init();

// to finalize the environment. Use this at the end
void finalize();

// Allocate a qubit
Qubit* qAlloc();

// Free a qubit
void qFree(Qubit* q);

// Allocate a cbit
CBit* cAlloc();

// Free a cbit
void cFree(CBit* c);

// load a program
void load(QProg& q);

// append a program after the loaded program
void append(QProg& q);

// get the status(ptr) of the quantum machine
QMachineStatus* getstat();

// get the result(ptr)
QResult* getResult();

// directly get the result map
map<string, bool> getResultMap();

bool getCBitValue(CBit* cbit);

// bind a cbit to a classicalcondition variable
ClassicalCondition bind_a_cbit(CBit* c);

// run the loaded program
void run();

#endif // !_QPANDA_H

