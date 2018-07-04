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
#include "QuantumCircuit/QProgram.h"
#include "QuantumMachine/OriginQuantumMachine.h"
#include "TraversalAlgorithm/StatisticsQGateCountAlgorithm.h"

extern size_t countQGateUnderQCircuit(AbstractQuantumCircuit *);

extern HadamardQCircuit CreateHadamardQCircuit(vector<Qubit *> & pQubitVector);

extern QNodeMap _G_QNodeMap;

// Create an empty QProg Container
extern QProg  CreateEmptyQProg();

// Create a While Program
extern QWhileProg CreateWhileProg(
    ClassicalCondition &,
    QNode * trueNode);

// Create an If Program
extern QIfProg CreateIfProg(
    ClassicalCondition &,
    QNode *trueNode);

// Create an If Program
extern QIfProg CreateIfProg(
    ClassicalCondition &,
    QNode *trueNode,
    QNode *falseNode);

// Create an empty QCircuit Container
extern QCircuit  CreateEmptyCircuit();

// Create a Measure operation
extern QMeasure Measure(Qubit * targetQuBit, CBit * targetCbit);

// Create a X gate
extern QGate  X(Qubit* qbit);

// Create a X rotation
extern QGate RX(Qubit*, double angle);

// Create a Y gate
extern QGate  Y(Qubit* qbit);

// Create a Y rotation
extern QGate RY(Qubit*, double angle);

// Create a Z gate
extern QGate Z(Qubit* qbit);

// Create a Z rotation
extern QGate RZ(Qubit*, double angle);

// Create S_GATE gate
extern QGate S(Qubit* qbit);

// Create Hadamard Gate
extern QGate H(Qubit* qbit);

// Create an instance of CNOT gate
extern QGate CNOT(Qubit* targetQBit, Qubit* controlQBit);

// Create an instance of CZ gate
extern QGate CZ(Qubit* targetQBit, Qubit* controlQBit);

// Create an arbitrary single unitary gate
extern QGate QSingle(double alpha, double beta, double gamma, double delta, Qubit *);

// Create a control-U gate
extern QGate CU(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);

extern QGate  iSWAP(Qubit * targitBit, Qubit * controlBit);
extern QGate  iSWAP(Qubit * targitBit, Qubit * controlBit, double theta);

extern QGate  SqiSWAP(Qubit * targitBit, Qubit * controlBit);
// to init the environment. Use this at the beginning
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
