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

#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/QuantumCircuit/ClassicalProgam.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/OriginCollection.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Transform/QGateCounter.h"
#include "Core/Utilities/Transform/QProgToQRunes.h"
#include "Core/Utilities/Transform/QProgToQASM.h"
#include "Core/Utilities/Transform/QProgToQuil.h"
#include "Core/Utilities/Transform/QRunesToQProg.h"
#include "Core/Utilities/Transform/QProgStored.h"
#include "Core/Utilities/Transform/QProgDataParse.h"
#include "Core/Utilities/QPandaException.h"


QPANDA_BEGIN

std::string qProgToQRunes(QProg &pQProg);
std::string qProgToQASM(QProg &pQProg);
void qRunesToQProg(std::string sFilePath,QProg& newQProg);

// to init the environment. Use this at the beginning
bool init(QMachineType type = CPU);

// to finalize the environment. Use this at the end
void finalize();

// Allocate a qubit
Qubit* qAlloc();

// Allocate a qubit
Qubit* qAlloc(size_t stQubitAddr);

std::map<std::string, bool> directlyRun(QProg & qProg);

// Allocate many qubits
QVec qAllocMany(size_t stQubitNumber);

// Allocate a cbit
ClassicalCondition cAlloc();

// Allocate a cbit
ClassicalCondition cAlloc(size_t stCBitaddr);

// Allocate many cbits
std::vector<ClassicalCondition> cAllocMany(size_t stCBitNumber);

// Free a cbit
void cFree(ClassicalCondition &);

// Free a list of CBits
void cFreeAll(std::vector<ClassicalCondition> vCBit);

// get the status(ptr) of the quantum machine
QMachineStatus* getstat();


 size_t getAllocateQubitNum();

 size_t getAllocateCMem();

std::vector<std::pair<size_t, double>> getProbTupleList(QVec &,int selectMax=-1);
std::vector<double> getProbList(QVec &, int selectMax = -1);
std::map<std::string, double>  getProbDict(QVec &, int selectMax = -1);
std::vector<std::pair<size_t, double>> probRunTupleList(QProg &,QVec &, int selectMax = -1);
std::vector<double> probRunList(QProg &,QVec&, int selectMax = -1);
std::map<std::string, double>  probRunDict(QProg &,QVec &, int selectMax = -1);
std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int);
std::map<std::string, size_t> quickMeasure(QVec &, int);

std::vector<std::pair<size_t, double>> PMeasure(QVec& qubit_vector, int select_max);
std::vector<double> PMeasure_no_index(QVec& qubit_vector);
std::vector<double> accumulateProbability(std::vector<double> &prob_list);
std::map<std::string, size_t> quick_measure(QVec& qubit_vector, int shots,
    std::vector<double>& accumulate_probabilites);

size_t getQProgClockCycle(QProg &prog);
QStat getQState();
QuantumMachine *initQuantumMachine(QMachineType type=CPU);
void destroyQuantumMachine(QuantumMachine * qvm);
QPanda::QProg MeasureAll(QVec&, std::vector<ClassicalCondition> &);
QPANDA_END
#endif // !_QPANDA_H
