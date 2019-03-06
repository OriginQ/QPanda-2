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

#ifndef QUANTUM_MACHINE_INTERFACE_H
#define QUANTUM_MACHINE_INTERFACE_H
#include <map>
#include <vector>
#include "Core/QuantumMachine/QResultFactory.h"
#include "Core/QuantumMachine/QVec.h"
#include "VirtualQuantumProcessor/QPUImpl.h"                             
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/OriginCollection.h"
QPANDA_BEGIN

enum QMachineType {
    CPU,
    GPU,
    CPU_SINGLE_THREAD,
    NOISE
};

class QubitAddr 
{
    // Qubit Address Class, the address is expressed as
    // a 2-dimensional coordinate
    // I don't think there is anything needed comment
private:
    int x;
    int y;
public:
    virtual int getX() const;
    virtual int getY() const;
    virtual void setX(int x);
    virtual void setY(int y);
    QubitAddr(int ix, int iy);
    QubitAddr() {};
    virtual ~QubitAddr() {}
};

class QMachineStatus
{
    // this class contains the status of the quantum machine
public:
    virtual int getStatusCode() const = 0;
    virtual void setStatusCode(int) = 0;
    virtual ~QMachineStatus() {}
};

class IdealMachineInterface
{
public:
    virtual std::vector<double> PMeasure_no_index(QVec qubit_vector) = 0;
    virtual std::vector<std::pair<size_t, double>> getProbTupleList(QVec, int) = 0;
    virtual std::vector<double> getProbList(QVec, int) = 0;
    virtual std::map<std::string, double>  getProbDict(QVec, int) = 0;
    virtual std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec, int) = 0;
    virtual std::vector<double> probRunList(QProg &, QVec, int) = 0;
    virtual std::map<std::string, double> probRunDict(QProg &, QVec, int) = 0;
    virtual std::map<std::string, size_t> quickMeasure(QVec, size_t) = 0;
    virtual std::vector<std::pair<size_t, double>> PMeasure(QVec qubit_vector, int select_max) = 0;
    virtual QStat getQStat() = 0;
    virtual ~IdealMachineInterface() {}
};
 
class QuantumMachine
{
public:
    virtual void init() = 0; // to initialize the quantum machine
    virtual Qubit* Allocate_Qubit() = 0; // allocate and return a qubit
    virtual ClassicalCondition Allocate_CBit() = 0; // allocate and run a cbit
    virtual std::vector<ClassicalCondition> Allocate_CBits(size_t) = 0; // allocate and return a list of cbits
    virtual Qubit* Allocate_Qubit(size_t) = 0; // allocate and return a qubit
    virtual QVec Allocate_Qubits(size_t qubit_count) = 0;
    virtual ClassicalCondition Allocate_CBit(size_t) = 0; // allocate and run a cbit
    virtual void Free_Qubit(Qubit*) = 0; // free a qubit
    virtual void Free_Qubits(QVec &) = 0; //free a list of qubits
    virtual void Free_CBit(ClassicalCondition &) = 0; // free a cbit
    virtual void Free_CBits(std::vector<ClassicalCondition > &) = 0; //free a list of CBits
    virtual QMachineStatus* getStatus() const = 0; // get the status of the quantum machine
    virtual std::map<std::string, bool> directlyRun(QProg & qProg) = 0;
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, rapidjson::Document&) = 0;
    virtual size_t getAllocateQubit() = 0;
    virtual size_t getAllocateCMem() = 0;
    virtual std::map<int, size_t> getGateTimeMap() const = 0;
    virtual void finalize() = 0; // finish the program

    virtual ~QuantumMachine() {} // destructor
};
QPANDA_END
#endif
