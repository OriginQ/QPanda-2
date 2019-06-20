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
/*! \file QuantumMachineInterface.h */
#ifndef QUANTUM_MACHINE_INTERFACE_H
#define QUANTUM_MACHINE_INTERFACE_H
#include <map>
#include <complex>
#include <vector>
#include "Core/QuantumMachine/QResultFactory.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"                             
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/OriginCollection.h"
QPANDA_BEGIN

/*
*  @enum  QMachineType
*  @brief   Quantum machine type
*/
enum QMachineType {
    CPU,  /**< Cpu quantum machine  */
    GPU, /**< Gpu quantum machine  */
    CPU_SINGLE_THREAD, /**< Cpu quantum machine with single thread */
    NOISE  /**< Cpu quantum machine with  noise */
};
/**
* @class Configuration
* @brief  Quantum qubit and cbit Configuration
* @note  Default number is 25
* @see QVM
*/
struct Configuration
{
    size_t maxQubit = 25;/**< Config max qubit num   */
    size_t maxCMem = 25;/**< Config max cbit num   */
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

class MultiPrecisionMachineInterface
{
public:
    virtual stat_map getQStat() = 0;

    virtual qstate_type PMeasure_bin_index(std::string) = 0;
    virtual qstate_type PMeasure_dec_index(std::string) = 0;
     
    virtual prob_map PMeasure(std::string) = 0;
    virtual prob_map PMeasure(QVec, std::string) = 0;

    virtual prob_map getProbDict(QVec, std::string) = 0;
    virtual prob_map probRunDict(QProg &, QVec, std::string) = 0;

    virtual ~MultiPrecisionMachineInterface() {}
};

/**
* @class QuantumMachine
* @brief Abstract quantum machine base classes
* @ingroup QuantumMachine
*/
class QuantumMachine
{
public:
    virtual void init() = 0; //! To initialize the quantum machine
    virtual void setConfig(const Configuration &) = 0; //! To initialize the quantum machine
    virtual Qubit* allocateQubit() = 0; //! Allocate and return a qubit
    virtual ClassicalCondition allocateCBit() = 0; //! Allocate and run a cbit
    virtual std::vector<ClassicalCondition> allocateCBits(size_t) = 0; //! Allocate and return a list of cbits
    virtual Qubit* allocateQubitThroughPhyAddress(size_t) = 0; //! Allocate and return a qubit
    virtual Qubit* allocateQubitThroughVirAddress(size_t) = 0; //! Allocate and return a qubit
    virtual QVec allocateQubits(size_t qubit_count) = 0;//! allocateQubits
    virtual ClassicalCondition allocateCBit(size_t) = 0; //! Allocate and run a cbit
    virtual void Free_Qubit(Qubit*) = 0; //! Free a qubit
    virtual void Free_Qubits(QVec &) = 0; //!Gree a list of qubits
    virtual void Free_CBit(ClassicalCondition &) = 0; //! Gree a cbit
    virtual void Free_CBits(std::vector<ClassicalCondition > &) = 0; //!Gree a list of CBits
    virtual QMachineStatus* getStatus() const = 0; //! Get the status of the quantum machine
    virtual std::map<std::string, bool> directlyRun(QProg & qProg) = 0;//! directlyRun
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, rapidjson::Document&) = 0;//! Run with configuration
    virtual size_t getAllocateQubit() = 0;//! getAllocateQubit
    virtual size_t getAllocateCMem() = 0;//! getAllocateCMem
    virtual std::map<GateType, size_t> getGateTimeMap() const = 0;//! Get gate time map
    virtual void finalize() = 0; //! Finish the program
    virtual QStat getQState() const =0;//! Get quantum state
    virtual size_t getVirtualQubitAddress(Qubit *) const = 0;//! Get virtualqubit  address
    virtual bool swapQubitPhysicalAddress(Qubit *, Qubit*) = 0;//! Swap qubit physical address
    virtual ~QuantumMachine() {} //! Destructor
};
QPANDA_END
#endif
