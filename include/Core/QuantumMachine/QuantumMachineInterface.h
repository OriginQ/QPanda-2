/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

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
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/Module/DataStruct.h"
#include "Core/QuantumNoise/NoiseModelV2.h"

QPANDA_BEGIN

/*
*  @enum  QMachineType
*  @brief   Quantum machine type
*/
enum QMachineType {
    CPU,  /**< CPU quantum machine  */
    GPU, /**< GPU quantum machine  */
    CPU_SINGLE_THREAD, /**< CPU quantum machine with single thread */
    NOISE,  /**< CPU quantum machine with noise */
    QCloud,
    DENSITY_MATRIX,
    QPILOT_OS
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
    virtual prob_vec PMeasure_no_index(QVec qubit_vector) = 0;
    virtual prob_tuple getProbTupleList(QVec, int) = 0;
    virtual prob_vec getProbList(QVec, int) = 0;
    virtual prob_dict  getProbDict(QVec, int) = 0;
    virtual prob_tuple probRunTupleList(QProg &, QVec, int) = 0;
    virtual prob_vec probRunList(QProg &, QVec, int) = 0;
    virtual prob_dict probRunDict(QProg &, QVec, int) = 0;
    virtual std::map<std::string, size_t> quickMeasure(QVec, size_t) = 0;
    virtual prob_tuple PMeasure(QVec qubit_vector, int select_max) = 0;
    virtual QStat getQStat() = 0;
    virtual ~IdealMachineInterface() {}
};

class MultiPrecisionMachineInterface
{
public:
    virtual stat_map getQState() = 0;

    virtual qstate_type pMeasureBinIndex(std::string) = 0;
    virtual qstate_type pMeasureDecIndex(std::string) = 0;
     
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

    /**
    * @brief  allocateQubitThroughPhyAddress
    * @param[in]  size_t address
    * @return     QPanda::Qubit* qubit
    */
    virtual Qubit* allocateQubitThroughPhyAddress(size_t) = 0;

    /**
    * @brief  allocateQubitThroughVirAddress
    * @param[in]  size_t address
    * @return     QPanda::Qubit* qubit
    */
    virtual Qubit* allocateQubitThroughVirAddress(size_t) = 0;

    /**
    * @brief  init
    * @return     void
    */
    virtual void init() = 0;

    /**
    * @brief  getStatus
    * @return     QPanda::QMachineStatus*
    */
    virtual QMachineStatus* getStatus() const = 0;

    /**
    * @brief  directlyRun
    * @param[in]  QProg& quantum program
    * @param[in]  bg_run will run in background, default is not
    * @return     std::map<std::string, bool>
    */
    virtual std::map<std::string, bool> directlyRun(QProg & qProg, const NoiseModel& = NoiseModel()) = 0;

    /**
    * @brief  runWithConfiguration
    * @param[in]  QProg& quantum program
    * @param[in]  std::vector<ClassicalCondition>&
    * @param[in]  rapidjson::Document&
    * @return     std::map<std::string, Eigen::size_t>
    */
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, rapidjson::Document&, const NoiseModel& = NoiseModel()) = 0;

	/**
    * @brief  runWithConfiguration
    * @param[in]  QProg& quantum program
    * @param[in]  std::vector<ClassicalCondition>&
    * @param[in]  int
    * @return     std::map<std::string, Eigen::size_t>
    */
	virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int, const NoiseModel& = NoiseModel()) = 0;
   
    /**
    * @brief  runWithConfiguration
    * @param[in]  QProg& quantum program
    * @param[in]  int shots
    * @return     std::map<std::string, Eigen::size_t>
    */
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, int, const NoiseModel& = NoiseModel()) = 0;

    /**
    * @brief  runWithConfiguration
    * @param[in]  QProg& quantum program
    * @param[in]  std::vector<int>& cbit addrs
    * @param[in]  int
    * @return     std::map<std::string, Eigen::size_t>
    */
    virtual std::map<std::string, size_t> runWithConfiguration(QProg&, std::vector<int>&, int, const NoiseModel& = NoiseModel()) = 0;
    
    /**
    * @brief  getGateTimeMap
    * @return     std::map<GateType, Eigen::size_t>
    */
    virtual std::map<GateType, size_t> getGateTimeMap() const = 0;

    /**
    * @brief  finalize
    * @return     void
    */
    virtual void finalize() = 0;

    /**
    * @brief  getQState
    * @return     QStat
    */
    virtual QStat getQState() const = 0;

    /**
    * @brief  getVirtualQubitAddress
    * @param[in]  Qubit* qubit
    * @return     Eigen::size_t
    */
    virtual size_t getVirtualQubitAddress(Qubit *) const = 0;

    /**
    * @brief  swapQubitPhysicalAddress
    * @param[in]  Qubit* qubit
    * @param[in]  Qubit* qubit
    * @return     bool
    */
    virtual bool swapQubitPhysicalAddress(Qubit *, Qubit*) = 0;

    /*will delete*/
    virtual void setConfig(const Configuration &) = 0; //! To initialize the quantum machine
    virtual Qubit* allocateQubit() = 0; //! Allocate and return a qubit
    virtual QVec allocateQubits(size_t) = 0;//! allocateQubits
    virtual ClassicalCondition allocateCBit() = 0; //! Allocate and run a cbit
    virtual ClassicalCondition allocateCBit(size_t) = 0; //! Allocate and run a cbit
    virtual std::vector<ClassicalCondition> allocateCBits(size_t) = 0; //! Allocate and return a list of cbits
    virtual void Free_Qubit(Qubit*) = 0; //! Free a qubit
    virtual void Free_Qubits(QVec &) = 0; //!Gree a list of qubits
    virtual void Free_CBit(ClassicalCondition &) = 0; //! Gree a cbit
    virtual void Free_CBits(std::vector<ClassicalCondition > &) = 0; //!Gree a list of CBits
    virtual size_t getAllocateQubit() = 0;//! getAllocateQubit
    virtual size_t getAllocateCMem() = 0;//! getAllocateCMem

    /* new interface */
    /**
    * @brief  setConfigure
    * @param[in]  const Configuration& config
    * @return     void
    */
    virtual void setConfigure(const Configuration &) = 0;

    /**
    * @brief  qAlloc
    * @return     QPanda::Qubit* qubit
    */
    virtual Qubit* qAlloc() = 0;

    /**
    * @brief  qAllocMany
    * @param[in]  size_t qubit_count
    * @return     QPanda::QVec
    */
    virtual QVec qAllocMany(size_t qubit_count) = 0;

    /**
    * @brief  cAlloc
    * @return     QPanda::ClassicalCondition
    */
    virtual ClassicalCondition cAlloc() = 0;

    /**
    * @brief  cAlloc
    * @param[in]  size_t
    * @return     QPanda::ClassicalCondition
    */
    virtual ClassicalCondition cAlloc(size_t) = 0;

    /**
    * @brief  cAllocMany
    * @param[in]  size_t count
    * @return     std::vector<QPanda::ClassicalCondition>  cbit_vec
    */
    virtual std::vector<ClassicalCondition> cAllocMany(size_t) = 0;

    /**
    * @brief  qFree
    * @param[in]  Qubit*
    * @return     void
    */
    virtual void qFree(Qubit*) = 0;

    /**
    * @brief  qFreeAll
    * @return     void
    */
    virtual void qFreeAll() = 0;

    /**
    * @brief  qFreeAll
    * @param[in]  QVec&
    * @return     void
    */
    virtual void qFreeAll(QVec &) = 0;

    /**
    * @brief  cFree
    * @param[in]  ClassicalCondition& cbit
    * @return     void
    */
    virtual void cFree(ClassicalCondition &) = 0; 

    /**
    * @brief  cFreeAll
    * @param[in]  std::vector<ClassicalCondition >& cbit_vec
    * @return     void
    */
    virtual void cFreeAll(std::vector<ClassicalCondition > &) = 0;

    /**
    * @brief  cFreeAll
    * @return     void
    */
    virtual void cFreeAll() = 0;

    /**
    * @brief  getAllocateQubitNum
    * @return     Eigen::size_t count
    */
    virtual size_t getAllocateQubitNum() = 0;

    /**
    * @brief  getAllocateCMemNum
    * @return     Eigen::size_t count
    */
    virtual size_t getAllocateCMemNum() = 0;

    virtual void initState(const QStat &state = {}, const QVec &qlist = {}) = 0;
    virtual void initSparseState(const std::map<std::string, qcomplex_t> &state = {}, const QVec &qlist = {}) = 0;
    /**
    * @brief  get allocate qubits
    * @param[out] QVec& 
    * @return size_t  allocate qubits size
    */
    virtual size_t get_allocate_qubits(QVec&) = 0;

    /**
    * @brief  get allocate cbits
    * @param[out] std::vector<ClassicalCondition>& 
    * @return size_t  allocate cbits size
    */
    virtual size_t get_allocate_cbits(std::vector<ClassicalCondition>&) = 0;

    /**
    * @brief  gets the expected value of the Hamiltonian of the circuit
    * @param[in] QProg quantum program 
    * @param[in] QHamiltonian hamiltonian
    * @param[in] QVec qubits 
    * @return double  expectation
    */
	virtual double get_expectation(QProg, const QHamiltonian&, const QVec&) = 0;

    /**
    * @brief  gets the expected value of the Hamiltonian of the circuit
    * @param[in] QProg quantum program 
    * @param[in] QHamiltonian hamiltonian
    * @param[in] QVec qubits 
    * @param[in] int  run shots 
    * @return double  expectation
    */
    virtual double get_expectation(QProg, const QHamiltonian&, const QVec&, int) = 0;

    /**
    * @brief  gets the probably [P1, P2, ... , Pn] of the all Hamiltonian Subitem of the circuit
    * @param[in] QProg quantum program
    * @param[in] QHamiltonian hamiltonian
    * @param[in] QVec qubits
    * @param[in] vector_d measure probably
    * @return void
    */
    virtual void get_expectation_vector(QProg, const QHamiltonian&, const QVec&, vector_d&) = 0;

    /**
    * @brief  gets the probably [P1, P2, ... , Pn] of the all Hamiltonian Subitem of the circuit
    * @param[in] QProg quantum program
    * @param[in] QHamiltonian hamiltonian
    * @param[in] QVec qubits
    * @param[in] vector_d measure probably
    * @param[in] int run shots
    * @return void
    */
    virtual void get_expectation_vector(QProg, const QHamiltonian&, const QVec& qv, vector_d&, int) = 0;

    /**
     * @brief get gate num have been processed while machine running prog async
     * 
     * @return size_t gate count
     */
    virtual size_t get_processed_qgate_num() = 0;

    /**
     * @brief like @direclyRun but run prog async
     * 
     */
    virtual void async_run(QProg & qProg, const NoiseModel& = NoiseModel()) = 0;

    /**
     * @brief check async_run is finished
     * 
     */
    virtual bool is_async_finished() = 0;

    /**
     * @brief get the async_run result
     * 
     * @return std::map<std::string, bool> 
     */
    virtual std::map<std::string, bool> get_async_result() = 0;

    virtual ~QuantumMachine() {} //! Destructor
};
QPANDA_END
#endif
