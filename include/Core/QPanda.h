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
/*! \file QPanda.h */
#ifndef _QPANDA_H
#define _QPANDA_H
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
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
#include "Core/Utilities/XMLConfigParam.h"
#include "Core/Utilities/QPandaException.h"
#include "Core/Utilities/Transform/QGateCompare.h"
#include "Core/Utilities/MetadataValidity.h"
#include "Core/Utilities/Transform/QProgClockCycle.h"

QPANDA_BEGIN
/**
* @brief  Init the environment
* @ingroup QuantumMachine
* @param[in]  QMachineType   Quantum machine type
* @retval 1   Init success
* @retval 0   Init failed
* @see    QMachineType
* @note   Use this at the beginning
*/
bool init(QMachineType type = CPU);

/**
* @brief  Finalize the environment
* @ingroup QuantumMachine
* @return    void
* @note   Use this at the end
*/
void finalize();

/**    
* @brief  Allocate a qubit
* @ingroup Core
* @return    void
* @note   Brfore use this,call init()
*/
Qubit* qAlloc();

/**
* @brief  Allocate a qubit
* @ingroup Core
* @param[in]  size_t set qubit address
* @return    void
* @note   Brfore use this,call init()
*/
Qubit* qAlloc(size_t stQubitAddr);

/**
* @brief  Directly run a quantum program
* @ingroup QuantumMachine
* @param[in]  QProg& Quantum program
* @return     std::map<std::string, bool>   result
*/
std::map<std::string, bool> directlyRun(QProg & qProg);

/**
* @brief  Allocate many qubits
* @ingroup Core
* @param[in]  size_t set qubit number 
* @note    Brfore use this,call init()
*/
QVec qAllocMany(size_t stQubitNumber);

/**
* @brief  Allocate a cbit
* @ingroup Core
* @return    ClassicalCondition  cbit
* @note   Brfore use this,call init()
*/
ClassicalCondition cAlloc();

/**
* @brief  Allocate a cbit
* @ingroup Core
* @param[in]  size_t set cbit address
* @return    ClassicalCondition  Cbit
* @note   Brfore use this,call init()
*/
ClassicalCondition cAlloc(size_t stCBitaddr);

/**
* @brief  Allocate many cbits
* @ingroup Core
* @param[in]  size_t set cbit number
* @note    Brfore use this,call init()
*/
std::vector<ClassicalCondition> cAllocMany(size_t stCBitNumber);



/**
* @brief  Free a cbit
* @ingroup Core
* @param[in]  ClassicalCondition&  a reference to a cbit
* @return     void  
*/
void cFree(ClassicalCondition &);

/**
* @brief  Free a list of cbits
* @ingroup Core
* @param[in]  std::vector<ClassicalCondition>    a list of cbits
* @return     void  
*/
void cFreeAll(std::vector<ClassicalCondition> vCBit);



/**
* @brief  Get the status(ptr) of the Quantum machine
* @ingroup QuantumMachine
* @return     QPanda::QMachineStatus*  Quantum machine  status(ptr)
*/
QMachineStatus* getstat();


 /**
 * @brief  Get all allocate qubit num
 * @ingroup Core
 * @return     size_t  Qubit num
 */
 size_t getAllocateQubitNum();

 /**
 * @brief  Get allocate cbit nmu
 * @ingroup Core
 * @return    size_t Cbit num
 */
 size_t getAllocateCMem();

/**
* @brief  Get pmeasure result as tuple list
* @ingroup QuantumMachine
* @param[in]  QVec&  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::vector<std::pair<size_t, double>>  result
* @note   selectMax  can not exceed (1ull << the size of qubits vector)
*/
std::vector<std::pair<size_t, double>> getProbTupleList(QVec &,int selectMax=-1);

/**
* @brief  Get pmeasure result as list
* @ingroup QuantumMachine
* @param[in]  QVec&  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::vector<double>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
std::vector<double> getProbList(QVec &, int selectMax = -1);

/**
* @brief  Get pmeasure result as dict
* @ingroup QuantumMachine
* @param[in]  QVec&  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::map<std::string, double>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
std::map<std::string, double>  getProbDict(QVec &, int selectMax = -1);

/**
* @brief  Get pmeasure result as dict
* @ingroup QuantumMachine
* @param[in]  QProg&  Quantum program
* @param[in]  QVec&  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::vector<std::pair<size_t, double>>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
std::vector<std::pair<size_t, double>> probRunTupleList(QProg &,QVec &, int selectMax = -1);

/**
* @brief  Get pmeasure result as list
* @ingroup QuantumMachine
* @param[in]  QProg&  Quantum program
* @param[in]  QVec&  Pmeasure qubits vector
* @param[in]  int selectmax:the returned value num
* @return     std::vector<double>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
std::vector<double> probRunList(QProg &,QVec&, int selectMax = -1);


/**
* @brief  Get pmeasure result as dict
* @ingroup QuantumMachine
* @param[in]  QProg&  Quantum program
* @param[in]  QVec&  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::map<std::string, double>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
std::map<std::string, double>  probRunDict(QProg &,QVec &, int selectMax = -1);

/**
* @brief  Measure run with configuration
* @ingroup QuantumMachine
* @param[in]  QProg&  Quantum program
* @param[in]  std::vector<ClassicalCondition>&  cbits vector
* @param[in]  int Shots:the repeat num  of measure operate
* @return     std::map<std::string, size_t>   result
*/
std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int);

/**
* @brief  Quick measure operate
* @ingroup QuantumMachine
* @param[in]  QVec&  qubits vector
* @param[in]  int Shots:the repeat num  of measure operate
* @return     std::map<std::string,size_t>  result
*/
std::map<std::string, size_t> quickMeasure(QVec &, int);



/**
* @brief  PMeasure
* @ingroup QuantumMachine
* @param[in]  QVec& qubit  vector
* @param[in]  int  Selectmax:the returned value num
* @return     std::vector<std::pair<size_t, double>>   result
*/
std::vector<std::pair<size_t, double>> PMeasure(QVec& qubit_vector, int select_max);

/**
* @brief  PMeasure only return result  with no index
* @ingroup QuantumMachine
* @param[in]  QVec& qubit vector
* @return     std::vector<double>  result
*/
std::vector<double> PMeasure_no_index(QVec& qubit_vector);


/**
* @brief  AccumulateProbability
* @ingroup QuantumMachine
* @param[in]  std::vector<double> & prob_list  Abstract Quantum program pointer
* @return     std::vector<double>  
*/
std::vector<double> accumulateProbability(std::vector<double> &prob_list);


/**
* @brief  Quick measure
* @ingroup QuantumMachine
* @param[in]  QVec&  qubits vector
* @param[in]  int Shots:the repeat num  of measure operate
* @param[in]  std::vector<double>& accumulate  Probabilites 
* @return     std::map<std::string,size_t>  Results
*/
std::map<std::string, size_t> quick_measure(QVec& qubit_vector, int shots,
    std::vector<double>& accumulate_probabilites);

/**
* @brief  Get quantum state
* @ingroup QuantumMachine
* @return     qstat  Quantum state  vector
*/
QStat getQState();
/**
* @brief  Init a Quantum machine
* @ingroup QuantumMachine
* @param[in]  QMachineType
* @return     QPanda::QuantumMachine*  Quantum machine pointer
* @see  QMachineType
* @note  default  Quantum machine type :cpu
*/
QuantumMachine *initQuantumMachine(QMachineType type=CPU);

/**
* @brief  Destroy Quantum machine
* @ingroup QuantumMachine
* @param[in]  QuantumMachine* Quantum machine pointer
* @return     void  
*/
void destroyQuantumMachine(QuantumMachine * qvm);

/**
* @brief  Measure All  ClassicalCondition
* @ingroup QuantumMachine
* @param[in]  QVec&  qubits vector
* @param[in]  std::vector<ClassicalCondition>&  Cbits vector
* @return    QPanda::QProg   Quantum program
*/
QPanda::QProg MeasureAll(QVec, std::vector<ClassicalCondition>);

QPANDA_END
#endif // !_QPANDA_H
