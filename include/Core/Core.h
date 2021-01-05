/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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
/*! \file Core.h */
#ifndef _CORE_H
#define _CORE_H
#include "Core/Module/Module.h"

#include "Core/QuantumCircuit/ClassicalConditionInterface.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QReset.h"

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h" 
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/QuantumMachine/QVec.h"	

#include "Core/Utilities/Compiler/QProgDataParse.h"
#include "Core/Utilities/Compiler/QProgStored.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"
#include "Core/Utilities/Compiler/QProgToQASM.h"
#include "Core/Utilities/Compiler/QProgToQuil.h"
#include "Core/Utilities/Compiler/QRunesToQProg.h"
#include "Core/Utilities/Compiler/QuantumChipAdapter.h"

#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgInfo/QGateCompare.h"
#include "Core/Utilities/QProgInfo/QGateCounter.h"
#include "Core/Utilities/QProgInfo/QProgClockCycle.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"

#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include "Core/Utilities/QProgTransform/QProgToQCircuit.h"
#include "Core/Utilities/QProgTransform/QProgToQGate.h"
#include "Core/Utilities/QProgTransform/QProgToQMeasure.h"
#include "Core/Utilities/QProgTransform/TopologyMatch.h"
#include "Core/Utilities/QProgTransform/SU4TopologyMatch.h"
#include "Core/Utilities/QProgTransform/QCodarMatch.h"
#include "Core/Utilities/QProgTransform/BMT/BMT.h"

#include "Core/Utilities/Tools/OriginCollection.h"  
#include "Core/Utilities/Tools/QPandaException.h"  
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/base64.hpp"
#include "Core/Utilities/Tools/QString.h" 
#include "Core/Utilities/Tools/Utils.h"  
#include "Core/Utilities/Tools/JsonConfigParam.h"  
#include "Core/Utilities/Tools/FillQProg.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities/Tools/RandomCircuit.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/Tools/Fidelity.h"
#include "Core/Utilities/Tools/GetQubitTopology.h"

#include "Core/Variational/var.h"
#include "Core/Variational/Optimizer.h"  
#include "Core/Variational/expression.h"
#include "Core/Variational/utils.h"
#include "Core/Variational/Optimizer.h"

#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"  
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSQVM.h"

QPANDA_BEGIN
/**
* @brief  Init the environment
* @ingroup Core
* @param[in]  QMachineType   Quantum machine type
* @return bool   
* @see    QMachineType
* @note   Use this at the beginning
*/
bool init(QMachineType type = CPU);

/**
* @brief  Finalize the environment
* @ingroup Core
* @return    void
* @note   Use this at the end
*/
void finalize();

/**    
* @brief  Allocate a qubit
* @ingroup Core
* @return    void
* @note   Call init() before you use this
*/
Qubit* qAlloc();

/**
* @brief  Allocate a qubit
* @ingroup Core
* @param[in]  size_t set qubit address
* @return    void
* @note   Call init() before you use this
*/
Qubit* qAlloc(size_t stQubitAddr);

/**
* @brief  Directly run a quantum program
* @ingroup Core
* @param[in]  QProg& Quantum program
* @return     std::map<std::string, bool>   result
*/
std::map<std::string, bool> directlyRun(QProg & qProg);

/**
* @brief  Allocate many qubits
* @ingroup Core
* @param[in]  size_t set qubit number 
* @note    Call init() before you use this
*/
QVec qAllocMany(size_t stQubitNumber);

/**
* @brief  Allocate a cbit
* @ingroup Core
* @return    ClassicalCondition  cbit
* @note   Call init() before you use this
*/
ClassicalCondition cAlloc();

/**
* @brief  Allocate a cbit
* @ingroup Core
* @param[in]  size_t set cbit address
* @return    ClassicalCondition  Cbit
* @note   Call init() before you use this
*/
ClassicalCondition cAlloc(size_t stCBitaddr);

/**
* @brief  Allocate many cbits
* @ingroup Core
* @param[in]  size_t set cbit number
* @note    Call init() before you use this
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
* @ingroup Core
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
* @brief  Get pmeasure result as tuple list
* @ingroup Core
* @param[in]  QVec  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::vector<std::pair<size_t, double>>  result
* @note   selectMax  can not exceed (1ull << the size of qubits vector)
*/
 prob_tuple getProbTupleList(QVec ,int selectMax=-1);

/**
* @brief  Get pmeasure result as list
* @ingroup Core
* @param[in]  QVec  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     prob_vec  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
prob_vec getProbList(QVec , int selectMax = -1);

/**
* @brief  Get pmeasure result as dict
* @ingroup Core
* @param[in]  QVec  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::map<std::string, double>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
prob_dict  getProbDict(QVec , int selectMax = -1);

/**
* @brief  Get pmeasure result as dict
* @ingroup Core
* @param[in]  QProg&  Quantum program
* @param[in]  QVec  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::vector<std::pair<size_t, double>>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
prob_tuple probRunTupleList(QProg &,QVec , int selectMax = -1);

/**
* @brief  Get pmeasure result as list
* @ingroup Core
* @param[in]  QProg&  Quantum program
* @param[in]  QVec  Pmeasure qubits vector
* @param[in]  int selectmax:the returned value num
* @return     prob_vec  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
prob_vec probRunList(QProg &, QVec , int selectMax = -1);


/**
* @brief  Get pmeasure result as dict
* @ingroup Core
* @param[in]  QProg&  Quantum program
* @param[in]  QVec  pmeasure qubits vector
* @param[in]  int Selectmax:the returned value num
* @return     std::map<std::string, double>  result
* @note   SelectMax  can not exceed (1ull << the size of qubits vector)
*/
prob_dict  probRunDict(QProg &, QVec , int selectMax = -1);

/**
* @brief  Measure run with configuration
* @ingroup Core
* @param[in]  QProg&  Quantum program
* @param[in]  std::vector<ClassicalCondition>&  cbits vector
* @param[in]  int Shots:the repeat num  of measure operate
* @return     std::map<std::string, size_t>   result
*/
std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int);

/**
* @brief  Quick measure operate
* @ingroup Core
* @param[in]  QVec  qubits vector
* @param[in]  int Shots:the repeat num  of measure operate
* @return     std::map<std::string,size_t>  result
*/
std::map<std::string, size_t> quickMeasure(QVec , int);


/**
* @brief  AccumulateProbability
* @ingroup Core
* @param[in]  prob_vec & prob_list  Abstract Quantum program pointer
* @return     prob_vec  
*/
prob_vec accumulateProbability(prob_vec &prob_list);


/**
* @brief  Quick measure
* @ingroup Core
* @param[in]  QVec  qubits vector
* @param[in]  int Shots:the repeat num  of measure operate
* @param[in]  prob_vec& accumulate  Probabilites 
* @return     std::map<std::string,size_t>  Results
*/
std::map<std::string, size_t> quick_measure(QVec qubit_vector, int shots,
    prob_vec& accumulate_probabilites);

/**
* @brief  Get quantum state
* @ingroup Core
* @return     qstat  Quantum state  vector
*/
QStat getQState();
/**
* @brief  Init a Quantum machine
* @ingroup Core
* @param[in]  QMachineType
* @return     QPanda::QuantumMachine*  Quantum machine pointer
* @see  QMachineType
* @note  default  Quantum machine type :cpu
*/
QuantumMachine *initQuantumMachine(QMachineType type=CPU);

/**
* @brief  Destroy Quantum machine
* @ingroup Core
* @param[in]  QuantumMachine* Quantum machine pointer
* @return     void  
*/
void destroyQuantumMachine(QuantumMachine * qvm);

/**
* @brief  Measure All  ClassicalCondition
* @ingroup Core
* @param[in]  QVec&  qubits vector
* @param[in]  std::vector<ClassicalCondition>  Cbits vector
* @return    QPanda::QProg   Quantum program
*/
QPanda::QProg MeasureAll(QVec, std::vector<ClassicalCondition>);

extern QProg transformOriginIRToQProg(std::string filePath, QuantumMachine* qm, QVec &qv, std::vector<ClassicalCondition> &cv);

extern QProg convert_originir_to_qprog(std::string file_path, QuantumMachine *qm);
extern QProg convert_originir_to_qprog(std::string file_path, QuantumMachine * qm, QVec &qv, std::vector<ClassicalCondition> &cv);

extern QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine *qm);
extern QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine *qm, QVec &qv, std::vector<ClassicalCondition> &cv);

extern QProg convert_qasm_to_qprog(std::string file_path, QuantumMachine* qvm);
extern QProg convert_qasm_to_qprog(std::string file_path, QuantumMachine* qvm, QVec &qv, std::vector<ClassicalCondition> &cv);

extern QProg convert_qasm_string_to_qprog(std::string qasm_str, QuantumMachine* qvm);
extern QProg convert_qasm_string_to_qprog(std::string qasm_str, QuantumMachine* qvm, QVec &qv, std::vector<ClassicalCondition> &cv);

/*will delete*/
size_t getAllocateCMem();
prob_tuple PMeasure(QVec qubit_vector, int select_max);
prob_vec PMeasure_no_index(QVec qubit_vector);

/* new interface */

 /**
 * @brief  Get allocate cbit number
 * @ingroup Core
 * @return    size_t Cbit number
 */
size_t getAllocateCMemNum();

/**
* @brief  pMeasure
* @ingroup Core
* @param[in]  QVec qubit  vector
* @param[in]  int  Selectmax:the returned value num
* @return     std::vector<std::pair<size_t, double>>   result
*/
prob_tuple pMeasure(QVec qubit_vector, int select_max);

/**
* @brief  pMeasure only return result  with no index
* @ingroup Core
* @param[in]  QVec qubit vector
* @return     prob_vec  result
*/
prob_vec pMeasureNoIndex(QVec qubit_vector);


QPANDA_END
#endif // !_CORE_H
