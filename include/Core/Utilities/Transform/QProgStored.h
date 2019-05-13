/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgStored.h
Author: Wangjing
Created in 2018-8-4

Classes for saving QProg as binary file.

Update@2018-8-30
update comment

*/
/*! \file QProgStored.h */
#ifndef QPROGSTORED_H
#define QPROGSTORED_H

#include <stdio.h>
#include <fstream>
#include <istream>
#include <list>
#include <string>
#include <map>
#include "Core/Utilities/Transform/QProgToQuil.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"

/**
* @namespace QPanda
* @namespace QGATE_SPACE
*/
QPANDA_BEGIN


/**
* @class  QProgStored
* @ingroup Utilities
* @brief  Utilities class for quantum program stored to binary data
*/
class QProgStored
{
    union DataNode
    {
        DataNode() {}
        DataNode(uint32_t uiData) : qubit_data(uiData) {}
        DataNode(float fData) : angle_data(fData) {}

        uint32_t qubit_data;
        float angle_data;
    };
    typedef std::vector<std::pair<uint32_t, DataNode>>      dataList_t;
    typedef std::map<int, std::string>                      gateMap_t;

    typedef std::map<std::string, int>                      operatorMap_t;
public:
  QProgStored(QuantumMachine *qm);
  ~QProgStored();

  /**
   * @brief  transform quantum  program
   * @param[in]  QProg& quantum  program
   * @return     void
   */
  void transform(QProg &prog);

  /**
    * @brief  Store quantum program data
    * @param[in]  const std::string& filename  
    * @return     void  
    */
  void store(const std::string &);

  /**
    * @brief  get quantum program binary data
    * @return     std::vector<uint8_t>   quantum program binary data vector
    */
  std::vector<uint8_t> getInsturctions();
  
private:
    void transformQProg(AbstractQuantumProgram *prog);
    void transformQCircuit(AbstractQuantumCircuit *circuit);
    void transformQControlFlow(AbstractControlFlowNode *controlflow);
    void transformQGate(AbstractQGateNode *gate);
    void transformQMeasure(AbstractQuantumMeasure *measure);
    void transformQNode(QNode *node);

    void transformQIfProg(AbstractControlFlowNode *controlflow);
    void transformQWhilePro(AbstractControlFlowNode *controlflow);
    void transformCExpr(CExpr *cexpr);
    void transformClassicalProg(AbstractClassicalProg *cc_pro);
    void handleQGateWithOneAngle(AbstractQGateNode *gate);
    void handleQGateWithFourAngle(AbstractQGateNode *gate);
    void addDataNode(const QProgStoredNodeType &type, const DataNode &data, const bool &is_dagger = false);

    QProg m_QProg;
    uint32_t m_node_counter;
    uint32_t m_qubit_number;

    uint32_t m_cbit_number;
    dataList_t m_data_vector;
    operatorMap_t m_operator_map;
    QuantumMachine * m_quantum_machine;
};

/**
* @brief  Store quantum program in binary
* @ingroup Utilities
* @param[in]  const uint32_t& qubit number  
* @param[in]  const uint32_t& cbit number 
* @param[in]  QProg& quantum program  
* @param[in]  const std::string& filename  
* @return     void  
*/
void storeQProgInBinary(QProg &prog, QuantumMachine *qm, const std::string &filename);

/**
* @brief  Get quantum program binary data
* @ingroup Utilities
* @param[in]  QProg& quantum program  
* @param[in]  QuantumMachine& quantum 
* @return     std::vector<uint8_t>   quantum program binary data
*/
std::vector<uint8_t> transformQProgToBinary(QProg &prog, QuantumMachine *qm);

QPANDA_END

#endif // QPROGSTORED_H
