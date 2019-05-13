/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgDataParse.h
Author: Wangjing
Created in 2018-8-9

Classes for parsing binary file to QProg.

Update@2018-8-30
update comment

*/
/*! \file QProgDataParse.h */
#ifndef QPROGDATPARSE_H
#define QPROGDATPARSE_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <list>
#include <string>
#include <map>
#include "Core/Utilities/Transform/QProgToQuil.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QProgram.h"
#include <stack>
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Transform/QProgTransform.h"

/**
* @namespace QPanda
* @namespace QGATE_SPACE
*/
QPANDA_BEGIN

/**
* @class QProgDataParse
* @ingroup Utilities
* @brief parse binary file to quantum program
*/
class QProgDataParse
{
    /**
    * @class DataNode
    * @brief Quantum program node data
    */
    union DataNode
    {
        DataNode() {}
        DataNode(uint32_t data) : qubit_data(data) {}
        DataNode(float data) : angle_data(data) {}

        uint32_t qubit_data;
        float angle_data;
    };

    typedef std::vector<std::pair<uint32_t, DataNode>>     dataList_t;
    typedef std::map<int, std::string>                     gateMap_t;
    typedef std::map<int, std::string>                     operatorMap_t;
public:
    QProgDataParse(QuantumMachine *qm);
    ~QProgDataParse();


    /**
    * @brief  Load  qprog data from file
    * @param[in]  std::string& filename
    * @retval 1    load  success
    * @retval 0    load  failed
    */
    bool load(const std::string &filename);

    /**
    * @brief  Load  qprog data from data vector
    * @param[in]  std::vector<uint8_t>& data
    * @retval 1     load  success
    * @retval 0     load  failed
    */
    bool load(const std::vector<uint8_t> &data);

    /**
    * @brief   Parse binary file to QProg
    * @param[out]  QProg& prog
    * @retval 1     parse success
    * @retval 0     parse failed
    * @exception  invalid_argument  parse error
    */
    bool parse(QProg &prog);
    QVec getQubits() const;
    std::vector<ClassicalCondition> getCbits() const;

private:

    void parseDataNode(QProg &prog, const uint32_t &tail_number);
    void parseQGateDataNode(QProg &prog, const uint32_t &type_and_number, const uint32_t &qubits_data);
    void parseQMeasureDataNode(QProg &prog, uint32_t qubits_data);
    void parseCExprCBitDataNode(const uint32_t &data);
    void parseCExprOperateDataNode(const uint32_t &data);
    void parseCExprConstValueDataNode(const int &data);
    void parseCExprEvalDataNode(const int &data);
    void parseQIfDataNode(QProg &prog, const uint32_t &data);
    void parseQWhileDataNode(QProg &prog, uint32_t data);
    float getAngle(const std::pair<uint32_t, DataNode> &data);
    int getCBitValue(const std::pair<uint32_t, DataNode> &data_node);

    std::string m_filename;
    uint32_t m_node_counter;
    dataList_t m_data_vector;
    QVec m_qubits;
    std::vector<ClassicalCondition> m_cbits;

    std::vector<std::pair<uint32_t, DataNode>>::iterator m_iter;
    std::stack<ClassicalCondition> m_stack_cc;
    QuantumMachine *m_quantum_machine;
};

/**
* @brief  Parse quantum program interface for  binary file
* @ingroup Utilities
* @param[in]  QuantumMachine* quantum machine pointer
* @param[in]  std::string& filename
* @param[out]  QVec& qubits  
* @param[out]  std::vector<ClassicalCondition>& cbits
* @param[out]  QProg& Quantum program
* @retval 1   parse success
* @retval 0   parse failed
* @exception  runtime_error  parse file error
*/
bool binaryQProgFileParse(QuantumMachine *qm, const std::string &filename, QVec &qubits,
                          std::vector<ClassicalCondition> &cbits, QProg &prog);

/**
* @brief  Parse quantum program interface for  binary data vector
* @ingroup Utilities
* @param[in]  QuantumMachine* quantum machine pointer
* @param[in]  std::vector<uint8_t>& data   binary data vector
* @param[out]  QVec& qubits
* @param[out]  std::vector<ClassicalCondition>& cbits
* @param[out]  QProg& Quantum program
* @retval 1  parse success
* @retval 0  parse failed
* @exception  runtime_error  parse file error
*/
bool binaryQProgDataParse(QuantumMachine *qm, const std::vector<uint8_t>& data, QVec & qubits, 
                          std::vector<ClassicalCondition>& cbits, QProg & prog);

QPANDA_END

#endif // QPROGDATPARSE_H
