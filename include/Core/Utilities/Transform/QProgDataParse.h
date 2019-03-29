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
#include "Core/QuantumCircuit/ClassicalProgam.h"


QPANDA_BEGIN
#ifndef _DATPARSE_
#define _DATPARSE_

const unsigned short kUshortMax = 65535;
const int kCountMoveBit = 16;
#define DEF_QPROG_FILENAME        "QProg.dat"

enum QProgStoredNodeType {
    QPROG_PAULI_X_GATE = 1u,
    QPROG_PAULI_Y_GATE,
    QPROG_PAULI_Z_GATE,
    QPROG_X_HALF_PI,
    QPROG_Y_HALF_PI,
    QPROG_Z_HALF_PI,
    QPROG_HADAMARD_GATE,
    QPROG_T_GATE,
    QPROG_S_GATE,
    QPROG_RX_GATE,
    QPROG_RY_GATE,
    QPROG_RZ_GATE,
    QPROG_U1_GATE,
    QPROG_U2_GATE,
    QPROG_U3_GATE,
    QPROG_U4_GATE,
    QPROG_CU_GATE,
    QPROG_CNOT_GATE,
    QPROG_CZ_GATE,
    QPROG_CPHASE_GATE,
    QPROG_ISWAP_GATE,
    QPROG_ISWAP_THETA_GATE,
    QPROG_SQISWAP_GATE,
    QPROG_GATE_ANGLE,
    QPROG_MEASURE_GATE,
    QPROG_QIF_NODE,
    QPROG_QWHILE_NODE,
    QPROG_CEXPR_CBIT,
    QPROG_CEXPR_OPERATOR,
    QPROG_CEXPR_CONSTVALUE,
    QPROG_CEXPR_EVAL
};
#endif // _DATPARSE_
using QGATE_SPACE::angleParameter;

/*
parse binary file to QProg
*/
class QProgDataParse
{
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
    QProgDataParse();
    ~QProgDataParse();

    /*
    open file
    param:
        None
    return:
        None

    Note:
        None
    */
    bool load(const std::string &filename);

    /*
    open file
    param:
        None
    return:
        None

    Note:
        None
    */
    bool load(const std::vector<uint8_t> &data);

    /*
    parse file data to QProg
    param:
        prog : file data to QProg
    return:
        None

    Note:
        None
    */
    bool parse(QProg &prog);
    inline QVec getQubits() const;
    inline  std::vector<ClassicalCondition> getCbits() const;

protected:
    void parseDataNode(QProg &prog, const uint32_t &tail_number);

    /*
    Parse QGate node data
    param:
        prog : QProg
        type_and_number: data can get node type and qubit number
        qubits_data: data can get all qubits
    return:
        None

    Note:
        None
    */
    void parseQGateDataNode(QProg &prog, const uint32_t &type_and_number, const uint32_t &qubits_data);

    /*
    Parse QMeasure node data
    param:
        prog : QProg
        qubits_data: data can get all qubits
    return:
        None

    Note:
        None
    */
    void parseQMeasureDataNode(QProg &prog, uint32_t qubits_data);

    /*
    Parse CExpr cbit node data
    param:
        data : data can get CExpr cbit msg
    return:
        None

    Note:
        None
    */
    void parseCExprCBitDataNode(const uint32_t &data);

    /*
    Parse CExpr operator node data
    param:
        data : data can get CExpr oprator
    return:
        None

    Note:
        None
    */
    void parseCExprOperateDataNode(const uint32_t &data);

    /*
    Parse CExpr const value node data
    param:
        data : data can get CExpr const value
    return:
        None

    Note:
        None
    */
    void parseCExprConstValueDataNode(const int &data);

    void parseCExprEvalDataNode(const int &data);

    /*
    Parse QIf node data
    param:
        prog : QProg
        data : data can get QIf msg
    return:
        None

    Note:
        None
    */
    void parseQIfDataNode(QProg &prog, const uint32_t &data);

    /*
    Parse QWhile node data
    param:
        prog : QProg
        data : data can get QWhile msg
    return:
        None

    Note:
        None
    */
    void parseQWhileDataNode(QProg &prog, uint32_t data);

    /*
    get angle from data
    param:
        data : store angle msg
    return:
        angle

    Note:
        None
    */
    float getAngle(const std::pair<uint32_t, DataNode> &data);
    int getCBitValue(const std::pair<uint32_t, DataNode> &data_node);
private:
    std::string m_filename;
    uint32_t m_node_counter;
    dataList_t m_data_vector;
    QVec m_qubits;
    std::vector<ClassicalCondition> m_cbits;

    std::vector<std::pair<uint32_t, DataNode>>::iterator m_iter;
    std::stack<ClassicalCondition> m_stack_cc;
};

bool binaryQProgFileParse(QVec &qubits, std::vector<ClassicalCondition> &cbits,
                          QProg &prog, const std::string &filename = DEF_QPROG_FILENAME);

bool binaryQProgDataParse(QVec & qubits, std::vector<ClassicalCondition>& cbits, QProg & prog,
                          const std::vector<uint8_t>& data);

QPANDA_END

#endif // QPROGDATPARSE_H
