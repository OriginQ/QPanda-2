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
#include <list>
#include <string>
#include <map>
#include "QProgToQuil.h"
#include "QuantumCircuit/QGlobalVariable.h"
#include "QuantumCircuit/QProgram.h"
#include <stack>

QPANDA_BEGIN
#ifndef _DATPARSE_
#define _DATPARSE_

const unsigned short kUshortMax = 65535;
const int kCountMoveBit = 16;
#define DEF_QPROG_FILENAME        "QProg.dat"

enum QProgStoredNodeType{
    QPROG_NODE_TYPE_PAULI_X_GATE = 1,
    QPROG_NODE_TYPE_PAULI_Y_GATE,
    QPROG_NODE_TYPE_PAULI_Z_GATE,
    QPROG_NODE_TYPE_X_HALF_PI,
    QPROG_NODE_TYPE_Y_HALF_PI,
    QPROG_NODE_TYPE_Z_HALF_PI,
    QPROG_NODE_TYPE_HADAMARD_GATE,
    QPROG_NODE_TYPE_T_GATE,
    QPROG_NODE_TYPE_S_GATE,
    QPROG_NODE_TYPE_RX_GATE,
    QPROG_NODE_TYPE_RY_GATE,
    QPROG_NODE_TYPE_RZ_GATE,
    QPROG_NODE_TYPE_U1_GATE,
    QPROG_NODE_TYPE_U2_GATE,
    QPROG_NODE_TYPE_U3_GATE,
    QPROG_NODE_TYPE_U4_GATE,
    QPROG_NODE_TYPE_CU_GATE,
    QPROG_NODE_TYPE_CNOT_GATE,
    QPROG_NODE_TYPE_CZ_GATE,
    QPROG_NODE_TYPE_CPHASE_GATE,
    QPROG_NODE_TYPE_ISWAP_GATE,
    QPROG_NODE_TYPE_SQISWAP_GATE,
    QPROG_NODE_TYPE_GATE_ANGLE,
    QPROG_NODE_TYPE_MEASURE_GATE,
    QPROG_NODE_TYPE_QIF_NODE,
    QPROG_NODE_TYPE_QWHILE_NODE,
    QPROG_NODE_TYPE_CEXPR_CBIT,
    QPROG_NODE_TYPE_CEXPR_OPERATOR,
};

#endif
using QGATE_SPACE::angleParameter;


/*
parse binary file to QProg
*/
class QProgDataParse
{
    typedef unsigned short                      ushort_t;
    typedef unsigned int                        uint_t;
    union DataNode
    {
        DataNode(){}
        DataNode(uint_t data): qubit_data(data){}
        DataNode(float data): angle_data(data){}

        uint_t qubit_data;
        float angle_data;
    };

    typedef std::list<std::pair<uint_t, DataNode>>        dataList_t;
    typedef std::map<int, std::string>                    gateMap_t;
    typedef std::map<int, std::string>                    operatorMap_t;
public:
    QProgDataParse(const std::string &filename);
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
    bool loadFile();

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

protected:
    void parseDataNode(QProg &prog, const uint_t tail_number);

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
    void parseQGateDataNode(QProg &prog, const uint_t type_and_number, const uint_t qubits_data);

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
    void parseQMeasureDataNode(QProg &prog, uint_t qubits_data);

    /*
    Parse CExpr cbit node data
    param:
        data : data can get CExpr cbit msg
    return:
        None

    Note:
        None
    */
    void parseCExprCBitDataNode(const uint_t data);

    /*
    Parse CExpr operator node data
    param:
        data : data can get CExpr oprator
    return:
        None

    Note:
        None
    */
    void parseCExprOperateDataNode(const uint_t data);

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
    void parseQIfDataNode(QProg &prog, const uint_t data);

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
    void parseQWhileDataNode(QProg &prog, uint_t data);

    /*
    get angle from data
    param:
        data : store angle msg
    return:
        angle

    Note:
        None
    */
    float getAngle(const std::pair<uint_t, DataNode> &data);
private:
    std::string m_filename;
    uint_t m_file_length;
    uint_t m_node_counter;
    dataList_t m_data_list;

    std::list<std::pair<uint_t, DataNode>>::iterator m_iter;
    std::stack<ClassicalCondition> m_stack_cc;
};
bool binaryQProgFileParse(QProg &prog, 
    const std::string &filename = DEF_QPROG_FILENAME);
QPANDA_END
#endif // QPROGDATPARSE_H
