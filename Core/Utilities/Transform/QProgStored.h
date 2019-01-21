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


#ifndef QPROGSTORED_H
#define QPROGSTORED_H

#include <stdio.h>
#include <istream>
#include <list>
#include <string>
#include <map>
#include "QProgToQuil.h"
#include "QuantumCircuit/QGlobalVariable.h"
#include "QuantumCircuit/QProgram.h"

QPANDA_BEGIN
#ifndef _DATPARSE_
#define _DATPARSE_


const unsigned short kUshortMax = 65535;
const int kCountMoveBit = 16;
#define DEF_QPROG_FILENAME        "QProg.dat"


enum QProgStoredNodeType {
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
store QProg witt binary data
*/
class QProgStored
{
    typedef unsigned short                      ushort_t;
    typedef unsigned int                        uint_t;
    union DataNode
    {
        DataNode() {}
        DataNode(uint_t uiData) : qubit_data(uiData) {}
        DataNode(float fData) : angle_data(fData) {}

        uint_t qubit_data;
        float angle_data;
    };
    typedef std::list<std::pair<uint_t, DataNode>>        dataList_t;
    typedef std::map<int, std::string>                    gateMap_t;

    typedef std::map<std::string, int>                    operatorMap_t;
public:
    QProgStored(QProg &prog);
    ~QProgStored();

    /*
    Traversal QProg
    param:
        None
    return:
        None

    Note:
        None
    */
    void traversal();

    /*
    store QProg data as a binary file
    param:
        filename : file name 
    return:
        None

    Note:
        None
    */
    void store(const std::string &filename = DEF_QPROG_FILENAME);
protected:
    /*
    traverse QProg
    param:
        p_prog : AbstractQuantumProgram pointer
    return:
        None

    Note:
        None
    */
    void traversalQProg(AbstractQuantumProgram *p_prog);

    /*
    traverse Qcircuit
    param:
        p_circuit : AbstractQuantumCircuit pointer
    return:
        None

    Note:
        None
    */
    void traversalQCircuit(AbstractQuantumCircuit *p_circuit);
    
    /*
    traverse QControlFlow
    param:
        p_controlflow : AbstractControlFlowNode pointer
    return:
        None

    Note:
        None
    */
    void traversalQControlFlow(AbstractControlFlowNode *p_controlflow);

    /*
    traverse QIf
    param:
        p_controlflow : AbstractControlFlowNode pointer
    return:
        None

    Note:
        None
    */
    void traversalQIfProg(AbstractControlFlowNode *p_controlflow);

    /*
    traverse QWhile
    param:
        p_controlflow : AbstractControlFlowNode pointer
    return:
        None

    Note:
        None
    */
    void traversalQWhilePro(AbstractControlFlowNode *p_controlflow);

    /*
    traverse QGate
    param:
        p_gate : AbstractQGateNode pointer
    return:
        None

    Note:
        None
    */
    void traversalQGate(AbstractQGateNode *p_gate);

    /*
    traverse QMeasure
    param:
        p_measure : AbstractQuantumMeasure pointer
    return:
        None

    Note:
        None
    */
    void traversalQMeasure(AbstractQuantumMeasure *p_measure);

    /*
    store QProg data as a binary file
    param:
        p_node : QNode pointer
    return:
        None

    Note:
        None
    */
    void traversalQNode(QNode *p_node);

    /*
    store QProg data as a binary file
    param:
        p_cexpr : CExpr pointer
    return:
        None

    Note:
        None
    */
    void traversalCExpr(CExpr *p_cexpr);
private:
    QProg m_QProg;
    uint_t m_file_length;
    uint_t m_node_counter;
    dataList_t m_data_list;

    gateMap_t m_gate_type_map;
    operatorMap_t m_operator_map;
};
void qProgBinaryStored(QProg &prog, 
    const std::string &filename = DEF_QPROG_FILENAME);
QPANDA_END
#endif // QPROGSTORED_H
