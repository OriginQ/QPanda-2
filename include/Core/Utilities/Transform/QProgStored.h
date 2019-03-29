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
#include <fstream>
#include <istream>
#include <list>
#include <string>
#include <map>
#include "Core/Utilities/Transform/QProgToQuil.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
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

#endif

using QGATE_SPACE::angleParameter;

/*
store QProg witt binary data
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
    QProgStored(const uint32_t &qubit_number, const uint32_t &cbit_number, QProg &prog);
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
    std::vector<uint8_t> getQProgBinaryData();
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
    void traversalClassicalProg(AbstractClassicalProg *cc_pro);
    void handleQGateWithOneAngle(AbstractQGateNode *gate);
    void handleQGateWithFourAngle(AbstractQGateNode *gate);
    void addDataNode(const QProgStoredNodeType &type, const DataNode &data, const bool &is_dagger = false);
private:
    QProg m_QProg;
    uint32_t m_node_counter;
    uint32_t m_qubit_number;

    uint32_t m_cbit_number;
    dataList_t m_data_vector;
    operatorMap_t m_operator_map;
};

void storeQProgInBinary(const uint32_t &qubit_number, const uint32_t &cbit_number, QProg &prog,
                        const std::string &filename = DEF_QPROG_FILENAME);

std::vector<uint8_t> getQProgBinaryData(const uint32_t &qubit_number, const uint32_t &cbit_number, QProg &prog);

QPANDA_END

#endif // QPROGSTORED_H
