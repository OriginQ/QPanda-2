/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGateCompare.h
Author: Wangjing
Created in 2018-6-25

Classes for get count that QVM don't support QGates.

Update@2018-8-30
update comment

*/

#ifndef  QGATE_COMPARE_H_
#define  QGATE_COMPARE_H_

#pragma once

#include "QuantumCircuit/QProgram.h"
#include <map>
#include "QuantumCircuit/QGlobalVariable.h"
QPANDA_BEGIN
/*
get the number that QGate of QProg is not in setting instructions
*/
class QGateCompare {
public:
    QGateCompare();
    virtual ~QGateCompare();
    
    /*
    get the number that QGate of QProg is not in setting instructions
    param:
        p_prog: target QProg
        instructions: set instructions
    return:
        not support number

    Note:
        None
    */
    static size_t countQGateNotSupport(AbstractQuantumProgram *p_prog,
                                       const std::vector<std::vector<std::string>> &instructions);

    /*
    get the number that QGate of QGate is not in setting instructions
    param:
        p_gata: target QGate
        instructions: set instructions
    return:
        not support number

    Note:
        None
    */
    static size_t countQGateNotSupport(AbstractQGateNode *p_gata,
                                       const std::vector<std::vector<std::string>> &instructions);

    /*
    get the number that QGate of AbstractControlFlowNode is not in setting instructions
    param:
        p_controlflow: target AbstractControlFlowNode
        instructions: set instructions
    return:
        not support number

    Note:
        None
    */
    static size_t countQGateNotSupport(AbstractControlFlowNode *p_controlflow,
                                       const std::vector<std::vector<std::string>> &instructions);

    /*
    get the number that QGate of QCircuit is not in setting instructions
    param:
        p_circuit: target QCircuit
        instructions: set instructions
    return:
        not support number

    Note:
        None
    */
    static size_t countQGateNotSupport(AbstractQuantumCircuit *p_circuit,
                                       const std::vector<std::vector<std::string>> &instructions);

protected:
    /*
    get the number that QGate of QNode is not in setting instructions
    param:
        p_node: target QNode
        instructions: set instructions
    return:
        not support number

    Note:
        None
    */
    static size_t countQGateNotSupport(QNode *p_node, 
                                       const std::vector<std::vector<std::string>> &instructions);

    /*
    get the result of the item is or not in setting instructions
    param:
        item: target item
        instructions: set instructions
    return:
        not support number

    Note:
        None
    */
    static bool isItemExist(const std::string &item,
                            const std::vector<std::vector<std::string>> &instructions);
private:
};



QPANDA_END


#endif