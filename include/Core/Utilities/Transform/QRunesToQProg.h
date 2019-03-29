/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QRunesToQprog.h
Author: Yulei
Created in 2018-08-01

Classes for Travesing QRunes instruction set into Qprog

Update@2018-8-31
update comment

*/

#ifndef  _QRUNESTOQPROG_H_
#define  _QRUNESTOQPROG_H_
#include "Core/QPanda.h"
#include <functional>
QPANDA_BEGIN

enum QRunesKeyWords {
    DAGGER = 24,
    ENDAGGER,
    CONTROL,
    ENCONTROL,
    QIF,
    ELSE,
    ENDQIF,
    QWHILE,
    ENDQWHILE,
    MEASURE,
};


class QRunesToQprog {
public:
    QRunesToQprog() = delete;
    QRunesToQprog(std::string);
    ~QRunesToQprog() {};

    void qRunesParser(QProg&);


    /*
    Traversal QRunes instructions
    param:
    m_QRunes: std::vector<std::string>
    return:
    None

    Note:
    None
    */

    void qRunesAllocation(std::vector<std::string> &, QProg&);

    /*
    Traversal QRunes instructions
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    return:
    instructions number

    Note:
    None
    */
    int traversalQRunes(std::vector<std::string>::iterator, QNode *);

    /*
    handle SingleGate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gateName:gate type
    qubit_addr
    return:
    instructions number

    Note:
    None
    */
    int handleSingleGate(std::vector<std::string>::iterator, QNode *, 
                         const std::string &, int);

    /*
    handle Double Gate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gateName:gate type
    ctr_qubit_addr
    tar_qubit_addr
    return:
    instructions number

    Note:
    None
    */
    int handleDoubleGate(std::vector<std::string>::iterator , QNode *, 
                         const std::string &, int,int);

    /*
    handle Angle Gate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gateName:gate type
    qubit_addr
    gate_angle
    return:
    instructions number

    Note:
    None
    */
    int handleAngleGate(std::vector<std::string>::iterator, QNode *, 
                         const std::string &, int, double);

    /*
    handle Double Angle Gate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gateName:gate type
    qubit_addr
    gate_angle
    return:
    instructions number

    Note:
    None
    */
    int handleDoubleAngleGate(std::vector<std::string>::iterator , QNode *,
        const std::string &, int, int, double);
    /*
    handle Measure Gate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gateName:gate type
    qubit_addr
    creg_addr
    return:
    instructions number

    Note:
    None
    */
    int handleMeasureGate(std::vector<std::string>::iterator , QNode *, 
                         const std::string &, int, int);

    /*
    handle DaggerCircuit
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    return:
    instructions number

    Note:
    None
    */
    int handleDaggerCircuit(std::vector<std::string>::iterator, QNode *);

    /*
    handle Control Circuit
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    all_ctr_qubits :std::vector<Qubit*>
    ctr_info: conctrl info
    return:
    instructions number

    Note:
    None
    */
    int handleControlCircuit(std::vector<std::string>::iterator, QNode *,
                             std::vector<Qubit*> &, std::string &);

private:

    std::vector<std::string> m_QRunes;
    std::vector<std::string> m_keyWords;

    QVec m_all_qubits;
    std::vector<ClassicalCondition > m_all_cregs;

    std::map<std::string, std::function<QGate(Qubit *)> > m_singleGateFunc;
    std::map<std::string, std::function<QGate(Qubit *, Qubit*)> > m_doubleGateFunc;
    std::map<std::string, std::function<QGate(Qubit *,double)> > m_angleGateFunc;
    std::map<std::string, std::function<QGate(Qubit *, Qubit*, double)> > m_doubleAngleGateFunc;

    std::string  m_sFilePath;
};
QPANDA_END

#endif
