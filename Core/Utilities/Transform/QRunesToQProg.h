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


#include "QuantumCircuit/QProgram.h"
#include <map>
#include "QuantumCircuit/QGlobalVariable.h"
QPANDA_BEGIN

enum QRunesLines {
    FIRST_LINE,
    SECOND_LINE,
    THIRD_LINE
};

enum QRunesKeyWords {
    DAGGER = 23,
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

/*
Travesal QRunes instruction set
*/
class QRunesToQprog {
public:
    QRunesToQprog();
    virtual ~QRunesToQprog();


    /*
    qRunesParser
    param: None
    return:
    QProg

    Note:
    None
    */
    QProg qRunesParser();


    /*
    Traversal QRunes instructions
    param:
    m_qrunes: std::vector<std::string>
    return:
    None

    Note:
    None
    */
    void qRunesAllocation(std::vector<std::string> &m_qrunes);

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
    int traversalQRunes(std::vector<std::string>::iterator iter, QNode *qNode);

    /*
    handle SingleGate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gate_name:gate type
    qubit_addr
    return:
    instructions number

    Note:
    None
    */
    int handleSingleGate(std::vector<std::string>::iterator iter, QNode *qNode, 
                         const std::string &gate_name, int qubit_addr);

    /*
    handle Double Gate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gate_name:gate type
    ctr_qubit_addr
    tar_qubit_addr
    return:
    instructions number

    Note:
    None
    */
    int handleDoubleGate(std::vector<std::string>::iterator iter, QNode *qNode, 
                         const std::string &gate_name, int ctr_qubit_addr,int tar_qubit_addr);

    /*
    handle Angle Gate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gate_name:gate type
    qubit_addr
    gate_angle
    return:
    instructions number

    Note:
    None
    */
    int handleAngleGate(std::vector<std::string>::iterator iter, QNode *qNode, 
                         const std::string &gate_name, int qubit_addr, double gate_angle);

    /*
    handle Measure Gate
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    gate_name:gate type
    qubit_addr
    creg_addr
    return:
    instructions number

    Note:
    None
    */
    int handleMeasureGate(std::vector<std::string>::iterator iter, QNode *qNode, 
                         const std::string &key_word, int qubit_addr, int creg_addr);

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
    int handleDaggerCircuit(std::vector<std::string>::iterator iter, QNode *qNode);

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
    int handleControlCircuit(std::vector<std::string>::iterator iter, QNode *qNode,
                             std::vector<Qubit*> &all_ctr_qubits, std::string &ctr_info);

    /*
    handle QifProg
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    exper :ClassicalCondition
    return:
    instructions number

    Note:
    None
    */
    int handleQifProg(std::vector<std::string>::iterator iter, QNode *qNode, ClassicalCondition &exper);

    /*
    handle QWhileProg
    param:
    iter: std::vector<std::string>::iterator
    qNode:QNode pointer
    exper :ClassicalCondition
    return:
    instructions number

    Note:
    None
    */
    int handleQWhileProg(std::vector<std::string>::iterator iter, QNode *qNode, ClassicalCondition &exper);

    /*
    check IfWhile Node Legal
    param:
    ctr_str :ClassicalCondition
    return:
    ClassicalCondition  statement

    Note:
    None
    */
    ClassicalCondition checkIfWhileLegal(std::string ctr_str);

    /*
    set QRunes File Path
    param:
    file_path :QRunes file path
    return:
    None

    Note:
    None
    */
    void static setFilePath(std::string file_path);

private:
    std::vector<std::string> m_qrunes;
    std::vector<std::string> m_check_legal;
    std::vector<std::string> m_first_check;
    std::vector<Qubit* > m_all_qubits;
    std::vector<ClassicalCondition > m_all_cregs;

    int m_max_qubits;
    int m_max_cregs;

    static std::string  m_file_path;

    QProg m_new_qprog;
};
QPANDA_END

#endif
