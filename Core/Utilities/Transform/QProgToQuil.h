/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgToQuil.h
Author: Wangjing
Created in 2018-7-19

Classes for Travesing QProg and store QGates as std::string use Quil instruction set .

Update@2018-8-30
update comment

*/
#ifndef  _QPROG_TO_QUIL_
#define  _QPROG_TO_QUIL_

#include "QuantumCircuit/QProgram.h"
#include <map>
#include "QuantumCircuit/QGlobalVariable.h"
#include <string>
QPANDA_BEGIN


using QGATE_SPACE::angleParameter;


enum MetadataGateType {
    METADATA_SINGLE_GATE,
    METADATA_DOUBLE_GATE
};



/*
Travesal QProg print QGates us Quil instruction set
*/
class QProgToQuil
{
public:
    QProgToQuil();
    ~QProgToQuil();

    /*
    Traversal QProg to instructions
    param:
        p_prog: AbstractQuantumProgram pointer
    return:
        None

    Note:
        None
    */
    void progToQuil(AbstractQuantumProgram *p_prog);

    /*
    overload operator <<
    param:
        out: output stream
        prog: QProg
    return:
        output stream
    Note:
        None
    */
    friend std::ostream & operator<<(std::ostream &out, const QProgToQuil &prog);

    /*
    Traversal QProg
    param:
        None
    return:
        instructions

    Note:
        None
    */
    std::string getInsturctions();
protected:
    /*
    QGate to Quil instruction
    param:
        p_gate: AbstractQGateNode pointer
    return:
        None

    Note:
        None
    */
    void gateToQuil(AbstractQGateNode *p_gate);
    /*
    Traversal QCircuit to Quil instructions
    param:
        p_circuit: AbstractQuantumCircuit pointer
    return:
        None

    Note:
        None
    */
    void circuitToQuil(AbstractQuantumCircuit *p_circuit);
    /*
    QMeasure to Quil instruction
    param:
        p_measure: AbstractQuantumMeasure pointer
    return:
        None

    Note:
        None
    */
    void measureToQuil(AbstractQuantumMeasure *p_measure);
    /*
    Traversal QNode to Quil
    param:
        p_node: QNode pointer
    return:
        None

    Note:
        None
    */
    void nodeToQuil(QNode * p_node);

    /*
    convert QGate data to std::string and store the std::string in m_instructs
    param:
        p_gate: AbstractQGateNode
    return:
        None

    Note:
        None
    */
    void dealWithQuilGate(AbstractQGateNode *p_gate);

    /*
    transform QPanda gate to Quil base gate and store Quil gate as a QCircuit
    param:
        p_gate: AbstractQGateNode
    return:
        Quil base store as a QCircuit

    Note:
        None
    */
    QCircuit transformQPandaBaseGateToQuilBaseGate(AbstractQGateNode *p_gate);
private:
    std::map<int, std::string> m_gate_type_map;
    std::vector<std::string>  m_instructs;
};

std::string qProgToQuil(QProg &prog);
QPANDA_END
#endif // ! _QPROG_TO_QUIL_
