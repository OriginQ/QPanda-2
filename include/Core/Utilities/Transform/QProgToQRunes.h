/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgToQRunes.h
Author: Yulei
Created in 2018-7-19

Classes for Travesing QProg QGates as string use QRunes instruction set .

Update@2018-8-31
update comment

*/

#ifndef  _PROGTOQRUNES_H_
#define  _PROGTOQRUNES_H_
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/ControlFlow.h"
QPANDA_BEGIN

class QProgToQRunes 
{
public:
    QProgToQRunes();
   ~QProgToQRunes();

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
    friend std::ostream & operator<<(std::ostream &, const QProgToQRunes &);

    /*
    out insturctionsQRunes
    param:
    return:
    string

    Note:
    None
    */
    std::string insturctionsQRunes();

    /*
    Traversal QProg to instructions
    param:
    pQProg: AbstractQuantumProgram pointer
    return:
    None

    Note:
    None
    */
    void qProgToQRunes(AbstractQuantumProgram *);

    /*
     Traversal QProg to instructions
     param:
     pQProg: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
     void transformQProg(AbstractQuantumProgram *);

    /*
     QGate to QRunes instruction
     param:
     pGate: AbstractQGateNode pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(AbstractQGateNode *);

    /*
     Traversal QProg to instructions
     param:
     pCtrFlow: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(AbstractControlFlowNode *);

    /*
     Traversal QCircuit to QRunes instructions
     param:
     pQProg: AbstractQuantumCircuit pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(AbstractQuantumCircuit *);

    /*
    QMeasure to QRunes instruction
    param:
    pMeasure: AbstractQuantumMeasure pointer
    return:
    None

    Note:
    None
    */
    void transformQProg(AbstractQuantumMeasure *);

    /*
     Traversal QNode to QRunes
     param:
     pNode: QNode pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(QNode *);
private:
    std::vector<std::string> m_QRunes;
    std::map<int, std::string>  m_gatetype;
};

std::ostream & operator<<(std::ostream & out, const QProgToQRunes &qrunes_prog);
QPANDA_END
#endif
