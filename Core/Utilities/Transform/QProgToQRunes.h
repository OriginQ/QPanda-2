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

#include "QuantumCircuit/QProgram.h"
#include "QuantumCircuit/QGlobalVariable.h"
#include "QuantumCircuit/ControlFlow.h"
#include "QuantumCircuit/QuantumMeasure.h"
#include "QuantumCircuit/QGate.h"
#include "QuantumCircuit/QCircuit.h"
#include <map>
QPANDA_BEGIN

/*
Travesal QProg print QGates us QRunes instruction set
*/
class QProgToQRunes 
{
public:
    QProgToQRunes();
    virtual ~QProgToQRunes();

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
    friend std::ostream & operator<<(std::ostream & out, const QProgToQRunes &qrunes_prog);

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
    pQpro: AbstractQuantumProgram pointer
    return:
    None

    Note:
    None
    */
    void qProgToQRunes(AbstractQuantumProgram *pQpro);

    /*
     Traversal QProg to instructions
     param:
     pQpro: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
     void transformQProg(AbstractQuantumProgram *pQpro);

    /*
     QGate to QRunes instruction
     param:
     pGate: AbstractQGateNode pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(AbstractQGateNode *pQGata);

    /*
     Traversal QProg to instructions
     param:
     pCtrFlow: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(AbstractControlFlowNode *pCtrFlow);

    /*
     Traversal QCircuit to QRunes instructions
     param:
     pQpro: AbstractQuantumCircuit pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(AbstractQuantumCircuit *pCircuit);

    /*
    QMeasure to QRunes instruction
    param:
    pMeasure: AbstractQuantumMeasure pointer
    return:
    None

    Note:
    None
    */
    void transformQProg(AbstractQuantumMeasure *pMeasure);

    /*
     Traversal QNode to QRunes
     param:
     pNode: QNode pointer
     return:
     None

     Note:
     None
     */
    void transformQProg(QNode * pNode);
private:
    std::vector<std::string> m_QRunes;
    std::map<int, std::string>  m_gatetype;
};
QPANDA_END
#endif
