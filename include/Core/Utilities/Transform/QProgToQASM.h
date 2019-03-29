/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgToQASM.h
Author: Yulei
Created in 2018-7-19

Classes for Travesing QProg QGates as std::string use QASM instruction set .

Update@2018-8-31
update comment

*/
#ifndef  _QPROGTOQASM_H_
#define  _QPROGTOQASM_H_
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/ControlFlow.h"
QPANDA_BEGIN

/*
Travesal QProg print QGates us QASM instruction set
*/
class QProgToQASM 
{
public:
    QProgToQASM();
   ~QProgToQASM();

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
     friend std::ostream & operator<<(std::ostream &, const QProgToQASM &);

     /*
     out insturctionsQASM
     param:
     return:
     std::string

     Note:
     None
     */
     std::string insturctionsQASM();

     /*
     Traversal QProg to instructions
     param:
     pQProg: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
     void progToQASM(AbstractQuantumProgram *);

     /*
     Traversal QProg to instructions
     param:
     pQProg: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
     void qProgToQASM(AbstractQuantumProgram *);

     /*
     QGate to QASM instruction
     param:
     pGate: AbstractQGateNode pointer
     return:
     None

     Note:
     None
     */
     void qProgToQASM(AbstractQGateNode *);

     /*
     Traversal QProg to instructions
     param:
     pCtrFlow: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
     void qProgToQASM(AbstractControlFlowNode *);

     /*
     handleIfWhileQASM
     param:
     pCtrFlow: AbstractQuantumProgram pointer
     return:
     None

     Note:
     None
     */
     void handleIfWhileQASM(AbstractControlFlowNode *, std::string);

     /*
     Traversal QCircuit to QASM instructions
     param:
     pQProg: AbstractQuantumCircuit pointer
     return:
     None

     Note:
     None
     */
     void qProgToQASM(AbstractQuantumCircuit *);

     /*
     QMeasure to QASM instruction
     param:
     pMeasure: AbstractQuantumMeasure pointer
     return:
     None

     Note:
     None
     */
     void qProgToQASM(AbstractQuantumMeasure *);

     /*
     handleDaggerNode
     param:
     pNode: QNode pointer
     return:
     None

     Note:
     None
     */
     void handleDaggerNode(QNode *, int);

     /*
     Traversal Dagger Circuit to QASM
     param:
     pNode: QNode pointer
     return:
     None

     Note:
     None
     */
     void handleDaggerCir(QNode *);
     
     /*
     Traversal QNode to QASM
     param:
     pNode: QNode pointer
     return:
     None

     Note:
     None
     */
     void qProgToQASM(QNode *);

private:
    std::map<int, std::string>  m_gatetype;
    std::vector<std::string> m_qasm;
};

std::ostream & operator<<(std::ostream & out, const QProgToQASM &qasm_qprog);
QPANDA_END
#endif
