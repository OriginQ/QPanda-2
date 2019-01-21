/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGateTypeCounter.h
Author: Wangjing
Created in 2018-10-12

Classes for counting QProg clock cycle.

Update@2018-10-12
update comment

*/
#ifndef _QPROG_CLOCK_CYCLE_H
#define _QPROG_CLOCK_CYCLE_H
#include "QuantumCircuit/QProgram.h"
#include <map>
QPANDA_BEGIN

/*
count QProg clock cycle
*/
class QProgClockCycle
{
public:
    QProgClockCycle(std::map<int, size_t> gate_time);
    ~QProgClockCycle();

    /*
    count clock cycle of QProg
    param:
        prog: target QProg
    return:
        clock cycle

    Note:
        None
    */
    size_t countQProgClockCycle(AbstractQuantumProgram *prog);

    /*
    count clock cycle of QCurcuit
    param:
        circuit: target QCurcuit
    return:
        clock cycle

    Note:
        None
    */
    size_t countQCircuitClockCycle(AbstractQuantumCircuit *circuit);

    /*
    count clock cycle of Qwhile
    param:
        circuit: target qwhile
    return:
        clock cycle

    Note:
        None
    */
    size_t countQWhileClockCycle(AbstractControlFlowNode *qwhile);

    /*
    count clock cycle of QIf
    param:
        circuit: target qif
    return:
        clock cycle

    Note:
        None
    */
    size_t countQIfClockCycle(AbstractControlFlowNode *qif);

    /*
    get QGate time
    param:
        circuit: target QGate
    return:
        QGate time

    Note:
        None
    */
    size_t getQGateTime(AbstractQGateNode *gate);
protected:
    /*
    count clock cycle of QNode
    param:
        circuit: target QNode
    return:
        clock cycle

    Note:
        None
    */
    size_t countQNodeClockCycle(QNode * node);

    /*
    count clock cycle of QNode
    param:
        circuit: target QNode
    return:
        clock cycle

    Note:
        None
    */
    size_t getDefalutQGateTime(int gate_type);
private:
    std::map<int, size_t> m_gate_time;
};
QPANDA_END
#endif // _QPROG_CLOCK_CYCLE_H



