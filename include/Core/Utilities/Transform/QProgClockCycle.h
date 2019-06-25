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
/*! \file QProgClockCycle.h */
#ifndef _QPROG_CLOCK_CYCLE_H
#define _QPROG_CLOCK_CYCLE_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include <map>
QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @class QProgClockCycle
* @brief  Count Quantum Program clock cycle
* @ingroup Utilities
* @see
      @code
            init();
            auto qubits = qAllocMany(4);
            auto prog = CreateEmptyQProg();
            prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
                    << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI/4);
            auto time = getQProgClockCycle(prog);
            std::cout << "clockCycle : " << time << std::endl;

            finalize();
      @endcode
*/
class QProgClockCycle
{
public:
    QProgClockCycle(QuantumMachine *qm);
    ~QProgClockCycle();
    void traversal(QProg &prog);
    size_t count();
private:
    size_t countQProgClockCycle(AbstractQuantumProgram *prog);
    size_t countQCircuitClockCycle(AbstractQuantumCircuit *circuit);
    size_t countQWhileClockCycle(AbstractControlFlowNode *qwhile);
    size_t countQIfClockCycle(AbstractControlFlowNode *qif);

    size_t getQGateTime(AbstractQGateNode *gate);
    size_t countQNodeClockCycle(QNode * node);
    size_t getDefalutQGateTime(GateType gate_type);
    std::map<GateType, size_t> m_gate_time;
    size_t m_count;
};

/**
* @brief  Get quantum program clock cycle
* @ingroup QuantumMachine
* @param[in]  QProg& quantum program
* @return     Eigen::size_t   Clock cycle  result
*/
size_t getQProgClockCycle(QProg &prog, QuantumMachine *qm);
QPANDA_END
#endif // _QPROG_CLOCK_CYCLE_H



