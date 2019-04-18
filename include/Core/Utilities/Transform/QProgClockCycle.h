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
    QProgClockCycle(std::map<GateType, size_t> gate_time);
    ~QProgClockCycle();


    /**
    * @brief  Count QProg clock cycle
    * @param[in]  AbstractQuantumProgram*  Abstract Quantum program pointer
    * @return     size_t  Clock cycle
    * @exception  invalid_argument  Quantum program pointer is a nullptr
    */
    size_t countQProgClockCycle(AbstractQuantumProgram *prog);

    /**
    * @brief  Count QCircuit Clock Cycle
    * @param[in]  AbstractQuantumCircuit*  Abstract Quantum circuit pointer
    * @return     size_t   Clock cycle
    * @exception  invalid_argument  Quantum circuit pointer is a nullptr
    */
    size_t countQCircuitClockCycle(AbstractQuantumCircuit *circuit);

    /**
    * @brief  Count QProg clock cycle
    * @param[in]  AbstractControlFlowNode*  Abstract Quantum ControlFlow pointer
    * @return     size_t   Clock cycle
    * @exception  invalid_argument  Quantum controlflow qwhile pointer is a nullptr
    */
    size_t countQWhileClockCycle(AbstractControlFlowNode *qwhile);

    /**
    * @brief  Count QProg clock cycle
    * @param[in]  AbstractControlFlowNode*  Abstract Quantum ControlFlow pointer
    * @return     size_t    Clock cycle
    * @exception  invalid_argument  Quantum controlflow qif pointer is a nullptr
    */
    size_t countQIfClockCycle(AbstractControlFlowNode *qif);

    /**
    * @brief  Get QGate Time
    * @param[in]  AbstractQGateNode * gate  Abstract Quantum gate pointer
    * @return     size_t   Set time
    * @exception   invalid_argument  Gate is a nullptr
    */
    size_t getQGateTime(AbstractQGateNode *gate);
protected:
    size_t countQNodeClockCycle(QNode * node);

    size_t getDefalutQGateTime(GateType gate_type);
private:
    std::map<GateType, size_t> m_gate_time;
};

/**
* @brief  Get quantum program clock cycle
* @ingroup QuantumMachine
* @param[in]  QProg& quantum program
* @return     Eigen::size_t   Clock cycle  result
*/
size_t getQProgClockCycle(QuantumMachine *qm, QProg &prog);
QPANDA_END
#endif // _QPROG_CLOCK_CYCLE_H



