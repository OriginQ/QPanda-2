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
#include "Core/Utilities/Tools/Traversal.h"
#include <map>
QPANDA_BEGIN
/**
* @namespace QPanda
*/


class QProgClockCycle {
public:
    QProgClockCycle(QuantumMachine *qm);
    ~QProgClockCycle();
    size_t count(QProg &prog);
private:
    size_t getDefalutQGateTime(GateType gate_type);
    size_t getQGateTime(GateType gate_type);
    std::map<GateType, size_t> m_gate_time;
};

/**
* @class QProgClockCycle
* @brief  Count Quantum Program clock cycle
* @param[in]  QProg &   quantum program
* @param[in]	QuantumMachine *		quantum machine pointer
* @ingroup Utilities
* @see
	  @code
			init();
			auto qubits = qAllocMany(4);
			auto prog = CreateEmptyQProg();
			prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
					<< iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI/4);
			extern QuantumMachine* global_quantum_machine;
			auto time = getQProgClockCycle(prog,global_quantum_machine );
			std::cout << "clockCycle : " << time << std::endl;

			finalize();
	  @endcode
*/
size_t getQProgClockCycle(QProg &prog, QuantumMachine *qm);
QPANDA_END
#endif // _QPROG_CLOCK_CYCLE_H



