/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
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
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include <map>
QPANDA_BEGIN

struct QProgInfoCount
{
    size_t node_num{ 0 };
    size_t gate_num{ 0 };
    size_t layer_num{ 0 };

    size_t single_gate_num{ 0 };
    size_t double_gate_num{ 0 };
    size_t multi_control_gate_num{ 0 };

    size_t single_gate_layer_num{ 0 };
    size_t double_gate_layer_num{ 0 };

    std::map<GateType, size_t> selected_gate_nums;
};

/**
* @brief  Count Quantum Program clock cycle
* @ingroup Utilities
*/
class QProgClockCycle 
{
public:
    QProgClockCycle();
    QProgClockCycle(QuantumMachine *qm);
    ~QProgClockCycle();
    size_t count(QProg &prog, bool optimize = false);
    QProgInfoCount count_layer_info(QProg &prog, std::vector<GateType> selected_types);

private:

    void get_time_map();
    size_t getDefalutQGateTime(GateType gate_type);
    size_t getQGateTime(GateType gate_type);
    std::map<GateType, size_t> m_gate_time;
    QuantumMachine *m_machine{nullptr};
    QProgInfoCount m_prog_info;
};

/**
* @brief  Get  quantum program clock cycle
* @param[in]  QProg &   quantum program
* @param[in]	QuantumMachine*		quantum machine pointer
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
size_t getQProgClockCycle(QProg &prog, QuantumMachine *qm, bool optimize = false);

/*new interface*/

/**
* @brief  Get quantum program clock cycle
* @param[in]  QProg &   quantum program
* @param[in]  QuantumMachine*	quantum machine pointer
* @ingroup Utilities
*/
size_t get_qprog_clock_cycle(QProg &prog, QuantumMachine *qm, bool optimize = false);

/**
* @brief  Get  quantum program clock cycle by chip
* @param[in]  LayeredTopoSeq &   quantum program layer
* @ingroup Utilities
*/
size_t get_qprog_clock_cycle_chip(LayeredTopoSeq &layer_info, std::map<GateType, size_t> gate_time_map);

template <typename _Ty>
QProgInfoCount count_prog_info(_Ty& node, std::vector<GateType> selected_types = {})
{
    QProg prog;
    prog << node;

    QProgClockCycle counter;
    return counter.count_layer_info(prog, selected_types);
}

QPANDA_END
#endif // _QPROG_CLOCK_CYCLE_H



