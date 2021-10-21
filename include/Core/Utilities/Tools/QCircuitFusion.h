#ifndef  QCIRCUITFUSION_H
#define  QCIRCUITFUSION_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include <iostream>
#include <complex>
#include <vector>

QPANDA_BEGIN

class Fusion
{
public:

	/**
	* @brief
	* @param[in]  QCircuit& cir target optimize circuit
	* @param[in]  QuantumMachine* qvm  quantummachine get used qubits
	* @return    void
	* @ingroup QuantumCircuit
	*/
	void aggregate_operations(QCircuit& cir, QuantumMachine* qvm);

	/**
	* @brief
	* @param[in]  QProg& prog target optimize prog
	* @param[in]  QuantumMachine* qvm  quantummachine get used qubits
	* @return     void
	* @ingroup QuantumProg
	*/
	void aggregate_operations(QProg& prog, QuantumMachine* qvm);

protected:
	bool _exclude_escaped_qubits(std::vector<int>& fusing_qubits,
		const QGate& tgt_op)  const;

	template<class T>
	void _fusion_gate(T& prog,const int fusion_bit, QuantumMachine* qvm);

	QGate _generate_operation_internal(const std::vector<QGate>& fusioned_ops,
		const std::vector<int>& qubits, QuantumMachine* qvm);

	QGate _generate_operation(std::vector<QGate>& fusioned_ops, QuantumMachine* qvm);

	template<class T>
	void _allocate_new_operation(T& prog, NodeIter& index_itr,
		std::vector<NodeIter>& fusing_op_itrs, QuantumMachine* qvm);

};
QPANDA_END

#endif //  QCIRCUITFUSION_H