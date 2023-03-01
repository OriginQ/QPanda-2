/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

fusion gate

Author: Guowenbo
*/

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
    Fusion() {
        std::fill_n(distances_, 64, -1);
    };
	/**
	* @brief
	* @param[in]  QCircuit& cir target optimize circuit
	* @param[in]  QuantumMachine* qvm  quantummachine get used qubits
	* @return    void
	* @ingroup QuantumCircuit
	*/
	void aggregate_operations(QCircuit& cir);

	/**
	* @brief
	* @param[in]  QProg& prog target optimize prog
	* @param[in]  QuantumMachine* qvm  quantummachine get used qubits
	* @return     void
	* @ingroup QuantumProg
	*/
	void aggregate_operations(QProg& prog);


    /**
    * @brief
    * @param[in]  QProg& prog target optimize prog
    * @param[in]  QuantumMachine* qvm  quantummachine get used qubits
    * @return     void
    * @ingroup QuantumProg
    */
    void multi_bit_gate_fusion(QProg& prog);

protected:
    double distance_cost(const std::vector<QGate>& ops,
        const int from,
        const int until) const;
    /*double estimate_cost(QProg& prog,NodeIter &itr_star, NodeIter &itr_end)const;*/

    void add_optimize_qubits(std::vector<int>& fusion_qubits, const QGate& gate) const;

    bool aggreate(std::vector<QGate>& prog, QVec& used_qv);
    //bool aggreate(QProg& prog, QuantumMachine* qvm);

    QGate _generate_oracle_gate(const std::vector<QGate>& fusioned_ops,
        const std::vector<int>& qubits, QVec& used_qv);

	bool _exclude_escaped_qubits(std::vector<int>& fusing_qubits,
		const QGate& tgt_op)  const;

	template<class T>
	void _fusion_gate(T& prog,const int fusion_bit, QVec& used_qv);

	QGate _generate_operation_internal(const std::vector<QGate>& fusioned_ops,
		const std::vector<int>& qubits, QVec& used_qv);

	QGate _generate_operation(std::vector<QGate>& fusioned_ops, QVec& used_qv);

	template<class T>
	void _allocate_new_operation(T& prog, NodeIter& index_itr,
		std::vector<NodeIter>& fusing_op_itrs, QVec& used_qv);

    void _allocate_new_gate(std::vector<QGate>& prog, int index,
        std::vector<int>& fusing_op_itrs, QVec& used_qv);


private:
    double distance_factor = 1.8;
    double distances_[64];
};
QPANDA_END

#endif //  QCIRCUITFUSION_H