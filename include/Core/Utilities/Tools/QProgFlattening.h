/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file QProgFlattening.h */

#ifndef _QPROGFLATTENING_H
#define _QPROGFLATTENING_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"

QPANDA_BEGIN

struct QubitPointerCmp
{
	bool operator() (Qubit *left, Qubit *right) const
	{
		return (left->getPhysicalQubitPtr()->getQubitAddr() < right->getPhysicalQubitPtr()->getQubitAddr());
	}
};

/**
* @class QProgFlattening  
* @ingroup Utilities
* @brief flatten quantum program and quantum circuit
*/
class QProgFlattening : public TraversalInterface<QProg&>
{
public:

	QProgFlattening(bool is_full_faltten = false);
	~QProgFlattening();

	void flatten_circuit(QCircuit &src_cir);
	void flatten_prog(QProg &src_prog);

	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)override;
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)override;
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)override;
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)override;
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)override;
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)override;
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QProg &prog)override;

	/**
	* @brief Flatten QProg to QCircuit
	* @ingroup Utilities
	* @param[in]  QProg& the target QProg
	* @return Converted circuit 
	* @Note: The input QProg must be no-nesting, and only QGate type is supported. 
	*/
	static QCircuit prog_flatten_to_cir(QProg& prog);

protected:
	void flatten_by_type(std::shared_ptr<QNode> node, QProg& flattened_prog);
	QVec get_two_qvec_union(QVec qv_1, QVec qv_2);

private:
	bool m_full_flatten;
	QVec m_global_ctrl_qubits;
	bool m_global_dagger;
};

/**
* @brief Flatten Quantum Circuit
* @ingroup Utilities
* @param[in,out]  QCircuit&	  circuit program 
* @return     void
*/
void flatten(QCircuit& src_cir);

/**
* @brief Full Flatten Quantum Program
* @ingroup Utilities
* @param[in,out]  QProg&	  quantum program
* @param[in] bool whether to expand the target prog completely
* @return     void
*/
void flatten(QProg& src_prog, bool b_full_flatten = true);

QPANDA_END
#endif // !_QPROGFLATTENING_H