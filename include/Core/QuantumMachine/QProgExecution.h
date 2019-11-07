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
/*! \file QProgExecution.h */
#ifndef QPROG_EXECUTION_H
#define QPROG_EXECUTION_H


#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/Tools/Traversal.h"
#include <map>
#include <type_traits>


QPANDA_BEGIN

/**
* @namespace QPanda
*/

/**
* @defgroup Utilities
* @brief QPanda2  base  Utilities  classes and  interface
*/

/**
* @class QGateCompare
* @ingroup Utilities
* @brief Qunatum Gate Compare
*/
class QProgExecution 
{
public:

	/*!
	* @brief  Execution traversal qgatenode
	* @param[in|out]  AbstractQGateNode*  quantum gate
	* @param[in]  AbstractQGateNode*  quantum gate
	* @param[in]  TraversalConfig  traversal config
	* @param[in]  QPUImpl*  virtual quantum processor
	* @return     void
	* @exception invalid_argument
	* @note
	*/
	virtual void execute(std::shared_ptr<AbstractQGateNode> cur_node, 
		std::shared_ptr<QNode> parent_node,
		TraversalConfig & param,
		QPUImpl* qpu);

	/*!
	* @brief  Execution traversal measure node
	* @param[in|out]  AbstractQuantumMeasure*  measure node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @param[in]  TraversalConfig   traversal config
	* @param[in]  QPUImpl*  virtual quantum processor
	* @return     void
	* @exception invalid_argument
	* @note
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node,
		std::shared_ptr<QNode> parent_node,
		TraversalConfig & param,
		QPUImpl* qpu);

	/*!
	* @brief  Execution traversal control flow node
	* @param[in|out]  AbstractControlFlowNode*  control flow node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @param[in]  TraversalConfig  traversal config
	* @param[in]  QPUImpl*  virtual quantum processor
	* @return     void
	* @exception invalid_argument
	* @note
	*/
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node,
		std::shared_ptr<QNode> parent_node,
		TraversalConfig & param,
		QPUImpl* qpu)
	{
		Traversal::traversal(cur_node, *this, param, qpu);
	}


	/*!
	* @brief  Execution traversal qcircuit
	* @param[in|out]  AbstractQuantumCircuit*  quantum circuit
	* @param[in]  AbstractQGateNode*  quantum gate
	* @param[in]  TraversalConfig  traversal config
	* @param[in]  QPUImpl*  virtual quantum processor
	* @return     void
	* @exception invalid_argument
	* @note
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
		std::shared_ptr<QNode> parent_node,
		TraversalConfig & param,
		QPUImpl* qpu);
	/*!
	* @brief  Execution traversal qprog
	* @param[in|out]  AbstractQuantumProgram*  quantum prog
	* @param[in]  AbstractQGateNode*  quantum gate
	* @param[in]  TraversalConfig  traversal config
	* @param[in]  QPUImpl*  virtual quantum processor
	* @return     void
	* @exception invalid_argument
	* @note
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumProgram> cur_node,
		std::shared_ptr<QNode> parent_node,
		TraversalConfig & param,
		QPUImpl* qpu)
	{
		Traversal::traversal(cur_node, *this, param, qpu);
	}
	/*!
	* @brief  Execution traversal qprog
	* @param[in|out]  AbstractClassicalProg*  quantum prog
	* @param[in]  AbstractQGateNode*  quantum gate
	* @param[in]  TraversalConfig  traversal config
	* @param[in]  QPUImpl*  virtual quantum processor
	* @return     void
	* @exception invalid_argument
	* @note
	*/
	virtual void execute(std::shared_ptr<AbstractClassicalProg> cur_node,
		std::shared_ptr<QNode> parent_node,
		TraversalConfig & param,
		QPUImpl* qpu)
	{
		cur_node->eval();
	}

	void get_return_value(std::map<std::string, bool> & result)
	{
		for (auto aiter : m_result)
		{
			result.insert(aiter);
		}
	}
private:

	std::map<std::string, bool> m_result;
};

/* new interface */

/**
* @brief  execute qprog
* @param[in]  prog quantum program,
* @param[in]  const std::vector<std::vector<std::string>>&    support gates
* @return     size_t     Unsupported QGate number
* @note
*/
inline void execute_qprog(QProg prog,QPUImpl * qpu,TraversalConfig & param)
{
	if (nullptr == qpu)
	{
		QCERR("param error");
		throw std::invalid_argument("param error");
	}
	QProgExecution prog_exec;
	prog_exec.execute(prog.getImplementationPtr(),nullptr, param, qpu);
	
}



QPANDA_END



#endif
