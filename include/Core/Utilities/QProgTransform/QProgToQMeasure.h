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
/*! \file QProgToQMeasure.h */

#ifndef _QPROGTOQMEASURE_H
#define _QPROGTOQMEASURE_H

#include "Core/Utilities/Tools/Traversal.h"

QPANDA_BEGIN

/**
* @brief cast QProg to QMeasure
* @ingroup Utilities
*/
class QProgToQMeasure : public TraversalInterface<>
{
public:
	QProgToQMeasure()
		:m_qmeasure_count(0)
	{}

	~QProgToQMeasure() {}

	/*!
	* @brief  Execution traversal qgatenode
	* @param[in,out]  AbstractQGateNode*  quantum gate
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		QCERR("cast qprog to qmeasure fail!");
		throw run_fail("cast qprog to qmeasure fail!");
	}

	/*!
	* @brief  Execution traversal measure node
	* @param[in,out]  AbstractQuantumMeasure*  measure node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node)
	{
		m_qmeasure_count++;
		m_qmeasure_node = cur_node;
		if (m_qmeasure_count > 1)
		{

			QCERR("cast qprog to qmeasure fail!");
			throw run_fail("cast qprog to qmeasure fail!");
		}
	}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{
		QCERR("cast qprog to qmeasure fail!");
		throw run_fail("cast qprog to qmeasure fail!");
	}

	/*!
	* @brief  Execution traversal control flow node
	* @param[in,out]  AbstractControlFlowNode*  control flow node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
	{
		QCERR("cast qprog to qmeasure fail!");
		throw run_fail("cast qprog to qmeasure fail!");
	}


	/*!
	* @brief  Execution traversal qcircuit
	* @param[in,out]  AbstractQuantumCircuit*  quantum circuit
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
	{
		Traversal::traversal(cur_node, false, *this);
	}
	/*!
	* @brief  Execution traversal qprog
	* @param[in,out]  AbstractQuantumProgram*  quantum prog
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		Traversal::traversal(cur_node, *this);
	}
	/*!
	* @brief  Execution traversal qprog
	* @param[in,out]  AbstractClassicalProg*  quantum prog
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		QCERR("cast qprog to qmeasure fail!");
		throw run_fail("cast qprog to qmeasure fail!");
	}

	virtual void execute(std::shared_ptr<AbstractQNoiseNode>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		QCERR("cast qprog to qmeasure fail!");
		throw run_fail("cast qprog to qmeasure fail!");
	}

	virtual void execute(std::shared_ptr<AbstractQDebugNode>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		QCERR("cast qprog to qmeasure fail!");
		throw run_fail("cast qprog to qmeasure fail!");
	}

	std::shared_ptr<AbstractQuantumMeasure> get_qmeasure()
	{
		if (!m_qmeasure_node)
		{
			QCERR("cast qprog to qmeasure fail!");
			throw run_fail("cast qprog to qmeasure fail!");
		}
		return m_qmeasure_node;
	}

private:
	std::shared_ptr<AbstractQuantumMeasure> m_qmeasure_node;
	size_t m_qmeasure_count;
};


/**
* @brief Cast Quantum Program To Quantum Measure
* @ingroup Utilities
* @param[in]  QProg	  quantum program
* @return     QMeasure  quantum measure
*/
static QMeasure cast_qprog_qmeasure(QProg prog)
{
	QProgToQMeasure traversal_class;
	traversal_class.execute(prog.getImplementationPtr(), nullptr);

	QMeasure mea = QMeasure(traversal_class.get_qmeasure());

	return mea;
}

QPANDA_END
#endif //!_QPROGTOQMEASURE_H