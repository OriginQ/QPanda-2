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
/*! \file QProgToQGate.h */
#ifndef _QPROGTOQGATE_H
#define _QPROGTOQGATE_H

#include "Core/Utilities/Tools/Traversal.h"

QPANDA_BEGIN
class QProgToQGate : public TraversalInterface<>
{
public:
	QProgToQGate()
		:m_qgate_count(0)
	{}

	~QProgToQGate() {}

    /*!
    * @brief  Execution traversal qgatenode
    * @param[in|out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
    {
		m_qgate_count++;
		m_qgate_node = cur_node;
		if (m_qgate_count > 1)
		{
			
			QCERR("cast qprog to qgate fail!");
			throw run_fail("cast qprog to qgate fail!");
		}
    }

    /*!
    * @brief  Execution traversal measure node
    * @param[in|out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node)
    {
        QCERR("cast qprog to qgate fail!");
        throw run_fail("cast qprog to qgate fail!");
    }

    /*!
    * @brief  Execution traversal control flow node
    * @param[in|out]  AbstractControlFlowNode*  control flow node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
    {
        QCERR("cast qprog to qgate fail!");
        throw run_fail("cast qprog to qgate fail!");
    }


    /*!
    * @brief  Execution traversal qcircuit
    * @param[in|out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
    {
        Traversal::traversal(cur_node, false, *this);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in|out]  AbstractQuantumProgram*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
    {
		Traversal::traversal(cur_node, *this);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in|out]  AbstractClassicalProg*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		QCERR("cast qprog to qgate fail!");
		throw run_fail("cast qprog to qgate fail!");
	}

	std::shared_ptr<AbstractQGateNode> get_qgate()
	{
		if (!m_qgate_node)
		{
			QCERR("cast qprog to qgate fail!");
			throw run_fail("cast qprog to qgate fail!");
		}
		return m_qgate_node;
	}

private:
	std::shared_ptr<AbstractQGateNode> m_qgate_node;
	size_t m_qgate_count;
};

/**
* @brief Cast Quantum Program To Quantum Gate
* @ingroup Utilities
* @param[int]  QProg	  quantum program
* @return     QGate  quantum gate
* @exception  run_fail
* @note
*/
static QGate cast_qprog_qgate(QProg prog)
{
	QProgToQGate traversal_class;
	traversal_class.execute(prog.getImplementationPtr(), nullptr);

	QGate gate = QGate(traversal_class.get_qgate());

	return gate;
}

QPANDA_END
#endif //!_QPROGTOQGATE_H