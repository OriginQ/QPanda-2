/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Traversal.h
Author: doumenghan
Created in 2019-4-16

Classes for get the shortes path of graph

*/
#ifndef _TRAVERSAL_H
#define _TRAVERSAL_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QReset.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include <memory>
QPANDA_BEGIN

/**
* @class TraversalConfig
* @brief traversal config
* @ingroup Utilities
*/
class TraversalConfig
{
public:
    size_t m_qubit_number; /**< quantum bit number */
    std::map<std::string, bool> m_return_value; /**< MonteCarlo result */
    bool m_is_dagger;
    std::vector<QPanda::Qubit *> m_control_qubit_vector;

    double m_rotation_angle_error{ 0 };
    bool m_can_optimize_measure = true;
    std::vector<size_t> m_measure_qubits;
    std::vector<CBit *> m_measure_cc;

    TraversalConfig(double rotation_angle_error = 0)
        : m_qubit_number(0), m_is_dagger(false)

    {
        m_rotation_angle_error = rotation_angle_error / 2;
    }
};

 
/**
* @class Traversal
* @brief Traversing all the nodes of the linked qprog/qcircuit/control_flow_node
* @ingroup Utilities
*/
class Traversal
{
public:

    /*!
    * @brief  Traversing qprog control flow circuit
    * @param[in]  AbstractControlFlowNode*  Control flow nodes that need to be traversed
    * @param[in]  TraversalInterface*  The method object needed for traversal
    * @return     void
    */
    template<typename T,typename... Args>
    static void traversal(std::shared_ptr<AbstractControlFlowNode> control_flow_node, T & func_class, Args&& ... func_args)
    {
        //static_assert(!std::is_base_of<TraversalInterface, T>::value, "bad Traversal type");
        if (nullptr == control_flow_node)
        {
            QCERR("control_flow_node is nullptr");
            throw std::invalid_argument("control_flow_node is nullptr");
        }

        auto pNode = std::dynamic_pointer_cast<QNode>(control_flow_node);

        if (nullptr == pNode)
        {
            QCERR("Unknown internal error");
            throw std::runtime_error("Unknown internal error");
        }
        auto iNodeType = pNode->getNodeType();

        if (WHILE_START_NODE == iNodeType)
        {
            auto true_branch_node = control_flow_node->getTrueBranch();
            traversalByType(true_branch_node, pNode, func_class, std::forward<Args>(func_args)...);
        }
        else if (QIF_START_NODE == iNodeType)
        {
            auto true_branch_node = control_flow_node->getTrueBranch();
            traversalByType(true_branch_node, pNode, func_class, std::forward<Args>(func_args)...);
            auto false_branch_node = control_flow_node->getFalseBranch();

            if (nullptr != false_branch_node)
            {
                traversalByType(false_branch_node, pNode, func_class, std::forward<Args>(func_args)...);
            }
        }
    }

    /*!
    * @brief  Traversing qcircuit
    * @param[in]  AbstractQuantumCircuit*  QCircuit nodes that need to be traversed
    * @param[in]  TraversalInterface*  The method object needed for traversal
    * @param[in]  bool  Whether the quantum circuit needs to be transposed
    * @return     void
    */
   template<typename T,typename... Args>
    static void traversal(std::shared_ptr<AbstractQuantumCircuit> qcircuit_node, bool identify_dagger, T & func_class, Args&& ... func_args)
    {
        if (nullptr == qcircuit_node)
        {
            QCERR("pQCircuit is nullptr");
            throw std::invalid_argument("pQCircuit is nullptr");
        }

        auto aiter = qcircuit_node->getFirstNodeIter();

        if (aiter == qcircuit_node->getEndNodeIter())
            return;

        auto pNode = std::dynamic_pointer_cast<QNode>(qcircuit_node);

        if (nullptr == pNode)
        {
            QCERR("Unknown internal error");
            throw std::runtime_error("Unknown internal error");
        }
        auto is_dagger = false;
        if (identify_dagger)
        {
            is_dagger = qcircuit_node->isDagger();
        }

        if (is_dagger)
        {
            auto aiter = qcircuit_node->getLastNodeIter();
            if (nullptr == *aiter)
            {
                return;
            }
            while (aiter != qcircuit_node->getHeadNodeIter())
            {
                //auto next = --aiter;
				if (aiter == nullptr)
				{
					break;
				}
                traversalByType(*aiter, pNode, func_class, std::forward<Args>(func_args)...);
                //aiter = next;
				--aiter;
            }

        }
        else
        {
            auto aiter = qcircuit_node->getFirstNodeIter();
            auto end_iter = qcircuit_node->getEndNodeIter();
            while (aiter != end_iter)
            {
                auto next = aiter.getNextIter();
                traversalByType(*aiter, pNode, func_class, std::forward<Args>(func_args)...);
                aiter = next;
            }
        }
    }


    /*!
    * @brief  Traversing qprog
    * @param[in]  AbstractQuantumProgram*  QProg nodes that need to be traversed
    * @param[in]  TraversalInterface*  The method object needed for traversal
    * @return     void
    */
    template<typename T,typename... Args>
    static void traversal(std::shared_ptr<AbstractQuantumProgram> qprog_node,T & func_class, Args&& ... func_args)
    {
        if (nullptr == qprog_node)
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }

        auto aiter = qprog_node->getFirstNodeIter();
        auto end_iter = qprog_node->getEndNodeIter();
		if (aiter == qprog_node->getEndNodeIter())
			return;


        auto pNode = std::dynamic_pointer_cast<QNode>(qprog_node);

        if (nullptr == pNode)
        {
            QCERR("pNode is nullptr");
            throw std::invalid_argument("pNode is nullptr");
        }

        while (aiter != end_iter)
        {
            auto next = aiter.getNextIter();
            traversalByType(*aiter, pNode, func_class, std::forward<Args>(func_args)...);
            aiter = next;
        }
    }

    /*!
    * @brief  traversalByType
    * @param[in]  QNode*  nodes that need to be traversed
    * @param[in]  parent_node*  nodes that need to be traversed
    * @param[in]  TraversalInterface*  The method object needed for traversal
    * @return     void
    */
    template<typename T,typename... Args>
    static void traversalByType(std::shared_ptr<QNode>  node, std::shared_ptr<QNode> parent_node, T & func_class, Args&& ... func_args)
    {
        //static_assert(std::is_base_of<TraversalInterface, T>::value, "bad Traversal type");
        int iNodeType = node->getNodeType();

        if (NODE_UNDEFINED == iNodeType)
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }

        /*
        * Check node type
        */
        if (GATE_NODE == iNodeType)
        {
            auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(node);

            if (!gate_node)
            {
                QCERR("Unknown internal error");
                throw std::runtime_error("Unknown internal error");
            }
           func_class.execute(gate_node, parent_node, std::forward<Args>(func_args)...);
        }
        else if (CIRCUIT_NODE == iNodeType)
        {
            auto qcircuit_node = std::dynamic_pointer_cast<AbstractQuantumCircuit>(node);

            if (!qcircuit_node)
            {
                QCERR("Unknown internal error");
                throw std::runtime_error("Unknown internal error");
            }
            func_class.execute(qcircuit_node, parent_node, std::forward<Args>(func_args)...);
        }
        else if (PROG_NODE == iNodeType)
        {
            auto qprog_node = std::dynamic_pointer_cast<AbstractQuantumProgram>(node);

            if (!qprog_node)
            {
                QCERR("Unknown internal error");
                throw std::runtime_error("Unknown internal error");
            }
            func_class.execute(qprog_node, parent_node, std::forward<Args>(func_args)...);
        }
        else if ((WHILE_START_NODE == iNodeType) || (QIF_START_NODE == iNodeType))
        {
            auto control_flow_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(node);

            if (!control_flow_node)
            {
                QCERR("Unknown internal error");
                throw std::runtime_error("Unknown internal error");
            }
            func_class.execute(control_flow_node, parent_node, std::forward<Args>(func_args)...);
        }
        else if (MEASURE_GATE == iNodeType)
        {
            auto measure_node = std::dynamic_pointer_cast<AbstractQuantumMeasure >(node);

            if (!measure_node)
            {
                QCERR("Unknown internal error");
                throw std::runtime_error("Unknown internal error");
            }
            func_class.execute(measure_node, parent_node, std::forward<Args>(func_args)...);
        }
		else if (RESET_NODE == iNodeType)
		{
			auto reset_node = std::dynamic_pointer_cast<AbstractQuantumReset>(node);

			if (!reset_node)
			{
				QCERR("Unknown internal error");
				throw std::runtime_error("Unknown internal error");
			}
			func_class.execute(reset_node, parent_node, std::forward<Args>(func_args)...);
		}
        else if (CLASS_COND_NODE == iNodeType)
        {
            auto classical_node = std::dynamic_pointer_cast<AbstractClassicalProg>(node);

            if (!classical_node)
            {
                QCERR("Unknown internal error");
                throw std::runtime_error("Unknown internal error");
            }
            func_class.execute(classical_node, parent_node, std::forward<Args>(func_args)...);
        }
        else
        {
            QCERR("iNodeType error");
            throw std::runtime_error("iNodeType error");
        }
    }
};




/**
* @class TraversalInterface
* @brief All objects that want to use the class Traversal need to integrate this class
* @ingroup Utilities
*/
template<typename... Args >
class TraversalInterface
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  QNode*  parent Node
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, Args&& ... func_args) {};
    
    /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  QNode*  parent Node
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, Args&& ... func_args) {};

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  QNode*  parent Node
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, Args&& ... func_args) {};

    /*!
    * @brief  Execution traversal control flow node
    * @param[in,out]  AbstractControlFlowNode*  control flow node
    * @param[in]  QNode*  parent Node
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, Args&& ... func_args) 
    {
        Traversal::traversal(cur_node, *this, std::forward<Args>(func_args)...);
    }


    /*!
    * @brief  Execution traversal qcircuit
    * @param[in,out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  QNode*  parent Node
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, Args&& ... func_args)
    {
        Traversal::traversal(cur_node,false, *this, std::forward<Args>(func_args)...);
    }

    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractQuantumProgram*  quantum prog
    * @param[in]  QNode*  parent Node
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, Args&& ... func_args)
    {
         Traversal::traversal(cur_node, *this, std::forward<Args>(func_args)...);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractClassicalProg*  classical prog
    * @param[in]  QNode*  parent Node
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
        std::shared_ptr<QNode> parent_node,
        Args&& ... func_args) {};
};


QPANDA_END
#endif // !_TRAVERSAL_H
