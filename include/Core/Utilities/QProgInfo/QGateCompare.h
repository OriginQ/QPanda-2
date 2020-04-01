/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGateCompare.h
Author: Wangjing
Updated in 2019/04/09 15:05

Classes for QGateCompare.

*/
/*! \file QGateCompare.h */
#ifndef  QGATE_COMPARE_H_
#define  QGATE_COMPARE_H_

#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include <map>
#include <type_traits>
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/QProgInfo/QGateCounter.h"

QPANDA_BEGIN

/**
* @class QGateCompare
* @ingroup Utilities
* @brief Qunatum Gate Compare
*/
class QGateCompare : public TraversalInterface<>
{
public:
    QGateCompare(const std::vector<std::vector<std::string>> &);

    /**
    * @brief  traversal quantum program, quantum circuit, quantum while or quantum if
    * @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
    * @return     void
    * @note
    */
    template <typename _Ty>
    void traversal(_Ty node)
    {
	    execute(node.getImplementationPtr(),nullptr);
    }

	template <typename _Ty>
	void traversal(std::shared_ptr<_Ty> node)
	{
		
		Traversal::traversalByType(std::dynamic_pointer_cast<QNode>(node), nullptr, *this);
	}


    /**
    * @brief  get unsupported gate numner
    * @return     size_t     Unsupported QGate number
    * @note
    */
    size_t count();
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);

/*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

    /*!
    * @brief  Execution traversal control flow node
    * @param[in,out]  AbstractControlFlowNode*  control flow node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node) 
    {
        Traversal::traversal(cur_node,*this);
    }


    /*!
    * @brief  Execution traversal qcircuit
    * @param[in,out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
    {
        Traversal::traversal(cur_node,false,*this);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractQuantumProgram*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
    {
        Traversal::traversal(cur_node,*this);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractClassicalProg*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
        std::shared_ptr<QNode> parent_node)
        {}
private:

    size_t m_count;
    std::vector<std::vector<std::string>> m_gates;
};

/*will delete*/
template <typename _Ty>
size_t getUnSupportQGateNumber(_Ty node, const std::vector<std::vector<std::string>> &gates)
{
    QGateCompare compare(gates);
    compare.traversal(node);
    return compare.count();
}

/* new interface */

/**
* @brief  Count quantum program unsupported gate numner
* @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
* @param[in]  const std::vector<std::vector<std::string>>&    support gates
* @return     size_t     Unsupported QGate number
* @note
*/
template <typename _Ty>
size_t getUnsupportQGateNum(_Ty node, const std::vector<std::vector<std::string>> &gates)
{
	QGateCompare compare(gates);
	compare.traversal(node);
	return compare.count();
}


QPANDA_END


#endif
