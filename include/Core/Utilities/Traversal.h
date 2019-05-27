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
#include "Core/QuantumCircuit/ClassicalProgram.h"
QPANDA_BEGIN


/**
* @class TraversalInterface
* @brief All objects that want to use the class Traversal need to integrate this class
* @ingroup Utilities
*/
class TraversalInterface
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in|out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(AbstractQGateNode * cur_node, QNode * parent_node) {};
    
    /*!
    * @brief  Execution traversal measure node
    * @param[in|out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(AbstractQuantumMeasure * cur_node, QNode * parent_node) {};

    /*!
    * @brief  Execution traversal control flow node
    * @param[in|out]  AbstractControlFlowNode*  control flow node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(AbstractControlFlowNode * cur_node, QNode * parent_node);

    /*!
    * @brief  Execution traversal qcircuit
    * @param[in|out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(AbstractQuantumCircuit * cur_node, QNode * parent_node);

    /*!
    * @brief  Execution traversal qprog
    * @param[in|out]  AbstractQuantumCircuit*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(AbstractQuantumProgram * cur_node, QNode * parent_node);
    /*!
    * @brief  Execution traversal qprog
    * @param[in|out]  AbstractQuantumCircuit*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    virtual void execute(AbstractClassicalProg * cur_node, QNode * parent_node) {};
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
    * @exception  invalid_argument runtime_error
    * @note
    */
    static void traversal(AbstractControlFlowNode * pQCircuit, TraversalInterface * tarversal_object);

    /*!
    * @brief  Traversing qcircuit
    * @param[in]  AbstractQuantumCircuit*  QCircuit nodes that need to be traversed
    * @param[in]  TraversalInterface*  The method object needed for traversal
    * @param[in]  bool  Whether the quantum circuit needs to be transposed
    * @return     void
    * @exception  invalid_argument runtime_error
    * @note
    */
    static void traversal(AbstractQuantumCircuit *, TraversalInterface * tarversal_object,bool isdagger);


    /*!
    * @brief  Traversing qprog
    * @param[in]  AbstractQuantumProgram*  QProg nodes that need to be traversed
    * @param[in]  TraversalInterface*  The method object needed for traversal
    * @return     void
    * @exception  invalid_argument runtime_error
    * @note
    */
    static void traversal(AbstractQuantumProgram *, TraversalInterface * tarversal_object);

    /*!
    * @brief  traversalByType
    * @param[in]  QNode*  nodes that need to be traversed
    * @param[in]  parent_node*  nodes that need to be traversed
    * @param[in]  TraversalInterface*  The method object needed for traversal
    * @return     void
    * @exception  invalid_argument runtime_error
    * @note
    */
    static void traversalByType(QNode * node, QNode * parent_node, TraversalInterface * tarversal_object);
};
QPANDA_END
#endif // !_TRAVERSAL_H
