/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Traversal.h
Author: doumenghan
Created in 2019-4-16

Classes for get the shortes path of graph

*/
#include "Traversal.h"
USING_QPANDA
using namespace std;

void Traversal::traversal(AbstractControlFlowNode * control_flow_node, TraversalInterface * tarversal_object)
{
    if (nullptr == control_flow_node)
    {
        QCERR("control_flow_node is nullptr");
        throw invalid_argument("control_flow_node is nullptr");
    }

    auto pNode = dynamic_cast<QNode *>(control_flow_node);

    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    auto iNodeType = pNode->getNodeType();

    if (WHILE_START_NODE == iNodeType)
    {
        traversalByType(control_flow_node->getTrueBranch(), pNode, tarversal_object);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        traversalByType(control_flow_node->getTrueBranch(), pNode, tarversal_object);
        auto false_branch_node = control_flow_node->getFalseBranch();

        if (nullptr != false_branch_node)
        {
            traversalByType(false_branch_node, pNode, tarversal_object);
        }
    }
}

void Traversal::traversal(AbstractQuantumCircuit * qcircuit_node, 
                          TraversalInterface * tarversal_object, 
                          bool isdagger)
{
    if (nullptr == qcircuit_node)
    {
        QCERR("pQCircuit is nullptr");
        throw invalid_argument("pQCircuit is nullptr");
    }

    auto aiter = qcircuit_node->getFirstNodeIter();

    if (aiter == qcircuit_node->getEndNodeIter())
        return;

    auto pNode = dynamic_cast<QNode *>(qcircuit_node);

    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    if (isdagger)
    {
        auto aiter = qcircuit_node->getLastNodeIter();
        if (nullptr == *aiter)
        {
            return;
        }
        while (aiter != qcircuit_node->getHeadNodeIter())
        {
            auto next = --aiter;
            traversalByType((*aiter).get(), pNode, tarversal_object);
            aiter = next;
        }

    }
    else
    {
        auto aiter = qcircuit_node->getFirstNodeIter();

        if (aiter == qcircuit_node->getLastNodeIter())
            return;
        while (aiter != qcircuit_node->getEndNodeIter())
        {
            auto next = aiter.getNextIter();
            traversalByType((*aiter).get(), pNode, tarversal_object);
            aiter = next;
        }
    }

}

void Traversal::traversal(AbstractQuantumProgram *qprog_node,
                          TraversalInterface * tarversal_object)
{
    if (nullptr == qprog_node)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    auto aiter = qprog_node->getFirstNodeIter();

    if (aiter == qprog_node->getLastNodeIter())
        return;
    auto pNode = dynamic_cast<QNode *>(qprog_node);

    if (nullptr == pNode)
    {
        QCERR("pNode is nullptr");
        throw invalid_argument("pNode is nullptr");
    }

    while (aiter != qprog_node->getEndNodeIter())
    {
        auto next = aiter.getNextIter();
        traversalByType((*aiter).get(), pNode,tarversal_object);
        aiter = next;
    }
}

void Traversal::traversalByType(QNode * node, QNode * parent_node, TraversalInterface * tarversal_object)
{
    int iNodeType = node->getNodeType();

    if (NODE_UNDEFINED == iNodeType)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    /*
    * Check node type
    */
    if (GATE_NODE == iNodeType)
    {
        auto gate_node = dynamic_cast<AbstractQGateNode *>(node);

        if (nullptr == gate_node)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        tarversal_object->execute(gate_node, parent_node);           
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        auto qcircuit_node = dynamic_cast<AbstractQuantumCircuit *>(node);

        if (nullptr == qcircuit_node)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        tarversal_object->execute(qcircuit_node, parent_node);          
    }
    else if (PROG_NODE == iNodeType)
    {
        auto qprog_node = dynamic_cast<AbstractQuantumProgram *>(node);

        if (nullptr == qprog_node)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        tarversal_object->execute(qprog_node, parent_node);           
    }
    else if ((WHILE_START_NODE == iNodeType) || (QIF_START_NODE == iNodeType))
    {
        auto control_flow_node = dynamic_cast<AbstractControlFlowNode *>(node);

        if (nullptr == control_flow_node)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        tarversal_object->execute(control_flow_node, parent_node);            
    }
    else if (MEASURE_GATE == iNodeType)
    {
        auto measure_node = dynamic_cast<AbstractQuantumMeasure *>(node);

        if (nullptr == measure_node)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        tarversal_object->execute(measure_node, parent_node);
    }
    else if (CLASS_COND_NODE == iNodeType)
    {
        auto classical_node= dynamic_cast<AbstractClassicalProg *>(node);

        if (nullptr == classical_node)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        tarversal_object->execute(classical_node, parent_node);

        return;
    }
    else
    {
        QCERR("iNodeType error");
        throw runtime_error("iNodeType error");
    }
}

void TraversalInterface::execute(AbstractControlFlowNode * cur_node, QNode * parent_node)
{
    Traversal::traversal(cur_node, this);
}

void TraversalInterface::execute(AbstractQuantumCircuit * cur_node, QNode * parent_node)
{
    Traversal::traversal(cur_node, this ,false);
}

void TraversalInterface::execute(AbstractQuantumProgram * cur_node, QNode * parent_node)
{
    Traversal::traversal(cur_node, this);
}
