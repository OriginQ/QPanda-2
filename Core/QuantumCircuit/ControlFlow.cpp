/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ControlFlow.cpp 
Author: Menghan.Dou
Created in 2018-6-30

Classes for ControlFlow

Update@2018-8-30
    Update by code specification
*/

#include "ControlFlow.h"
#include "Utilities/ConfigMap.h"
USING_QPANDA
using namespace std;
QWhileProg  QPanda::CreateWhileProg(ClassicalCondition  classical_condition, QProg true_node)
{
    QWhileProg qwhile(classical_condition, true_node);
    return qwhile;
}

QIfProg  QPanda::CreateIfProg(ClassicalCondition  classical_condition, QProg true_node)
{
    QIfProg qif(classical_condition, true_node);
    return qif;
}

QIfProg  QPanda::CreateIfProg(ClassicalCondition classical_condition, QProg true_node, QProg false_node)
{
    QIfProg qif(classical_condition, true_node, false_node);
    return qif;
}

QWhileProg::~QWhileProg()
{
    m_control_flow.reset();
}

QWhileProg::QWhileProg(const QWhileProg &old_qwhile)
{
    m_control_flow = old_qwhile.m_control_flow;
}

QWhileProg::QWhileProg(ClassicalCondition classical_condition, QProg node)
{
    auto class_name = ConfigMap::getInstance()["QWhileProg"];
    auto qwhile = QWhileFactory::getInstance()
                    .getQWhile(class_name, classical_condition, &node);

    m_control_flow.reset(qwhile);
    
}

std::shared_ptr<QNode> QWhileProg::getImplementationPtr()
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return dynamic_pointer_cast<QNode>(m_control_flow);
}

NodeType QWhileProg::getNodeType() const
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return dynamic_pointer_cast<QNode > (m_control_flow)->getNodeType();
}

QNode * QWhileProg::getTrueBranch() const
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_control_flow->getTrueBranch();
}

QNode * QWhileProg::getFalseBranch() const
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_control_flow->getFalseBranch();
}


ClassicalCondition * QWhileProg::getCExpr() 
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_control_flow->getCExpr();
}

QIfProg::~QIfProg()
{
    m_control_flow.reset();
}

QIfProg::QIfProg(const QIfProg &old_qif)
{
    m_control_flow = old_qif.m_control_flow;
}

QIfProg::QIfProg(ClassicalCondition classical_condition, QProg true_node, QProg false_node)
{
    auto sClasNname = ConfigMap::getInstance()["QIfProg"];
    auto qif = QIfFactory::getInstance()
               .getQIf(sClasNname, classical_condition, &true_node, &false_node);
    m_control_flow.reset(qif);
}

QIfProg::QIfProg(ClassicalCondition classical_condition, QProg node)
{
    auto sClasNname = ConfigMap::getInstance()["QIfProg"];
    auto qif = QIfFactory::getInstance().getQIf(sClasNname, classical_condition, &node);
    m_control_flow.reset(qif);
}

NodeType QIfProg::getNodeType() const
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return dynamic_pointer_cast<QNode >(m_control_flow)->getNodeType();
}

QNode * QIfProg::getTrueBranch() const
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_control_flow->getTrueBranch();
}

QNode * QIfProg::getFalseBranch() const
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_control_flow->getFalseBranch();
}

ClassicalCondition * QIfProg::getCExpr() 
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_control_flow->getCExpr();
}

std::shared_ptr<QNode> QIfProg::getImplementationPtr()
{
    if (!m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return dynamic_pointer_cast<QNode>(m_control_flow);
}

OriginQIf::~OriginQIf()
{
    if (nullptr != m_true_item)
    {
        delete m_true_item;
        m_true_item = nullptr;
    }

    if (nullptr != m_false_item)
    {
        delete m_false_item;
        m_false_item = nullptr;
    }

}

OriginQIf::OriginQIf(ClassicalCondition classical_condition,
                     QProg true_node,
                     QProg false_node)
                     :m_classical_condition(classical_condition)
{

    auto true_shared_ptr = true_node.getImplementationPtr();
    m_true_item = new OriginItem();
    m_true_item->setNode(true_shared_ptr);

    auto false_shared_ptr = false_node.getImplementationPtr();
    m_false_item = new OriginItem();
    m_false_item->setNode(false_shared_ptr);
}

OriginQIf::OriginQIf(ClassicalCondition classical_condition, QProg node)
                    :m_classical_condition(classical_condition)
{
    auto node_shared_ptr = node.getImplementationPtr();
    m_true_item = new OriginItem();
    m_true_item->setNode(node_shared_ptr);
}

NodeType OriginQIf::getNodeType() const
{
    return m_node_type;
}

QNode * OriginQIf::getTrueBranch() const
{
    if (nullptr != m_true_item)
        return m_true_item->getNode().get();
    else
        return nullptr;
}

QNode * OriginQIf::getFalseBranch() const
{
    if(nullptr != m_false_item)
        return m_false_item->getNode().get();
    return nullptr;
}

void OriginQIf::setTrueBranch(QProg node)
{
    if (nullptr != m_true_item)
    {
        delete(m_true_item);
        m_true_item = nullptr;
        Item * temp = new OriginItem();
        auto node_shared_ptr = node.getImplementationPtr();
        temp->setNode(node_shared_ptr);
        m_true_item = temp;
    }
       
}

void OriginQIf::setFalseBranch(QProg node)
{
    if (nullptr != m_false_item)
    {
        delete(m_false_item);
        m_false_item = nullptr;
        Item * temp = new OriginItem();
        auto node_shared_ptr = node.getImplementationPtr();
        temp->setNode(node_shared_ptr);
        m_false_item = temp;
    }
}

ClassicalCondition * OriginQIf::getCExpr() 
{
    return &m_classical_condition;
}

void OriginQIf::execute(QPUImpl * quantum_gates, QuantumGateParam * param)
{
    auto aCExpr = getCExpr();
    QNode * node;
    if (aCExpr->eval())
    {
        node = getTrueBranch();
        if (nullptr == node)
        {
            return ;
        }
    }
    else
    {
        node = getFalseBranch();
        if (nullptr == node)
        {
            return ;
        }
    }

    node->execute(quantum_gates, param);

}

void QIfFactory::registClass(string name, CreateQIfTrueFalse_cb method)
{
    if((name.size() > 0) &&(nullptr != method))
        m_qif_true_false_map.insert(pair<string, CreateQIfTrueFalse_cb>(name, method));
    else
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
}

void QIfFactory::registClass(string name, CreateQIfTrueOnly_cb method)
{
    if ((name.size() > 0) && (nullptr != method))
        m_qif_true_only_map.insert(pair<string, CreateQIfTrueOnly_cb>(name, method));
    else
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
}

AbstractControlFlowNode * QIfFactory::getQIf(std::string & class_name, 
                                             ClassicalCondition & classical_condition,
                                             QNode * true_node,
                                             QNode * false_node)
{
    auto aiter = m_qif_true_false_map.find(class_name);
    if (aiter != m_qif_true_false_map.end())
    {
        return aiter->second(classical_condition, true_node, false_node);
    }
    else
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
}

AbstractControlFlowNode * QIfFactory::getQIf(std::string & class_name,
                                             ClassicalCondition & classical_condition,
                                             QNode * true_node)
{
    auto aiter = m_qif_true_only_map.find(class_name);
    if (aiter != m_qif_true_only_map.end())
    {
        return aiter->second(classical_condition, true_node);
    }
    else
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
}

QIF_REGISTER(OriginQIf);

OriginQWhile::~OriginQWhile()
{
    if (nullptr != m_true_item)
    {
        delete m_true_item;
        m_true_item = nullptr;
    }

}

OriginQWhile::OriginQWhile(ClassicalCondition classical_condition, QProg node)
                          : m_node_type(WHILE_START_NODE),
                            m_classical_condition(classical_condition)
{
    auto node_shared_ptr = node.getImplementationPtr();
    m_true_item = new OriginItem();
    m_true_item->setNode(node_shared_ptr);
}

NodeType OriginQWhile::getNodeType() const
{
    return m_node_type;
}

QNode * OriginQWhile::getTrueBranch() const
{
    if (nullptr != m_true_item)
        return m_true_item->getNode().get();
    return nullptr;
}

QNode * OriginQWhile::getFalseBranch() const
{
    QCERR("error");
    throw runtime_error("error");
}

void OriginQWhile::setTrueBranch(QProg node)
{
    if (nullptr != m_true_item)
    {
        delete(m_true_item);
        m_true_item = nullptr;

        Item * temp = new OriginItem();
        auto node_shared_ptr = node.getImplementationPtr();
        temp->setNode(node_shared_ptr);

        m_true_item = temp;
    }
}

ClassicalCondition * OriginQWhile::getCExpr() 
{
    return &m_classical_condition;
}

void OriginQWhile::execute(QPUImpl * quantum_gates, QuantumGateParam * param)
{
    auto aCExpr = getCExpr();
    
    while (aCExpr->eval())
    {
        auto node = getTrueBranch();
        if (node == nullptr)
        {
            return ;
        }
        
        node->execute(quantum_gates, param);
    }
}

QWHILE_REGISTER(OriginQWhile);

void QWhileFactory::registClass(string name, CreateQWhile_cb method)
{
    if (name.size() <= 0)
    {
        QCERR("name is empty string");
        throw invalid_argument("name is empty string");
    }
    if (nullptr == method)
    {
        QCERR("method is a nullptr");
        throw invalid_argument("method is a nullptr");
    }

    m_qwhile_map.insert(pair<string, CreateQWhile_cb>(name, method));
}

AbstractControlFlowNode *QWhileFactory::getQWhile(std::string & class_name,
                                                   ClassicalCondition & classical_condition,
                                                   QNode * true_node)
{
    if (class_name.size() <= 0)
    {
        QCERR("class_name is empty string");
        throw invalid_argument("class_name is empty string");
    }
    if (nullptr == true_node)
    {
        QCERR("true_node is a nullptr");
        throw invalid_argument("true_node is a nullptr");
    }
    auto aiter = m_qwhile_map.find(class_name);
    if (aiter != m_qwhile_map.end())
    {
        return aiter->second(classical_condition, true_node);
    }
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}
