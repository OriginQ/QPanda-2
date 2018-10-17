/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ControlFlow.cpp 
Author: Menghan.Dou
Created in 2018-6-30

Classes for ControlFlow

Update@2018-8-30
    Update by code specification
*/

#include "ControlFlow.h"
#include "QPanda/QPandaException.h"
#include "QPanda/ConfigMap.h"

QWhileProg  CreateWhileProg(ClassicalCondition & classical_condition, QNode * true_node)
{
    if (nullptr == true_node)
        throw param_error_exception("CreateWhileProg param err", false);
    QWhileProg qwhile(classical_condition, true_node);
    return qwhile;
}

QIfProg  CreateIfProg(ClassicalCondition & classical_condition, QNode * true_node)
{
    
    if (nullptr == true_node)
        throw param_error_exception("CreateIfProg param err", false);
    QIfProg qif(classical_condition, true_node);
    return qif;
}

QIfProg  CreateIfProg(ClassicalCondition &classical_condition, QNode * true_node, QNode * false_node)
{
    if (nullptr == true_node)
        throw param_error_exception("CreateIfProg true_node param err", false);
    if (nullptr == false_node)
        throw param_error_exception("CreateIfProg false_node param err", false);
    QIfProg qif(classical_condition, true_node, false_node);
    return qif;
}

QWhileProg::~QWhileProg()
{
    QNodeMap::getInstance().deleteNode(m_position);
}

QWhileProg::QWhileProg( const QWhileProg &old_qwhile)
{
    m_position = old_qwhile.getPosition();
    auto aiter = QNodeMap::getInstance().getNode(m_position);
    if (aiter != nullptr)
        m_control_flow = dynamic_cast<AbstractControlFlowNode *>(aiter);
    else
        throw circuit_not_found_exception("QWhileProg cant found",false);
    
    if (!QNodeMap::getInstance().addNodeRefer(m_position))
        throw exception();
}

QWhileProg::QWhileProg(ClassicalCondition & classical_condition, QNode * node)
{
    auto class_name = ConfigMap::getInstance()["QWhileProg"];
    auto qwhile = QWhileFactory::getInstance()
                    .getQWhile(class_name, classical_condition, node);
    auto temp = dynamic_cast<QNode*>(qwhile);
    m_position = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_position);
    if (!QNodeMap::getInstance().addNodeRefer(m_position))
        throw exception();
    m_control_flow = qwhile;
    
}

NodeType QWhileProg::getNodeType() const
{
    if (nullptr == m_control_flow)
        throw exception();
    return dynamic_cast<QNode *> (m_control_flow)->getNodeType();
}

QNode * QWhileProg::getTrueBranch() const
{
    if (nullptr == m_control_flow)
        throw exception();
    return m_control_flow->getTrueBranch();
}

QNode * QWhileProg::getFalseBranch() const
{
    if (nullptr == m_control_flow)
        throw exception();
    return m_control_flow->getFalseBranch();
}


ClassicalCondition * QWhileProg::getCExpr() 
{
    if (nullptr == m_control_flow)
        throw exception();
    return m_control_flow->getCExpr();
}


qmap_size_t QWhileProg::getPosition() const
{
    return m_position;
}

QIfProg::~QIfProg()
{
    QNodeMap::getInstance().deleteNode(m_position);
}

QIfProg::QIfProg(const QIfProg &old_qif)
{
    m_position = old_qif.getPosition();
    auto aiter = QNodeMap::getInstance().getNode(m_position);
    if (aiter != nullptr)
        m_control_flow = dynamic_cast<AbstractControlFlowNode *>(aiter);
    else
        throw circuit_not_found_exception("QIfProg cant found", false);
    if (!QNodeMap::getInstance().addNodeRefer(m_position))
        throw exception();
}

QIfProg::QIfProg(ClassicalCondition & classical_condition, QNode * true_node, QNode * false_node)
{
    auto sClasNname = ConfigMap::getInstance()["QIfProg"];
    auto qif = QIfFactory::getInstance()
               .getQIf(sClasNname, classical_condition, true_node, false_node);
    auto temp = dynamic_cast<QNode *>(qif);
    m_position = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_position);
    m_control_flow = qif;
    if (!QNodeMap::getInstance().addNodeRefer(m_position))
        throw exception();
}

QIfProg::QIfProg(ClassicalCondition & classical_condition, QNode * node)
{
    auto sClasNname = ConfigMap::getInstance()["QIfProg"];
    auto qif = QIfFactory::getInstance().getQIf(sClasNname, classical_condition, node);
    auto temp = dynamic_cast<QNode *>(qif);
    m_position = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_position);
    m_control_flow = qif;
    if (!QNodeMap::getInstance().addNodeRefer(m_position))
        throw exception();
}

NodeType QIfProg::getNodeType() const
{
    if (nullptr == m_control_flow)
        throw  exception();
    return dynamic_cast<QNode *>(m_control_flow)->getNodeType();
}

QNode * QIfProg::getTrueBranch() const
{
    if (nullptr == m_control_flow)
        throw  exception();
    return m_control_flow->getTrueBranch();
}

QNode * QIfProg::getFalseBranch() const
{

    return m_control_flow->getFalseBranch();
}

qmap_size_t QIfProg::getPosition() const
{
    return m_position;
}

ClassicalCondition * QIfProg::getCExpr() 
{
    if (nullptr == m_control_flow)
        throw  exception();
    return m_control_flow->getCExpr();
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

OriginQIf::OriginQIf(ClassicalCondition & classical_condition,
                     QNode * true_node,
                     QNode * false_node)
                     :m_node_type(QIF_START_NODE),m_classical_condition(classical_condition)
{
    if (nullptr != true_node)
    {
        m_true_item = new OriginItem();
        m_true_item->setNode(true_node);
    }
    else
        m_true_item = nullptr;

    if (nullptr != false_node)
    {
        m_false_item = new OriginItem();
        m_false_item->setNode(false_node);
    }
    else
        m_false_item = nullptr;
}

OriginQIf::OriginQIf(ClassicalCondition & classical_condition, QNode * node)
                    :m_node_type(QIF_START_NODE),m_classical_condition(classical_condition)
{
    if (nullptr != node)
    {
        m_true_item = new OriginItem();
        m_true_item->setNode(node);
    }
    else
        m_true_item = nullptr;
    m_false_item = nullptr;
}

NodeType OriginQIf::getNodeType() const
{
    return m_node_type;
}

QNode * OriginQIf::getTrueBranch() const
{
    if (nullptr != m_true_item)
        return m_true_item->getNode();
    else
        throw exception();
}

QNode * OriginQIf::getFalseBranch() const
{
    if(nullptr != m_false_item)
        return m_false_item->getNode();
    return nullptr;
}

void OriginQIf::setTrueBranch(QNode * node)
{
    if (nullptr == node)
        throw param_error_exception("param is nullptr",false);

    if (nullptr != m_true_item)
    {
        delete(m_true_item);
        m_true_item = nullptr;
        Item * temp = new OriginItem();
        temp->setNode(node);
        m_true_item = temp;
    }
       
}

void OriginQIf::setFalseBranch(QNode * node)
{
    if (nullptr != m_false_item)
    {
        delete(m_false_item);
        m_false_item = nullptr;
        Item * temp = new OriginItem();
        temp->setNode(node);
        m_false_item = temp;
    }
}

qmap_size_t OriginQIf::getPosition() const
{
    return m_position;
}

void OriginQIf::setPosition(qmap_size_t position)
{
    m_position = position;
}

ClassicalCondition * OriginQIf::getCExpr() 
{
    return &m_classical_condition;
}

void QIfFactory::registClass(string name, CreateQIfTrueFalse_cb method)
{
    if((name.size() > 0) &&(nullptr != method))
        m_qif_true_false_map.insert(pair<string, CreateQIfTrueFalse_cb>(name, method));
    else
        throw exception();
}

void QIfFactory::registClass(string name, CreateQIfTrueOnly_cb method)
{
    if ((name.size() > 0) && (nullptr != method))
        m_qif_true_only_map.insert(pair<string, CreateQIfTrueOnly_cb>(name, method));
    else
        throw exception();
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
        throw exception();
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
        throw exception();
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

OriginQWhile::OriginQWhile(ClassicalCondition & classical_condition, QNode * node)
                          : m_node_type(WHILE_START_NODE),
                            m_classical_condition(classical_condition)
{
    if (nullptr == node)
    {
        m_true_item = nullptr;
    }
    else
    {
        m_true_item = new OriginItem();
        m_true_item->setNode(node);
    }

}

NodeType OriginQWhile::getNodeType() const
{
    return m_node_type;
}

QNode * OriginQWhile::getTrueBranch() const
{
    if (nullptr != m_true_item)
        return m_true_item->getNode();
    return nullptr;
}

QNode * OriginQWhile::getFalseBranch() const
{
    throw exception();
}

void OriginQWhile::setTrueBranch(QNode * node)
{
    if (nullptr == node)
        throw param_error_exception("param is nullptr", false);

    if (nullptr != m_true_item)
    {
        delete(m_true_item);
        m_true_item = nullptr;

        Item * temp = new OriginItem();
        temp->setNode(node);

        m_true_item = temp;
    }

}

ClassicalCondition * OriginQWhile::getCExpr() 
{
    return &m_classical_condition;
}

qmap_size_t OriginQWhile::getPosition() const
{
    return m_position;
}

void OriginQWhile::setPosition(qmap_size_t position)
{
    m_position = position;
}

QWHILE_REGISTER(OriginQWhile);

void QWhileFactory::registClass(string name, CreateQWhile_cb method)
{
    if ((name.size() <= 0) || (nullptr == method))
    {
        throw exception();
    }

    m_qwhile_map.insert(pair<string, CreateQWhile_cb>(name, method));
}

AbstractControlFlowNode * QWhileFactory::getQWhile(std::string & class_name,
                                                   ClassicalCondition & classical_condition,
                                                   QNode * true_node)
{
    if ((class_name.size() <= 0) || (nullptr == true_node))
    {
        throw exception();
    }

    auto aiter = m_qwhile_map.find(class_name);
    if (aiter != m_qwhile_map.end())
    {
        return aiter->second(classical_condition, true_node);
    }
    else
    {
        throw exception();
    }
}
