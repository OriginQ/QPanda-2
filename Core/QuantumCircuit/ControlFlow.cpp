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
QWhileProg  QPanda::CreateWhileProg(ClassicalCondition  classical_condition, QNode * true_node)
{
    if (nullptr == true_node)
    {
        QCERR("CreateWhileProg parameter invalid");
        throw invalid_argument("CreateWhileProg parameter invalid");
    }

    QWhileProg qwhile(classical_condition, true_node);
    return qwhile;
}

QIfProg  QPanda::CreateIfProg(ClassicalCondition  classical_condition, QNode * true_node)
{
    
    if (nullptr == true_node)
    {
        QCERR("CreateIfProg parameter invalid");
        throw invalid_argument("CreateIfProg parameter invalid");
    }

    QIfProg qif(classical_condition, true_node);
    return qif;
}

QIfProg  QPanda::CreateIfProg(ClassicalCondition classical_condition, QNode * true_node, QNode * false_node)
{
    if (nullptr == true_node)
    {
        QCERR("CreateIfProg parameter invalid");
        throw invalid_argument("CreateIfProg parameter invalid");
    }

    if (nullptr == false_node)
    {
        QCERR("CreateIfProg parameter invalid");
        throw invalid_argument("CreateIfProg parameter invalid");
    }
    QIfProg qif(classical_condition, true_node, false_node);
    return qif;
}

QWhileProg::~QWhileProg()
{
    QNodeMap::getInstance().deleteNode(m_position);
}

QWhileProg::QWhileProg(const QWhileProg &old_qwhile)
{
    m_position = old_qwhile.getPosition();
    auto aiter = QNodeMap::getInstance().getNode(m_position);
    if (aiter != nullptr)
        m_control_flow = dynamic_cast<AbstractControlFlowNode *>(aiter);
    else
    {
        QCERR("QWhileProg cant found");
        throw invalid_argument("QWhileProg cant found");
    }

    
    if (!QNodeMap::getInstance().addNodeRefer(m_position))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
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
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_control_flow = qwhile;
    
}

NodeType QWhileProg::getNodeType() const
{
    if (nullptr == m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return dynamic_cast<QNode *> (m_control_flow)->getNodeType();
}

QNode * QWhileProg::getTrueBranch() const
{
    if (nullptr == m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_control_flow->getTrueBranch();
}

QNode * QWhileProg::getFalseBranch() const
{
    if (nullptr == m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_control_flow->getFalseBranch();
}


ClassicalCondition * QWhileProg::getCExpr() 
{
    if (nullptr == m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

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
    {
        QCERR("QIfProg can't be found");
        throw invalid_argument("QIfProg can't be found");
    }

    if (!QNodeMap::getInstance().addNodeRefer(m_position))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

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
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

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
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

}

NodeType QIfProg::getNodeType() const
{
    if (nullptr == m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return dynamic_cast<QNode *>(m_control_flow)->getNodeType();
}

QNode * QIfProg::getTrueBranch() const
{
    if (nullptr == m_control_flow)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

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
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
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
    else m_true_item = nullptr;
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
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
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
    {
        QCERR("node is a nullptr");
        throw invalid_argument("node is a nullptr");
    }

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
    QCERR("error");
    throw runtime_error("error");
}

void OriginQWhile::setTrueBranch(QNode * node)
{
    if (nullptr == node)
    {
        QCERR("node is a nullptr");
        throw invalid_argument("node is a nullptr");
    }

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
