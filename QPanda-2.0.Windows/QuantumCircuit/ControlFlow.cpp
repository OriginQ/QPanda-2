/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

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

#include "ControlFlow.h"

QuantumWhile  CreateWhileProg(ClassicalCondition & ccCon, QNode * trueNode)
{
    if (nullptr == trueNode)
        throw exception();
    QuantumWhile quantumWhile(ccCon, trueNode);
    return quantumWhile;
}

QuantumIf  CreateIfProg(ClassicalCondition & ccCon, QNode * trueNode)
{
    if (nullptr == trueNode)
        throw exception();
    QuantumIf quantumIf(ccCon, trueNode);
    return quantumIf;
}

QuantumIf  CreateIfProg(ClassicalCondition &ccCon, QNode * trueNode, QNode * falseNode)
{
    if (nullptr == trueNode)
        throw exception();
    if (nullptr == falseNode)
        throw exception();
    QuantumIf quantumIf(ccCon, trueNode, falseNode);
    return quantumIf;
}

QuantumWhile::QuantumWhile(QuantumWhile &oldQWhile)
{
    m_iPosition = oldQWhile.getPosition();
    auto aiter = _G_QNodeVector.getNode(m_iPosition);
    if (aiter != _G_QNodeVector.getEnd())
        m_pControlFlow = dynamic_cast<AbstractControlFlowNode *>(*aiter);
    else
        throw exception();
}

QuantumWhile::QuantumWhile(ClassicalCondition & ccCon, QNode * node)
{
    string sClasNname = "OriginWhile";
    auto aMeasure = QuantunIfFactory::getInstance().getQuantumIf(sClasNname, ccCon, node);
    _G_QNodeVector.pushBackNode(dynamic_cast<QNode *>(aMeasure));
    m_iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    m_pControlFlow = aMeasure;
}

NodeType QuantumWhile::getNodeType() const
{
    if (nullptr == m_pControlFlow)
        throw exception();
    return dynamic_cast<QNode *> (m_pControlFlow)->getNodeType();
}



QNode * QuantumWhile::getTrueBranch() const
{
    if (nullptr == m_pControlFlow)
        throw exception();
    return m_pControlFlow->getTrueBranch();
}

QNode * QuantumWhile::getFalseBranch() const
{
    if (nullptr == m_pControlFlow)
        throw exception();
    return m_pControlFlow->getFalseBranch();
}


ClassicalCondition * QuantumWhile::getCExpr() 
{
    if (nullptr == m_pControlFlow)
        throw exception();
    return m_pControlFlow->getCExpr();
}


int QuantumWhile::getPosition() const
{
    return m_iPosition;
}

QuantumIf::QuantumIf(const QuantumIf &oldQIf)
{
    m_iPosition = oldQIf.getPosition();
    auto aiter = _G_QNodeVector.getNode(m_iPosition);
    if (aiter != _G_QNodeVector.getEnd())
        m_pControlFlow = dynamic_cast<AbstractControlFlowNode *>(*aiter);
    else
        throw exception();
}

QuantumIf::QuantumIf(ClassicalCondition & ccCon, QNode * pTrueNode, QNode * pFalseNode)
{
    string sClasNname = "OriginIf";
    auto aMeasure = QuantunIfFactory::getInstance().getQuantumIf(sClasNname, ccCon, pTrueNode, pFalseNode);
    _G_QNodeVector.pushBackNode(dynamic_cast<QNode *>(aMeasure));
    m_iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    m_pControlFlow = aMeasure;
}

QuantumIf::QuantumIf(ClassicalCondition & ccCon, QNode * node)
{
    string sClasNname = "OriginIf";
    auto aMeasure = QuantunIfFactory::getInstance().getQuantumIf(sClasNname, ccCon, node);
    _G_QNodeVector.pushBackNode(dynamic_cast<QNode *>(aMeasure));
    m_iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    m_pControlFlow = aMeasure;
}

NodeType QuantumIf::getNodeType() const
{
    if (nullptr == m_pControlFlow)
        throw  exception();
    return dynamic_cast<QNode *>(m_pControlFlow)->getNodeType();
}


QNode * QuantumIf::getTrueBranch() const
{
    if (nullptr == m_pControlFlow)
        throw  exception();
    return m_pControlFlow->getTrueBranch();
}

QNode * QuantumIf::getFalseBranch() const
{

    return m_pControlFlow->getFalseBranch();
}

int QuantumIf::getPosition() const
{
    return m_iPosition;
}

ClassicalCondition * QuantumIf::getCExpr() 
{
    if (nullptr == m_pControlFlow)
        throw  exception();
    return m_pControlFlow->getCExpr();
}

OriginIf::OriginIf(ClassicalCondition & ccCon, QNode * pTrueNode, QNode * pFalseNode):m_iNodeType(QIF_START_NODE),m_CCondition(ccCon)
{
    if (nullptr != pTrueNode)
        iTrueNum = pTrueNode->getPosition();
    else
        iTrueNum = -1;

    if (nullptr != pFalseNode)
        iFalseNum = pFalseNode->getPosition();
    else
        iFalseNum = -1;
}

OriginIf::OriginIf(ClassicalCondition & ccCon, QNode * node):m_iNodeType(QIF_START_NODE),m_CCondition(ccCon)
{
    if (nullptr != node)
        iTrueNum = node->getPosition();
    else
        iTrueNum = -1;
    iFalseNum = -1;
}

NodeType OriginIf::getNodeType() const
{
    return m_iNodeType;
}

QNode * OriginIf::getTrueBranch() const
{
    auto aiter = _G_QNodeVector.getNode(iTrueNum);
    if (aiter != _G_QNodeVector.getEnd())
    {
        return *aiter;
    }
    else
    {
        return nullptr;
    }
}

QNode * OriginIf::getFalseBranch() const
{
    auto aiter = _G_QNodeVector.getNode(iFalseNum);
    if (aiter != _G_QNodeVector.getEnd())
    {
        return *aiter;
    }
    else
    {
        return nullptr;
    }
}

int OriginIf::getPosition() const
{
    throw exception();
}

ClassicalCondition * OriginIf::getCExpr() 
{
    return &m_CCondition;
}

void QuantunIfFactory::registClass(string name, CreateIfDoubleB method)
{
    if((name.size() > 0) &&(nullptr != method))
        m_QIfDoubleMap.insert(pair<string, CreateIfDoubleB>(name, method));
    else
        throw exception();
}

void QuantunIfFactory::registClass(string name, CreateIfSingleB method)
{
    if ((name.size() > 0) && (nullptr != method))
        m_QIfSingleMap.insert(pair<string, CreateIfSingleB>(name, method));
    else
        throw exception();
}

AbstractControlFlowNode * QuantunIfFactory::getQuantumIf(std::string & classname, ClassicalCondition & ccCon, QNode * pTrueNode, QNode * pFalseNode)
{
    auto aiter = m_QIfDoubleMap.find(classname);
    if (aiter != m_QIfDoubleMap.end())
    {
        return aiter->second(ccCon, pTrueNode, pFalseNode);
    }
    else
    {
        throw exception();
    }
}

AbstractControlFlowNode * QuantunIfFactory::getQuantumIf(std::string & classname, ClassicalCondition & ccCon, QNode * pTrueNode)
{
    auto aiter = m_QIfSingleMap.find(classname);
    if (aiter != m_QIfSingleMap.end())
    {
        return aiter->second(ccCon, pTrueNode);
    }
    else
    {
        throw exception();
    }
}

REGISTER_QIF(OriginIf);

OriginWhile::OriginWhile(ClassicalCondition & ccCon, QNode * node): m_iNodeType(WHILE_START_NODE),m_CCondition(ccCon)
{
    if (nullptr == node)
    {
        iTrueNum = -1;
    }
    else
    {
        iTrueNum = node->getPosition();
    }

}

NodeType OriginWhile::getNodeType() const
{
    return m_iNodeType;
}

QNode * OriginWhile::getTrueBranch() const
{
    auto aiter = _G_QNodeVector.getNode(iTrueNum);
    if (aiter != _G_QNodeVector.getEnd())
    {
        return *aiter;
    }
    else
    {
        return nullptr;
    }
}

QNode * OriginWhile::getFalseBranch() const
{
    throw exception();
}

ClassicalCondition * OriginWhile::getCExpr() 
{
    return &m_CCondition;
}

int OriginWhile::getPosition() const
{
    throw exception();
}

REGISTER_QWHILE(OriginWhile);

void QuantunWhileFactory::registClass(string name, CreateWhile method)
{
    if ((name.size() <= 0) || (nullptr == method))
    {
        throw exception();
    }

    m_QWhileMap.insert(pair<string, CreateWhile>(name, method));
}

AbstractControlFlowNode * QuantunWhileFactory::getQuantumWhile(std::string & className, ClassicalCondition & ccCon, QNode * pTrueNode)
{
    if ((className.size() <= 0) || (nullptr == pTrueNode))
    {
        throw exception();
    }
    auto aiter = m_QWhileMap.find(className);
    if (aiter != m_QWhileMap.end())
    {
        return aiter->second(ccCon, pTrueNode);
    }
    else
    {
        throw exception();
    }
}
