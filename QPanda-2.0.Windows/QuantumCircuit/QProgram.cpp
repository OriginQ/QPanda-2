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

#include "QProgram.h"



QuantumProgram & CreateEmptyQProg()
{
    QuantumProgram * temp = new QuantumProgram();
    _G_QNodeVector.pushBackNode(temp);
    temp->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *temp;
}


QuantumCircuit & CreateEmptyCircuit()
{
    QuantumCircuit * temp = new QuantumCircuit();
    _G_QNodeVector.pushBackNode(temp);
    temp->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *temp;
}


QuantumGate::QuantumGate(Qubit * qbit, QGate *pQGate) : m_iNodeType(GATE_NODE)
{

    m_pGate = pQGate;
    m_QuBitVector.push_back(qbit);
}

QuantumGate::QuantumGate(Qubit * targetQuBit, Qubit * controlQuBit, QGate *pQGate)
{
    m_pGate = pQGate;
    m_QuBitVector.push_back(targetQuBit);
    m_QuBitVector.push_back(controlQuBit);
    m_iNodeType = GATE_NODE;
}

NodeType QuantumGate::getNodeType() const
{
    return m_iNodeType;
}

size_t QuantumGate::getQuBitVector(vector<Qubit *>& vector) const
{
    for (auto aiter : m_QuBitVector)
    {
        vector.push_back(aiter);
    }
    return m_QuBitVector.size();
}

size_t QuantumGate::getQuBitNum() const
{
    return m_QuBitVector.size();
}

QGate * QuantumGate::getQGate() const
{
    return m_pGate;
}

GateType QuantumGate::getQGateType() const
{
    return m_iGateType;
}

int QuantumGate::getPosition() const
{
    return iPosition;
}

bool QuantumGate::setDagger(bool bIsDagger)
{
    m_bIsDagger = bIsDagger;
    return m_bIsDagger;
}

bool QuantumGate::setControl(vector<Qubit *>& quBitVector)
{
    for (auto aiter : quBitVector)
    {
        m_controlQuBitVector.push_back(aiter);
    }
    return true;
}

bool QuantumGate::isDagger() const
{
    return m_bIsDagger;
}

size_t QuantumGate::getControlVector(vector<Qubit *>& quBitVector) const
{
    for (auto aiter : m_controlQuBitVector)
    {
        quBitVector.push_back(aiter);
    }
    return quBitVector.size();
}



QuantumCircuit::~QuantumCircuit()
{
    Item *temp;
    if (m_pHead != nullptr)
    {
        while (m_pHead != m_pEnd)
        {
            temp = m_pHead;
            m_pHead = m_pHead->getNext();
            m_pHead->setPre(nullptr);
            delete temp;
        }

        delete m_pHead;
        m_pHead = nullptr;
        m_pEnd = nullptr;
    }

}

void QuantumCircuit::pushBackNode(QNode * pNode)
{
    try
    {

        if ((nullptr == m_pHead) && (nullptr == m_pEnd))
        {

            Item *iter = new OriginItem();
            iter->setNext(nullptr);
            iter->setPre(nullptr);
            iter->setNode(pNode);
            m_pHead = iter;
            m_pEnd = iter;
        }
        else
        {
            Item *iter = new OriginItem();
            iter->setNext(nullptr);
            iter->setPre(m_pEnd);
            m_pEnd->setNext(iter);
            m_pEnd = iter;
            iter->setNode(pNode);
        } 
    } 
    catch(exception &memExp)
    {
        throw memExp;
    }

}

QuantumCircuit & QuantumCircuit::operator<<(const QuantumGate & node)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&node);
    return *this;
}

QuantumCircuit & QuantumCircuit::operator<<(const QuantumMeasure & node)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&node);
    return *this;
}

QuantumCircuit & QuantumCircuit::dagger()
{
    m_bIsDagger = true;
    return *this;
}

QuantumCircuit & QuantumCircuit::control(vector<Qubit *>& quBitVector)
{
    for (auto aiter : quBitVector)
    {
        m_controlQuBitVector.push_back(aiter);
    }

    return *this;
}


NodeType QuantumCircuit::getNodeType() const
{
    return m_iNodeType;
}

bool QuantumCircuit::isDagger() const
{
    return m_bIsDagger;
}

bool QuantumCircuit::getControlVector(vector<Qubit *>& qBitVector)
{
    ReadLock rl(m_sm);
    for (auto aiter : m_controlQuBitVector)
    {
        qBitVector.push_back(aiter);
    }
    return true;
}

NodeIter  QuantumCircuit::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pHead);
    return temp;
}

NodeIter  QuantumCircuit::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pEnd);
    return temp;
}

NodeIter QuantumCircuit::getEndNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter QuantumCircuit::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

int QuantumCircuit::getPosition() const
{
    return this->iPosition;
}

QuantumProgram::~QuantumProgram()
{
    Item *temp;
    if (m_pHead != nullptr)
    {
        while (m_pHead != m_pEnd)
        {
            temp = m_pHead;
            m_pHead = m_pHead->getNext();
            m_pHead->setPre(nullptr);
            delete temp;
        }

        delete m_pHead;
        m_pHead = nullptr;
        m_pEnd = nullptr;
    }

}

void QuantumProgram :: pushBackNode(QNode * pNode)
{
    if (nullptr == m_pHead)
    {
        Item *iter = new OriginItem();
        iter->setNext(nullptr);
        iter->setPre(nullptr);
        iter->setNode(pNode);
        m_pHead = iter;
        m_pEnd = iter;
    }
    else
    {
        Item *iter = new OriginItem();
        iter->setNext(nullptr);
        iter->setPre(m_pEnd);
        m_pEnd->setNext(iter);
        m_pEnd = iter;
        iter->setNode(pNode);
    }
}

QuantumProgram & QuantumProgram::operator<<(const QuantumIf & ifNode)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&ifNode);
    return *this;
}

QuantumProgram & QuantumProgram::operator<<(const QuantumWhile & whileNode)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&whileNode);
    return *this;
}

QuantumProgram & QuantumProgram::operator<<(const QuantumMeasure & measure)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&measure);
    return *this;
}

QuantumProgram & QuantumProgram::operator<<(const QuantumProgram & qprog)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&qprog);
    return *this;
}

QuantumProgram & QuantumProgram::operator<<(const QuantumGate & node)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&node);
    return *this;
}

QuantumProgram & QuantumProgram::operator<<(const QuantumCircuit & qCircuit)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&qCircuit);
    return *this;
}

NodeIter  QuantumProgram::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pHead);
    return temp;
}

NodeIter  QuantumProgram::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pEnd);
    return temp;
}

NodeIter QuantumProgram::getEndNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter QuantumProgram::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeType QuantumProgram::getNodeType() const
{
    return m_iNodeType;
}

void QuantumProgram::clear()
{
    Item *temp;
    if (m_pHead != nullptr)
    {
        while (m_pHead != m_pEnd)
        {
            temp = m_pHead;
            m_pHead = m_pHead->getNext();
            m_pHead->setPre(nullptr);
            delete temp;
        }
        delete m_pHead;
        m_pHead = nullptr;
        m_pEnd = nullptr;
    }

}

int QuantumProgram:: getPosition() const
{
    return iPosition;
}

NodeIter &NodeIter::operator ++()
{
    if (nullptr != m_pCur) 
    {
        this->m_pCur = m_pCur->getNext();
    }
    return *this;

}

QNode * NodeIter::operator*()
{
    if (nullptr != m_pCur)
    {
        return m_pCur->getNode();
    }
    return nullptr;

}

NodeIter & NodeIter::operator--()
{
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getPre();

    }
    return *this;
}

bool NodeIter::operator!=(NodeIter  iter)
{
    return this->m_pCur != iter.m_pCur;
}

bool NodeIter::operator==(NodeIter iter)
{
    return this->m_pCur == iter.m_pCur;
}

 Item * OriginItem:: getNext()const
 {
     return m_pNext;
 }
 Item * OriginItem::getPre()const
 {
     return m_pPre;
 }
 QNode *OriginItem:: getNode() const
 {
     auto aiter = _G_QNodeVector.getNode(m_iNodeNum);
     return *aiter;
 }
 void  OriginItem::setNext(Item * pItem)
 {
     m_pNext = pItem;
 }
 void OriginItem ::setPre(Item * pItem)
 {
     m_pPre = pItem;
 }
 void OriginItem:: setNode(QNode * pNode)
 {
     m_iNodeNum = pNode->getPosition();
 }

QuantumGate & QGateNodeFactory::getGateNode(string & name, Qubit * qbit)
{
    QGate * pGate = m_pGateFact->getGateNode(name);
    QuantumGate * QGateNode = new QuantumGate(qbit,pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QuantumGate & QGateNodeFactory::getGateNode(string & name, Qubit * qbit, double angle)
{
    QGate * pGate = m_pGateFact->getGateNode(name, angle);
    QuantumGate * QGateNode = new QuantumGate(qbit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QuantumGate & QGateNodeFactory::getGateNode(string & name, Qubit * targetQBit, Qubit * controlQBit)
{
    QGate * pGate = m_pGateFact->getGateNode(name);
    QuantumGate * QGateNode = new QuantumGate(targetQBit, controlQBit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QuantumGate & QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    string name = "QSingleGate";
    QGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QuantumGate * QGateNode = new QuantumGate(qbit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QuantumGate & QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "QDoubleGate";
    QGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QuantumGate * QGateNode = new QuantumGate(targetQBit, controlQBit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

static QGateNodeFactory * _gs_pGateNodeFactory = QGateNodeFactory::getInstance();

QuantumGate & RX(Qubit * qbit)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QuantumGate & RX(Qubit * qbit,double angle)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QuantumGate & RY(Qubit * qbit)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QuantumGate & RY(Qubit * qbit, double angle)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QuantumGate & RZ(Qubit * qbit)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QuantumGate & RZ(Qubit * qbit, double angle)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QuantumGate & S(Qubit * qbit)
{
    string name = "SGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QuantumGate & H(Qubit * qbit)
{
    string name = "HadamardGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QuantumGate & CNOT(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CNOTGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

QuantumGate & CZ(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CZGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

QuantumGate & QSingle(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta ,qbit);
}

QuantumGate & QDouble(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, targetQBit, controlQBit);
}

