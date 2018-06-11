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



QProg  CreateEmptyQProg()
{
    QProg temp;
    return temp;
}


QCircuit  CreateEmptyCircuit()
{
    QCircuit temp;
    return temp;

}

QGate::QGate(Qubit * qbit, QuantumGate *pQGate) : m_iNodeType(GATE_NODE)
{

    m_pQGate = pQGate;
    m_QuBitVector.push_back(qbit);
}

QGate::QGate(Qubit * targetQuBit, Qubit * controlQuBit, QuantumGate *pQGate)
{
    m_pQGate = pQGate;
    m_QuBitVector.push_back(targetQuBit);
    m_QuBitVector.push_back(controlQuBit);
    m_iNodeType = GATE_NODE;
}

NodeType QGate::getNodeType() const
{
    return m_iNodeType;
}

size_t QGate::getQuBitVector(vector<Qubit *>& vector) const
{
    for (auto aiter : m_QuBitVector)
    {
        vector.push_back(aiter);
    }
    return m_QuBitVector.size();
}

size_t QGate::getQuBitNum() const
{
    return m_QuBitVector.size();
}

QuantumGate * QGate::getQGate() const
{
    return m_pQGate;
}

GateType QGate::getQGateType() const
{
    return m_iGateType;
}

int QGate::getPosition() const
{
    return iPosition;
}

bool QGate::setDagger(bool bIsDagger)
{
    m_bIsDagger = bIsDagger;
    return m_bIsDagger;
}

bool QGate::setControl(vector<Qubit *>& quBitVector)
{
    for (auto aiter : quBitVector)
    {
        m_controlQuBitVector.push_back(aiter);
    }
    return true;
}

bool QGate::isDagger() const
{
    return m_bIsDagger;
}

size_t QGate::getControlVector(vector<Qubit *>& quBitVector) const
{
    for (auto aiter : m_controlQuBitVector)
    {
        quBitVector.push_back(aiter);
    }
    return quBitVector.size();
}



QCircuit::QCircuit()
{
    string sClasNname = "OriginCircuit";
    auto aMeasure = QuantumCircuitFactory::getInstance().getQuantumCircuit(sClasNname);
    _G_QNodeVector.pushBackNode(dynamic_cast<QNode *>(aMeasure));
    m_iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    m_pQuantumCircuit = aMeasure;
}

QCircuit::QCircuit(const QCircuit & oldQCircuit)
{
    m_iPosition = oldQCircuit.getPosition();
    auto aiter = _G_QNodeVector.getNode(m_iPosition);
    if (aiter != _G_QNodeVector.getEnd())
        m_pQuantumCircuit = dynamic_cast<AbstractQuantumCircuit *>(*aiter);
    else
        throw exception();
}

QCircuit::~QCircuit()
{


}

void QCircuit::pushBackNode(QNode * pNode)
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    m_pQuantumCircuit->pushBackNode(pNode);
}

QCircuit & QCircuit::operator<<(QGate & node)
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    m_pQuantumCircuit->pushBackNode(dynamic_cast<QNode*>(&node));
    return *this;
}

QCircuit & QCircuit::operator<<( QuantumMeasure & node)
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    m_pQuantumCircuit->pushBackNode(dynamic_cast<QNode*>(&node));
    return *this;
}

QCircuit & QCircuit::dagger()
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    m_pQuantumCircuit->subDagger();
    return *this;
}

QCircuit & QCircuit::control(vector<Qubit *>& quBitVector)
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    m_pQuantumCircuit->subControl(quBitVector);
    return *this;
}


NodeType QCircuit::getNodeType() const
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    return  dynamic_cast<QNode * >(m_pQuantumCircuit)->getNodeType();
}

bool QCircuit::isDagger() const
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    return m_pQuantumCircuit->isDagger();
}

bool QCircuit::getControlVector(vector<Qubit *>& qBitVector)
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    return m_pQuantumCircuit->getControlVector(qBitVector);
}

NodeIter  QCircuit::getFirstNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    return m_pQuantumCircuit->getFirstNodeIter();
}

NodeIter  QCircuit::getLastNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    return m_pQuantumCircuit->getLastNodeIter();
}

NodeIter QCircuit::getEndNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    return m_pQuantumCircuit->getEndNodeIter();
}

NodeIter QCircuit::getHeadNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    return m_pQuantumCircuit->getHeadNodeIter();
}

int QCircuit::getPosition() const
{
    return this->m_iPosition;
}

QProg::QProg()
{
    string sClasNname = "OriginProgram";
    auto aMeasure = QuantumProgramFactory::getInstance().getQuantumCircuit(sClasNname);
    _G_QNodeVector.pushBackNode(dynamic_cast<QNode *>(aMeasure));
    m_iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    m_pQuantumProgram = aMeasure;
}

QProg::QProg(const QProg &oldQProg)
{
    m_iPosition = oldQProg.getPosition();
    auto aiter = _G_QNodeVector.getNode(m_iPosition);
    if (aiter != _G_QNodeVector.getEnd())
        m_pQuantumProgram = dynamic_cast<AbstractQuantumProgram *>(*aiter);
    else
        throw exception();
}

QProg::~QProg()
{

}

void QProg :: pushBackNode(QNode * pNode)
{
    if (nullptr == m_pQuantumProgram)
        throw exception();
    m_pQuantumProgram->pushBackNode(pNode);
}

QProg & QProg::operator<<( QIfProg  ifNode)
{
    if (nullptr == m_pQuantumProgram)
        throw exception();
    m_pQuantumProgram->pushBackNode(dynamic_cast<QNode *>(&ifNode));
    return *this;
}

QProg & QProg::operator<<( QWhileProg  whileNode)
{
    if (nullptr == m_pQuantumProgram)
        throw exception();
    m_pQuantumProgram->pushBackNode(dynamic_cast<QNode *>(&whileNode));
    return *this;
}

QProg & QProg::operator<<( QuantumMeasure  measure)
{
    if (nullptr == m_pQuantumProgram)
        throw exception();
    m_pQuantumProgram->pushBackNode(dynamic_cast<QNode *>(&measure));
    return *this;
}

QProg & QProg::operator<<( QProg  qprog)
{
    if (nullptr == m_pQuantumProgram)
        throw exception();
    m_pQuantumProgram->pushBackNode(dynamic_cast<QNode *>(&qprog));
    return *this;
}

QProg & QProg::operator<<(QGate & node)
{
    if (nullptr == m_pQuantumProgram)
        throw exception();
    m_pQuantumProgram->pushBackNode(dynamic_cast<QNode *>(&node));
    return *this;
}

QProg & QProg::operator<<( QCircuit  qCircuit)
{
    if (nullptr != m_pQuantumProgram)
        m_pQuantumProgram->pushBackNode(dynamic_cast<QNode *>(&qCircuit));
    return *this;
}

NodeIter  QProg::getFirstNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getFirstNodeIter();
    else
        throw exception();
}

NodeIter  QProg::getLastNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getLastNodeIter();
    else
        throw exception();
}

NodeIter QProg::getEndNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getEndNodeIter();
    else
        throw exception();
}

NodeIter QProg::getHeadNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getHeadNodeIter();
    else
        throw exception();
}

NodeType QProg::getNodeType() const
{
    if (nullptr != m_pQuantumProgram)
        return dynamic_cast<QNode * >(m_pQuantumProgram)->getNodeType();
    else
        throw exception();
}

void QProg::clear()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->clear();
    else
        throw exception();
}

int QProg:: getPosition() const
{
    return m_iPosition;
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

 QGate & QGateNodeFactory::getGateNode(string & name, Qubit * qbit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate * QGateNode = new QGate(qbit,pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QGate & QGateNodeFactory::getGateNode(string & name, Qubit * qbit, double angle)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, angle);
    QGate * QGateNode = new QGate(qbit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QGate & QGateNodeFactory::getGateNode(string & name, Qubit * targetQBit, Qubit * controlQBit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate * QGateNode = new QGate(targetQBit, controlQBit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QGate & QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    string name = "QSingleGate";
    QuantumGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QGate * QGateNode = new QGate(qbit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

QGate & QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "QDoubleGate";
    QuantumGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QGate * QGateNode = new QGate(targetQBit, controlQBit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

static QGateNodeFactory * _gs_pGateNodeFactory = QGateNodeFactory::getInstance();

QGate & RX(Qubit * qbit)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate & RX(Qubit * qbit,double angle)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QGate & RY(Qubit * qbit)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate & RY(Qubit * qbit, double angle)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QGate & RZ(Qubit * qbit)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate & RZ(Qubit * qbit, double angle)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate & S(Qubit * qbit)
{
    string name = "SGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate & H(Qubit * qbit)
{
    string name = "HadamardGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate & CNOT(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CNOTGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

QGate & CZ(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CZGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

QGate & QSingle(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta ,qbit);
}

QGate & QDouble(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, targetQBit, controlQBit);
}




OriginProgram::~OriginProgram()
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

OriginProgram::OriginProgram() : m_pHead(nullptr), m_pEnd(nullptr), m_iNodeType(PROG_NODE)
{
}

void OriginProgram::pushBackNode(QNode * pNode)
{
    WriteLock wl(m_sm);
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

NodeIter OriginProgram::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pHead);
    return temp;
}

NodeIter OriginProgram::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pEnd);
    return temp;
}

NodeIter OriginProgram::getEndNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter OriginProgram::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeType OriginProgram::getNodeType() const
{
    return m_iNodeType;
}

void OriginProgram::clear()
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

int OriginProgram::getPosition() const
{
    throw exception();
}

REGISTER_QPROGRAM(OriginProgram);

void QuantumProgramFactory::registClass(string name, CreateQProgram method)
{
    if ((name.size() <= 0) || (nullptr == method))
        throw exception();
    m_QProgMap.insert(pair<string, CreateQProgram>(name, method));
}

AbstractQuantumProgram * QuantumProgramFactory::getQuantumCircuit(std::string & name)
{
    if (name.size() <= 0)
        throw exception();
    auto aiter = m_QProgMap.find(name);
    if (aiter != m_QProgMap.end())
    {
        return aiter->second();
    }
    return nullptr;
}



OriginCircuit::~OriginCircuit()
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

void OriginCircuit::pushBackNode(QNode * pNode)
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
    catch (exception &memExp)
    {
        throw memExp;
    }
}

void OriginCircuit::subDagger()
{
    m_bIsDagger = true;
}

void OriginCircuit::subControl(vector<Qubit*>& quBitVector )
{
    for (auto aiter : quBitVector)
    {
        m_controlQuBitVector.push_back(aiter);
    }
}

NodeType OriginCircuit::getNodeType() const
{
    return m_iNodeType;
}

bool OriginCircuit::isDagger() const
{
    return m_bIsDagger;
}

bool OriginCircuit::getControlVector(vector<Qubit*>& quBitVector)
{
    for (auto aiter : m_controlQuBitVector)
    {
        quBitVector.push_back(aiter);
    }
    return quBitVector.size();
}

NodeIter OriginCircuit::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pHead);
    return temp;
}

NodeIter OriginCircuit::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pEnd);
    return temp;
}

NodeIter OriginCircuit::getEndNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter OriginCircuit::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

int OriginCircuit::getPosition() const
{
    throw exception();
}

void QuantumCircuitFactory::registClass(string name, CreateQCircuit method)
{
    if ((name.size() <= 0) || (nullptr == method))
        throw exception();
    m_QCirciutMap.insert(pair<string, CreateQCircuit>(name, method));
}

AbstractQuantumCircuit * QuantumCircuitFactory::getQuantumCircuit(std::string & name)
{
    if (name.size() <= 0)
        throw exception();
    auto aiter = m_QCirciutMap.find(name);
    if (aiter != m_QCirciutMap.end())
    {
        return aiter->second();
    }
    return nullptr;
}
REGISTER_QCIRCUIT(OriginCircuit);
