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

QNodeVector _G_QNodeVector;

QWhileNode & CreateWhileProg(ClassicalCondition * ccCon, QNode * trueNode)
{
    QWhileNode * temp = new QWhileNode(ccCon,(QNode *)trueNode);
    _G_QNodeVector.pushBackNode(temp);
    temp->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());  
    return *temp;
}

QProg & CreateEmptyQProg()
{
    QProg * temp = new QProg();
    _G_QNodeVector.pushBackNode(temp);
    temp->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *temp;
}

QMeasureNode & Measure(Qubit * targetQuBit, CBit *targetCbit)
{
    QMeasureNode * pMeasure = new QMeasureNode(targetQuBit, targetCbit);
    _G_QNodeVector.pushBackNode(pMeasure);
    pMeasure->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *pMeasure;
}
OriginQCircuit & CreateEmptyCircuit()
{
    OriginQCircuit * temp = new OriginQCircuit();
    _G_QNodeVector.pushBackNode(temp);
    temp->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *temp;
}

QIfNode & CreateIfProg(ClassicalCondition * ccCon, QNode * trueNode)
{
    QIfNode * temp = new QIfNode(ccCon, (QNode *)trueNode);
    _G_QNodeVector.pushBackNode(temp);
    temp->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *temp;
}

QIfNode & CreateIfProg(ClassicalCondition *ccCon, QNode * trueNode, QNode * falseNode)
{
    QIfNode * temp = new QIfNode(ccCon, (QNode *)trueNode, (QNode *)falseNode);
    _G_QNodeVector.pushBackNode(temp);
    temp->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *temp;
}



QNodeVector::QNodeVector()
{

}

QNodeVector::~QNodeVector()
{
    for (auto aiter = m_pQNodeVector.begin(); aiter != m_pQNodeVector.end(); aiter++)
    {
        QNode * pNode = *aiter;
        //std::cout<<"position = " << pNode->getPosition() << endl;
        //cout << "nodetype ="<< pNode->getNodeType() << endl;
        delete (pNode);
    }
}

bool QNodeVector::pushBackNode(QNode * pNode)
{
    WriteLock wl(m_sm);
    m_pQNodeVector.push_back(pNode);
    return true;
}

size_t QNodeVector::getLastNode()
{
    ReadLock rl(m_sm);
    return m_pQNodeVector.size();
}


bool QNodeVector::setHeadNode(QProg & prog)
{
    WriteLock wl(m_sm);
    if (prog.getPosition() > m_pQNodeVector.size())
    {
        return false;
    }
    m_currentIter = m_pQNodeVector.begin() + (prog.getPosition() - 1);
    return true;
}

vector<QNode*>::iterator QNodeVector::getNode(int iNum)
{
    ReadLock rl(m_sm);
    if (iNum > m_pQNodeVector.size())
    {
        return m_pQNodeVector.end();
    }
    return m_pQNodeVector.begin()+(iNum-1);
}

vector<QNode*>::iterator QNodeVector::getEnd()
{
    return  m_pQNodeVector.end();
}

QMeasureNode::QMeasureNode(Qubit * qbit, CBit * cbit) : targetQuBit(qbit), targetCbit(cbit)
{
    m_iNodeType = MEASURE_GATE;
}

NodeType QMeasureNode::getNodeType() const
{
    return m_iNodeType;
}

Qubit * QMeasureNode::getQuBit() const
{
    return targetQuBit;
}

CBit * QMeasureNode::getCBit() const
{
    return targetCbit;
}

int QMeasureNode::getQuBitNum() const
{
    return 1;
}

int QMeasureNode::getPosition() const
{
    return iPosition;
}

OriginQGateNode::OriginQGateNode(Qubit * qbit, QGate *pQGate) : m_iNodeType(GATE_NODE)
{

    m_pGate = pQGate;
    m_QuBitVector.push_back(qbit);
}

OriginQGateNode::OriginQGateNode(Qubit * targetQuBit, Qubit * controlQuBit, QGate *pQGate)
{
    m_pGate = pQGate;
    m_QuBitVector.push_back(targetQuBit);
    m_QuBitVector.push_back(controlQuBit);
    m_iNodeType = GATE_NODE;
}

NodeType OriginQGateNode::getNodeType() const
{
    return m_iNodeType;
}

size_t OriginQGateNode::getQuBitVector(vector<Qubit *>& vector) const
{
    for (auto aiter : m_QuBitVector)
    {
        vector.push_back(aiter);
    }
    return m_QuBitVector.size();
}

size_t OriginQGateNode::getQuBitNum() const
{
    return m_QuBitVector.size();
}

QGate * OriginQGateNode::getQGate() const
{
    return m_pGate;
}

GateType OriginQGateNode::getQGateType() const
{
    return m_iGateType;
}

int OriginQGateNode::getPosition() const
{
    return iPosition;
}

bool OriginQGateNode::setDagger(bool bIsDagger)
{
    m_bIsDagger = bIsDagger;
    return m_bIsDagger;
}

bool OriginQGateNode::setControl(vector<Qubit *>& quBitVector)
{
    for (auto aiter : quBitVector)
    {
        m_controlQuBitVector.push_back(aiter);
    }
    return true;
}

bool OriginQGateNode::isDagger() const
{
    return m_bIsDagger;
}

size_t OriginQGateNode::getControlVector(vector<Qubit *>& quBitVector) const
{
    for (auto aiter : m_controlQuBitVector)
    {
        quBitVector.push_back(aiter);
    }
    return quBitVector.size();
}

NodeType QWhileNode::getNodeType() const
{
    return m_iNodeType;
}

QNode * QWhileNode::getTrueBranch() const
{
    auto aiter = _G_QNodeVector.getNode(iTrueNum);
    return *aiter;
}

QNode * QWhileNode::getFalseBranch() const
{
    return nullptr;
}


ClassicalCondition * QWhileNode::getCExpr() const
{
    return ccCondition;
}


int QWhileNode::getPosition() const
{
    return this->iPosition;
}

NodeType QIfNode::getNodeType() const
{
    return m_iNodeType;
}


QNode * QIfNode::getTrueBranch() const
{
    auto aiter = _G_QNodeVector.getNode(iTrueNum);
    return *aiter;
}

QNode * QIfNode::getFalseBranch() const
{
    auto aiter = _G_QNodeVector.getNode(iFalseNum);
    if (aiter == _G_QNodeVector.getEnd())
    {
        return nullptr;
    }
    return *aiter;
}



int QIfNode::getPosition() const
{
    return this->iPosition;
}

ClassicalCondition * QIfNode::getCExpr() const
{
    return ccCondition;
}

OriginQCircuit::~OriginQCircuit()
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

void OriginQCircuit::pushBackNode(QNode * pNode)
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

OriginQCircuit & OriginQCircuit::operator<<(const OriginQGateNode & node)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&node);
    return *this;
}

OriginQCircuit & OriginQCircuit::operator<<(const QMeasureNode & node)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&node);
    return *this;
}

OriginQCircuit & OriginQCircuit::dagger()
{
    m_bIsDagger = true;
    return *this;
}

OriginQCircuit & OriginQCircuit::control(vector<Qubit *>& quBitVector)
{
    for (auto aiter : quBitVector)
    {
        m_controlQuBitVector.push_back(aiter);
    }

    return *this;
}


NodeType OriginQCircuit::getNodeType() const
{
    return m_iNodeType;
}

bool OriginQCircuit::isDagger() const
{
    return m_bIsDagger;
}

bool OriginQCircuit::getControlVector(vector<Qubit *>& qBitVector)
{
    ReadLock rl(m_sm);
    for (auto aiter : m_controlQuBitVector)
    {
        qBitVector.push_back(aiter);
    }
    return true;
}

NodeIter  OriginQCircuit::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pHead);
    return temp;
}

NodeIter  OriginQCircuit::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pEnd);
    return temp;
}

NodeIter OriginQCircuit::getEndNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter OriginQCircuit::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

int OriginQCircuit::getPosition() const
{
    return this->iPosition;
}

QProg::~QProg()
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

void QProg :: pushBackNode(QNode * pNode)
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

QProg & QProg::operator<<(const QIfNode & ifNode)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&ifNode);
    return *this;
}

QProg & QProg::operator<<(const QWhileNode & whileNode)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&whileNode);
    return *this;
}

QProg & QProg::operator<<(const QMeasureNode & measure)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&measure);
    return *this;
}

QProg & QProg::operator<<(const QProg & qprog)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&qprog);
    return *this;
}

QProg & QProg::operator<<(const OriginQGateNode & node)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&node);
    return *this;
}

QProg & QProg::operator<<(const OriginQCircuit & qCircuit)
{
    WriteLock wl(m_sm);
    pushBackNode((QNode *)&qCircuit);
    return *this;
}

NodeIter  QProg::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pHead);
    return temp;
}

NodeIter  QProg::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_pEnd);
    return temp;
}

NodeIter QProg::getEndNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter QProg::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeType QProg::getNodeType() const
{
    return m_iNodeType;
}

void QProg::clear()
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

int QProg:: getPosition() const
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

OriginQGateNode & QGateNodeFactory::getGateNode(string & name, Qubit * qbit)
{
    QGate * pGate = m_pGateFact->getGateNode(name);
    OriginQGateNode * QGateNode = new OriginQGateNode(qbit,pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

OriginQGateNode & QGateNodeFactory::getGateNode(string & name, Qubit * qbit, double angle)
{
    QGate * pGate = m_pGateFact->getGateNode(name, angle);
    OriginQGateNode * QGateNode = new OriginQGateNode(qbit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

OriginQGateNode & QGateNodeFactory::getGateNode(string & name, Qubit * targetQBit, Qubit * controlQBit)
{
    QGate * pGate = m_pGateFact->getGateNode(name);
    OriginQGateNode * QGateNode = new OriginQGateNode(targetQBit, controlQBit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

OriginQGateNode & QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    string name = "QSingleGate";
    QGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    OriginQGateNode * QGateNode = new OriginQGateNode(qbit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

OriginQGateNode & QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "QDoubleGate";
    QGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    OriginQGateNode * QGateNode = new OriginQGateNode(targetQBit, controlQBit, pGate);
    _G_QNodeVector.pushBackNode(QGateNode);
    QGateNode->iPosition = static_cast<int>(_G_QNodeVector.getLastNode());
    return *QGateNode;
}

static QGateNodeFactory * _gs_pGateNodeFactory = QGateNodeFactory::getInstance();

OriginQGateNode & RX(Qubit * qbit)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
OriginQGateNode & RX(Qubit * qbit,double angle)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
OriginQGateNode & RY(Qubit * qbit)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
OriginQGateNode & RY(Qubit * qbit, double angle)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
OriginQGateNode & RZ(Qubit * qbit)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
OriginQGateNode & RZ(Qubit * qbit, double angle)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

OriginQGateNode & S(Qubit * qbit)
{
    string name = "SGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

OriginQGateNode & H(Qubit * qbit)
{
    string name = "HadamardGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

OriginQGateNode & CNOT(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CNOTGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

OriginQGateNode & CZ(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CZGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

OriginQGateNode & QSingle(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta ,qbit);
}

OriginQGateNode & QDouble(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, targetQBit, controlQBit);
}

