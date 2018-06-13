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
#include "QPanda/QPandaException.h"


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

QGate::~QGate()
{
    _G_QNodeMap.deleteNode(m_iPosition);

}

QGate::QGate(const QGate & oldGate)
{
    m_iPosition = oldGate.getPosition();
    auto aiter = _G_QNodeMap.getNode(m_iPosition);
    if (aiter == nullptr)
        throw circuit_not_found_exception("there is no this QGate", false);
    m_pQGateNode = dynamic_cast<AbstractQGateNode *>(aiter);
    if (!_G_QNodeMap.addNodeRefer(m_iPosition))
        throw exception();
}

QGate::QGate(Qubit * qbit, QuantumGate *pQGate)
{
    if (nullptr == pQGate)
        throw param_error_exception("OriginGate param err", false);
    if (nullptr == qbit)
        throw param_error_exception("OriginGate param err", false);
    AbstractQGateNode * pTemp = new OriginQGate(qbit, pQGate);
    m_iPosition = _G_QNodeMap.pushBackNode(dynamic_cast<QNode *>(pTemp) );
    m_pQGateNode = pTemp;
    if (!_G_QNodeMap.addNodeRefer(m_iPosition))
        throw exception();

}

QGate::QGate(Qubit * targetQuBit, Qubit * controlQuBit, QuantumGate *pQGate)
{
    if (nullptr == pQGate)
        throw param_error_exception("OriginGate param err", false);
    if (nullptr == targetQuBit)
        throw param_error_exception("OriginGate param err", false);
    if (nullptr == controlQuBit)
        throw param_error_exception("OriginGate param err", false);
    AbstractQGateNode * pTemp = new OriginQGate(targetQuBit, controlQuBit, pQGate);
    m_iPosition = _G_QNodeMap.pushBackNode(dynamic_cast<QNode *>(pTemp));
    m_pQGateNode = pTemp;
    if (!_G_QNodeMap.addNodeRefer(m_iPosition))
        throw exception();
}

NodeType QGate::getNodeType() const
{
    if (nullptr == m_pQGateNode)
        throw exception();
    auto aTemp = dynamic_cast<QNode *>(m_pQGateNode);
    return aTemp->getNodeType();
}

size_t QGate::getQuBitVector(vector<Qubit *>& vector) const
{
    if (nullptr == m_pQGateNode)
        throw exception();
    return m_pQGateNode->getQuBitVector(vector);
}

size_t QGate::getQuBitNum() const
{
    if (nullptr == m_pQGateNode)
        throw exception();
    return m_pQGateNode->getQuBitNum();

}

QuantumGate * QGate::getQGate() const
{
    if (nullptr == m_pQGateNode)
        throw exception();
    return m_pQGateNode->getQGate();
}


int QGate::getPosition() const
{
    return m_iPosition;
}

bool QGate::setDagger(bool bIsDagger)
{
    if (nullptr == m_pQGateNode)
        throw exception();
    return m_pQGateNode->setDagger(bIsDagger);
}

bool QGate::setControl(vector<Qubit *>& quBitVector)
{
    if (nullptr == m_pQGateNode)
        throw exception();
    return m_pQGateNode->setControl(quBitVector);
}

bool QGate::isDagger() const
{
    if (nullptr == m_pQGateNode)
        throw exception();
    return m_pQGateNode->isDagger();
}

size_t QGate::getControlVector(vector<Qubit *>& quBitVector) const
{
    if (nullptr == m_pQGateNode)
        throw exception();
    return m_pQGateNode->getControlVector(quBitVector);
}



QCircuit::QCircuit()
{
    string sClasNname = "OriginCircuit";
    auto aMeasure = QuantumCircuitFactory::getInstance().getQuantumCircuit(sClasNname);
    m_iPosition =  _G_QNodeMap.pushBackNode(dynamic_cast<QNode *>(aMeasure));
    m_pQuantumCircuit = aMeasure;
    if (!_G_QNodeMap.addNodeRefer(m_iPosition))
        throw exception();
}

QCircuit::QCircuit(const QCircuit & oldQCircuit)
{
    m_iPosition = oldQCircuit.getPosition();
    auto aiter = _G_QNodeMap.getNode(m_iPosition);
    if (aiter !=nullptr)
        m_pQuantumCircuit = dynamic_cast<AbstractQuantumCircuit *>(aiter);
    else
        throw exception();
    if (!_G_QNodeMap.addNodeRefer(m_iPosition))
        throw exception();
}

QCircuit::~QCircuit()
{
    _G_QNodeMap.deleteNode(m_iPosition);
}


void QCircuit::pushBackNode(QNode * pNode)
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    m_pQuantumCircuit->pushBackNode(pNode);
}

QCircuit & QCircuit::operator<<(QGate  node)
{
    if (nullptr == m_pQuantumCircuit)
        throw exception();
    m_pQuantumCircuit->pushBackNode(dynamic_cast<QNode*>(&node));

    return *this;
}

QCircuit & QCircuit::operator<<( QMeasure  node)
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

NodeIter QCircuit::insertQNode(NodeIter & iter, QNode * pNode)
{
    if ((m_iPosition < 0))
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }
    auto aIter = _G_QNodeMap.getNode(m_iPosition);
    if (nullptr  == aIter)
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }

    auto pCircuit = dynamic_cast<AbstractQuantumCircuit *>(aIter);
    return pCircuit->insertQNode(iter, pNode);
}

NodeIter QCircuit::deleteQNode(NodeIter & iter)
{
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }
    auto aIter = _G_QNodeMap.getNode(m_iPosition);
    if (nullptr == aIter)
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }

    auto pCircuit = dynamic_cast<AbstractQuantumCircuit *>(aIter);
    return pCircuit->deleteQNode(iter);
}



int QCircuit::getPosition() const
{
    return this->m_iPosition;
}

QProg::QProg()
{
    string sClasNname = "OriginProgram";
    auto aMeasure = QuantumProgramFactory::getInstance().getQuantumCircuit(sClasNname);
    m_iPosition = _G_QNodeMap.pushBackNode(dynamic_cast<QNode *>(aMeasure));
    if (!_G_QNodeMap.addNodeRefer(m_iPosition))
        throw exception();
    m_pQuantumProgram = aMeasure;
}

QProg::QProg(const QProg &oldQProg)
{
    m_iPosition = oldQProg.getPosition();
    auto aiter = _G_QNodeMap.getNode(m_iPosition);
    if (nullptr != aiter)
        m_pQuantumProgram = dynamic_cast<AbstractQuantumProgram *>(aiter);
    else
        throw exception();
    if (!_G_QNodeMap.addNodeRefer(m_iPosition))
        throw exception();
}

QProg::~QProg()
{
    _G_QNodeMap.deleteNode(m_iPosition);
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

QProg & QProg::operator<<( QMeasure  measure)
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

QProg & QProg::operator<<(QGate  node)
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

NodeIter QProg::insertQNode(NodeIter & iter, QNode * pNode)
{
    if ((m_iPosition < 0))
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }
    auto aIter = _G_QNodeMap.getNode(m_iPosition);
    if (nullptr == aIter)
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }
    
    auto pProg = dynamic_cast<AbstractQuantumProgram *>(aIter);
    return pProg->insertQNode(iter, pNode);
}

NodeIter QProg::deleteQNode(NodeIter & iter)
{
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }
    auto aIter = _G_QNodeMap.getNode(m_iPosition);
    if (nullptr == aIter)
    {
        throw circuit_not_found_exception("there is no this circuit", false);
    }

    auto pProg = dynamic_cast<AbstractQuantumProgram *>(aIter);
    return pProg->deleteQNode(iter);
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

NodeIter  NodeIter::operator++(int)
{
    NodeIter temp(*this);
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getNext();
    }
    return temp;
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

NodeIter  NodeIter::operator--(int i)
{
    NodeIter temp(*this);
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getPre();

    }
    return temp;
}

bool NodeIter::operator!=(NodeIter  iter)
{
    return this->m_pCur != iter.m_pCur;
}

bool NodeIter::operator==(NodeIter iter)
{
    return this->m_pCur == iter.m_pCur;
}


 QGate  QGateNodeFactory::getGateNode(string & name, Qubit * qbit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate  QGateNodeFactory::getGateNode(string & name, Qubit * qbit, double angle)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, angle);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate  QGateNodeFactory::getGateNode(string & name, Qubit * targetQBit, Qubit * controlQBit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate  QGateNode(targetQBit, controlQBit, pGate);
    return QGateNode;
}

QGate  QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    string name = "QSingleGate";
    QuantumGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate  QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "QDoubleGate";
    QuantumGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QGate  QGateNode(targetQBit, controlQBit, pGate);
    return QGateNode;
}

static QGateNodeFactory * _gs_pGateNodeFactory = QGateNodeFactory::getInstance();

QGate  RX(Qubit * qbit)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate  RX(Qubit * qbit,double angle)
{
    string name = "XGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QGate  RY(Qubit * qbit)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate  RY(Qubit * qbit, double angle)
{
    string name = "YGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QGate  RZ(Qubit * qbit)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate  RZ(Qubit * qbit, double angle)
{
    string name = "ZGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate  S(Qubit * qbit)
{
    string name = "SGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate  H(Qubit * qbit)
{
    string name = "HadamardGate";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate  CNOT(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CNOTGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

QGate  CZ(Qubit * targetQBit, Qubit * controlQBit)
{
    string name = "CZGate";
    return _gs_pGateNodeFactory->getGateNode(name, targetQBit, controlQBit);
}

QGate  QSingle(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta ,qbit);
}

QGate  QDouble(double alpha, double beta, double gamma, double delta, Qubit * targetQBit, Qubit * controlQBit)
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

NodeIter OriginProgram::insertQNode(NodeIter & perIter, QNode * pQNode)
{

    Item * pPerItem = perIter.getPCur();
    if (nullptr == pPerItem)
    {
        throw exception();
    }



    auto aiter = this->getFirstNodeIter();

    if (this->getHeadNodeIter() == aiter)
    {
        throw exception();
    }

    for (; aiter != this->getEndNodeIter(); aiter++)
    {
        if (pPerItem == aiter.getPCur())
        {
            break;
        }
    }
    if (aiter == this->getEndNodeIter())
    {
        throw exception();
    }

    Item *pCurItem = new OriginItem();
    pCurItem->setNode(pQNode);

    if (nullptr != pPerItem->getNext())
    {
        pPerItem->getNext()->setPre(pCurItem);
        pCurItem->setNext(pPerItem->getNext());
        pPerItem->setNext(pCurItem);
        pCurItem->setPre(pPerItem);
    }
    else
    {
        pPerItem->setNext(pCurItem);
        pCurItem->setPre(pPerItem);
        pCurItem->setNext(nullptr);
    }
    NodeIter temp(pCurItem);
    return temp;
}

NodeIter OriginProgram::deleteQNode(NodeIter & targitIter)
{

    Item * pTargitItem = targitIter.getPCur();
    if (nullptr == pTargitItem)
        throw exception();

    if (nullptr == m_pHead)
    {
        throw exception();
    }

    auto aiter = this->getFirstNodeIter();
    for (; aiter != this->getEndNodeIter(); aiter++)
    {
        if (pTargitItem == aiter.getPCur())
        {
            break;
        }
    }
    if (aiter == this->getEndNodeIter())
    {
        throw exception();
    }

    if (m_pHead == pTargitItem)
    {
        if (m_pHead == m_pEnd)
        {
            delete pTargitItem;
            targitIter.setPCur(nullptr);
            m_pHead = nullptr;
            m_pEnd = nullptr;
        }
        else
        {
            m_pHead = pTargitItem->getNext();
            m_pHead->setPre(nullptr);
            delete pTargitItem;
            targitIter.setPCur(nullptr);
        }

        NodeIter temp(m_pHead);
        return temp;
    }

    if (m_pEnd = pTargitItem)
    {
        Item * pPerItem = pTargitItem->getPre();
        if (nullptr == pPerItem)
            throw exception();
        pPerItem->setNext(nullptr);
        delete(pTargitItem);
        targitIter.setPCur(nullptr);
        NodeIter temp(pPerItem);
        return temp;
    }

    Item * pPerItem = pTargitItem->getPre();
    if (nullptr == pPerItem)
        throw exception();
    pPerItem->setNext(nullptr);
    Item * pNextItem = pTargitItem->getNext();
    if (nullptr == pPerItem)
        throw exception();
    pPerItem->setNext(pNextItem);
    pNextItem->setPre(pPerItem);
    delete pTargitItem;
    targitIter.setPCur(nullptr);

    NodeIter temp(pPerItem);
    return temp;
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

NodeIter OriginCircuit::insertQNode(NodeIter & perIter, QNode * pQNode)
{

    Item * pPerItem = perIter.getPCur();
    if (nullptr == pPerItem)
    {
        throw exception();
    }



    auto aiter = this->getFirstNodeIter();

    if (this->getHeadNodeIter() == aiter)
    {
        throw exception();
    }

    for (;aiter != this->getEndNodeIter();aiter++)
    {
        if (pPerItem == aiter.getPCur())
        {
            break;
        }
    }
    if (aiter == this->getEndNodeIter())
    {
        throw exception();
    }

    Item *pCurItem = new OriginItem();
    pCurItem->setNode(pQNode);   

    if (nullptr != pPerItem->getNext())
    {
        pPerItem->getNext()->setPre(pCurItem);
        pCurItem->setNext(pPerItem->getNext());
        pPerItem->setNext(pCurItem);
        pCurItem->setPre(pPerItem);
    }
    else
    {
        pPerItem->setNext(pCurItem);
        pCurItem->setPre(pPerItem);
        pCurItem->setNext(nullptr);
    }
    NodeIter temp(pCurItem);
    return temp;
}

NodeIter OriginCircuit::deleteQNode(NodeIter & targitIter)
{

    Item * pTargitItem= targitIter.getPCur();
    if (nullptr == pTargitItem)
        throw exception();

    if (nullptr == m_pHead)
    {
        throw exception();
    }

    auto aiter = this->getFirstNodeIter();
    for (; aiter != this->getEndNodeIter(); aiter++)
    {
        if (pTargitItem == aiter.getPCur())
        {
            break;
        }
    }
    if (aiter == this->getEndNodeIter())
    {
        throw exception();
    }

    if (m_pHead == pTargitItem)
    {
        if (m_pHead == m_pEnd)
        {
            delete pTargitItem;
            targitIter.setPCur(nullptr);
            m_pHead = nullptr;
            m_pEnd = nullptr;
        }
        else
        {
            m_pHead = pTargitItem->getNext();
            m_pHead->setPre(nullptr);
            delete pTargitItem;
            targitIter.setPCur(nullptr);
        }

        NodeIter temp(m_pHead);
        return temp;
    }
    
    if (m_pEnd = pTargitItem)
    {
        Item * pPerItem = pTargitItem->getPre();
        if (nullptr == pPerItem)
            throw exception();
        pPerItem->setNext(nullptr);
        delete(pTargitItem);
        targitIter.setPCur(nullptr);
        NodeIter temp(pPerItem);
        return temp;
    }

    Item * pPerItem = pTargitItem->getPre();
    if (nullptr == pPerItem)
        throw exception();
    pPerItem->setNext(nullptr);
    Item * pNextItem = pTargitItem->getNext();
    if (nullptr == pPerItem)
        throw exception();
    pPerItem->setNext(pNextItem);
    pNextItem->setPre(pPerItem);
    delete pTargitItem;
    targitIter.setPCur(nullptr);

    NodeIter temp(pPerItem);
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

OriginQGate::~OriginQGate()
{
    if (nullptr != m_pQGate)
    {
        delete m_pQGate;
    }
}

OriginQGate::OriginQGate(Qubit * qbit, QuantumGate *pQGate)
{
    if (nullptr == pQGate)
        throw param_error_exception("OriginGate param err", false);
    if (nullptr == qbit)
        throw param_error_exception("OriginGate param err", false);
    m_pQGate = pQGate;
    m_QuBitVector.push_back(qbit);
    m_iNodeType = GATE_NODE;
}

OriginQGate::OriginQGate(Qubit *targetQuBit, Qubit * controlQuBit , QuantumGate * pQGate)
{
    if (nullptr == pQGate)
        throw param_error_exception("OriginGate param err", false);
    if (nullptr == targetQuBit)
        throw param_error_exception("OriginGate param err", false);
    if (nullptr == controlQuBit)
        throw param_error_exception("OriginGate param err", false);
    m_pQGate = pQGate;
    m_QuBitVector.push_back(targetQuBit);
    m_QuBitVector.push_back(controlQuBit);
    m_iNodeType = GATE_NODE;
}

NodeType OriginQGate::getNodeType() const
{
    return m_iNodeType;
}

size_t OriginQGate::getQuBitVector(vector<Qubit*>& vector ) const
{
    for (auto aiter : m_QuBitVector)
    {
        vector.push_back(aiter);
    }
    return m_QuBitVector.size();
}

size_t OriginQGate::getQuBitNum() const
{
    return m_QuBitVector.size();
}

QuantumGate * OriginQGate::getQGate() const
{
    if (nullptr == m_pQGate)
        throw exception();
    return m_pQGate;
}

int OriginQGate::getPosition() const
{
    throw exception();
}

bool OriginQGate::setDagger(bool isDagger)
{
    m_bIsDagger = isDagger;
    return m_bIsDagger;
}

bool OriginQGate::setControl(vector<Qubit *>& quBitVector)
{
    for (auto aiter : quBitVector)
    {
        m_controlQuBitVector.push_back(aiter);
    }
    return true;
}

bool OriginQGate::isDagger() const
{
    return m_bIsDagger;
}

size_t OriginQGate::getControlVector(vector<Qubit *>& quBitVector) const
{
    for (auto aiter : m_controlQuBitVector)
    {
        quBitVector.push_back(aiter);
    }
    return quBitVector.size();
}
