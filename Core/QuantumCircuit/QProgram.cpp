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

//#include "QProgram.h"

#include "QPanda.h"
#include "Utilities/ConfigMap.h"
#include "ClassicalProgam.h"
using namespace QGATE_SPACE;
using namespace std;
USING_QPANDA
QProg  QPanda::CreateEmptyQProg()
{
    QProg temp;
    return temp;
}

static QGateNodeFactory * _gs_pGateNodeFactory = QGateNodeFactory::getInstance();

QCircuit  QPanda::CreateEmptyCircuit()
{
    QCircuit temp;
    return temp;
}

QGate::~QGate()
{
    QNodeMap::getInstance().deleteNode(m_stPosition);
}

QGate::QGate(const QGate & oldGate)
{
    m_stPosition = oldGate.getPosition();
    auto aiter = QNodeMap::getInstance().getNode(m_stPosition);
    if (aiter == nullptr)
    {
        QCERR("Cannot find QGate");
        throw invalid_argument("Cannot find QGate");
    }


    m_pQGateNode = dynamic_cast<AbstractQGateNode *>(aiter);
    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

}

QGate::QGate(Qubit * qbit, QuantumGate *pQGate)
{
    if (nullptr == pQGate)
    {
        QCERR("pQGate param err");
        throw invalid_argument("pQGate param err");
    }
    if (nullptr == qbit)
    {
        QCERR("qbit param err");
        throw invalid_argument("qbit param err");
    }
    AbstractQGateNode * pTemp = new OriginQGate(qbit, pQGate);
    auto temp = dynamic_cast<QNode *>(pTemp);
    m_stPosition = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_stPosition);
    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQGateNode = pTemp;
}

QGate::QGate(Qubit *  controlQuBit, Qubit * targetQuBit, QuantumGate *pQGate)
{
    if (nullptr == pQGate)
    {
        QCERR("pQGate param err");
        throw invalid_argument("pQGate param err");
    }
    if (nullptr == targetQuBit)
    {
        QCERR("targetQuBit param err");
        throw invalid_argument("targetQuBit param err");
    }
    if (nullptr == controlQuBit)
    {
        QCERR("controlQuBit param err");
        throw invalid_argument("controlQuBit param err");
    }

    AbstractQGateNode * pTemp = new OriginQGate(controlQuBit, targetQuBit, pQGate);
    auto temp = dynamic_cast<QNode *>(pTemp);
    m_stPosition = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_stPosition);
    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }


    m_pQGateNode = pTemp;
}


NodeType QGate::getNodeType() const
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aTemp = dynamic_cast<QNode *>(m_pQGateNode);
    return aTemp->getNodeType();
}

size_t QGate::getQuBitVector(vector<Qubit *>& vector) const
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQGateNode->getQuBitVector(vector);
}

size_t QGate::getQuBitNum() const
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQGateNode->getQuBitNum();

}

QuantumGate * QGate::getQGate() const
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQGateNode->getQGate();
}


qmap_size_t QGate::getPosition() const
{
    return m_stPosition;
}

bool QGate::setDagger(bool bIsDagger)
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQGateNode->setDagger(bIsDagger);
}

bool QGate::setControl(QVec quBitVector)
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQGateNode->setControl(quBitVector);
}

/*****************************************************************
Name        : dagger
Description : dagger the QGate
argin       : 
argout      :
Return      : new QGate
*****************************************************************/
QGate QGate::dagger()
{
    vector<Qubit *> qubitVector;
    this->getQuBitVector(qubitVector);
    vector<Qubit*> controlVector;
    this->getControlVector(controlVector);

    QStat matrix;
    auto pQgate = this->m_pQGateNode->getQGate();
    pQgate->getMatrix(matrix);

    if (qubitVector.size() == 1)
    {
        string name = "U4";
        auto tempGate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubitVector[0]);
        tempGate.setControl(controlVector);
        tempGate.setDagger(this->isDagger() ^ true);
        return tempGate;
    }
    else
    {
        string name = "QDoubleGate";
        auto tempGate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubitVector[0], qubitVector[1]);
        tempGate.setControl(controlVector);
        tempGate.setDagger(this->isDagger() ^ true);
        return tempGate;
    }
}

/*****************************************************************
Name        : dagger
Description : set controlQubit to QGate
argin       :
argout      :
Return      : new QGate
*****************************************************************/
QGate QGate::control(QVec controlVector)
{
    vector<Qubit *> qubitVector;
    this->getQuBitVector(qubitVector);
    this->getControlVector(controlVector);

    QStat matrix;
    auto pQgate = this->m_pQGateNode->getQGate();

    pQgate->getMatrix(matrix);

    if (qubitVector.size() == 1)
    {
        string name = "U4";
        auto tempGate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubitVector[0]);
        tempGate.setControl(controlVector);
        tempGate.setDagger(this->isDagger());
        return tempGate;
    }
    else
    {
        string name = "QDoubleGate";
        auto tempGate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubitVector[0], qubitVector[1]);
        tempGate.setControl(controlVector);
        tempGate.setDagger(this->isDagger());
        return tempGate;
    }
}

bool QGate::isDagger() const
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQGateNode->isDagger();
}

size_t QGate::getControlVector(vector<Qubit *>& quBitVector) const
{
    if (nullptr == m_pQGateNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQGateNode->getControlVector(quBitVector);
}



QCircuit::QCircuit()
{
    auto sClasNname = ConfigMap::getInstance()["QCircuit"];
    auto aMeasure = QuantumCircuitFactory::getInstance().getQuantumCircuit(sClasNname);
    auto temp = dynamic_cast<QNode *>(aMeasure);
    m_stPosition = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_stPosition);
    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQuantumCircuit = aMeasure;
}

QCircuit::QCircuit(const QCircuit & oldQCircuit)
{
    m_stPosition = oldQCircuit.getPosition();
    auto aiter = QNodeMap::getInstance().getNode(m_stPosition);
    if (aiter !=nullptr)
        m_pQuantumCircuit = dynamic_cast<AbstractQuantumCircuit *>(aiter);
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

QCircuit::~QCircuit()
{
    QNodeMap::getInstance().deleteNode(m_stPosition);
}


void QCircuit::pushBackNode(QNode * pNode)
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQuantumCircuit->pushBackNode(pNode);
}

QCircuit QCircuit::dagger()
{
    QCircuit qCircuit;
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aiter = m_pQuantumCircuit->getFirstNodeIter();
    if (aiter == m_pQuantumCircuit->getEndNodeIter())
    {
        return qCircuit;
    }

    for (; aiter != m_pQuantumCircuit->getEndNodeIter(); ++aiter)
    {
        qCircuit.pushBackNode(*aiter);
    }

    qCircuit.setDagger(true^this->isDagger());
    return qCircuit;
}

QCircuit  QCircuit::control(vector<Qubit *>& quBitVector)
{
    QCircuit qCircuit;
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aiter = m_pQuantumCircuit->getFirstNodeIter();
    if (aiter == m_pQuantumCircuit->getEndNodeIter())
    {
        return qCircuit;
    }
    for (; aiter != m_pQuantumCircuit->getEndNodeIter(); ++aiter)
    {
        qCircuit.pushBackNode(*aiter);
    }

    qCircuit.setControl(quBitVector);
    return qCircuit;
}


NodeType QCircuit::getNodeType() const
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return dynamic_cast<QNode*>(m_pQuantumCircuit)->getNodeType();
}

bool QCircuit::isDagger() const
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->isDagger();
}

bool QCircuit::getControlVector(vector<Qubit *>& qBitVector)
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getControlVector(qBitVector);
}

NodeIter  QCircuit::getFirstNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getFirstNodeIter();
}

NodeIter  QCircuit::getLastNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getLastNodeIter();
}

NodeIter QCircuit::getEndNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getEndNodeIter();
}

NodeIter QCircuit::getHeadNodeIter()
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getHeadNodeIter();
}

NodeIter QCircuit::insertQNode(NodeIter & iter, QNode * pNode)
{
    if (m_stPosition < 0)
    {
        QCERR("Cannot find the circuit");
        throw invalid_argument("Cannot find the circuit");
    }
    auto aIter = QNodeMap::getInstance().getNode(m_stPosition);
    if (nullptr  == aIter)
    {
        QCERR("Cannot find the circuit");
        throw invalid_argument("Cannot find the circuit");
    }

    auto pCircuit = dynamic_cast<AbstractQuantumCircuit *>(aIter);
    return pCircuit->insertQNode(iter, pNode);
}

NodeIter QCircuit::deleteQNode(NodeIter & iter)
{
    if (m_stPosition < 0)
    {
        QCERR("Cannot find the circuit");
        throw invalid_argument("Cannot find the circuit");
    }

    auto aIter = QNodeMap::getInstance().getNode(m_stPosition);
    if (nullptr == aIter)
    {
        QCERR("Cannot find the circuit");
        throw invalid_argument("Cannot find the circuit");
    }

    auto pCircuit = dynamic_cast<AbstractQuantumCircuit *>(aIter);
    return pCircuit->deleteQNode(iter);
}

void QCircuit::setDagger(bool isDagger)
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQuantumCircuit->setDagger(isDagger);
}

void QCircuit::setControl(vector<Qubit*>& controlBitVector)
{
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQuantumCircuit->setControl(controlBitVector);
}

qmap_size_t QCircuit::getPosition() const
{
    return this->m_stPosition;
}

void QCircuit::setPosition(qmap_size_t stPosition)
{
    m_stPosition = stPosition;
}

QProg::QProg()
{
    auto sClasNname = ConfigMap::getInstance()["QProg"];
    auto aMeasure = QuantumProgramFactory::getInstance().getQuantumQProg(sClasNname);
    auto temp = dynamic_cast<QNode *>(aMeasure);
    m_stPosition = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_stPosition);
    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    m_pQuantumProgram = aMeasure;
}

QProg::QProg(const QProg &oldQProg)
{
    m_stPosition = oldQProg.getPosition();
    auto aiter = QNodeMap::getInstance().getNode(m_stPosition);
    if (nullptr != aiter)
        m_pQuantumProgram = dynamic_cast<AbstractQuantumProgram *>(aiter);
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

QProg::~QProg()
{
    QNodeMap::getInstance().deleteNode(m_stPosition);
}

void QProg :: pushBackNode(QNode * pNode)
{
    if (nullptr == m_pQuantumProgram)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    m_pQuantumProgram->pushBackNode(pNode);
}

NodeIter  QProg::getFirstNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getFirstNodeIter();
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

NodeIter  QProg::getLastNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getLastNodeIter();
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

NodeIter QProg::getEndNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getEndNodeIter();
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

NodeIter QProg::getHeadNodeIter()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->getHeadNodeIter();
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

NodeIter QProg::insertQNode(NodeIter & iter, QNode * pNode)
{
    if ((m_stPosition < 0))
    {
        QCERR("Cannot find circuit");
        throw invalid_argument("Cannot find circuit");
    }
    auto aIter = QNodeMap::getInstance().getNode(m_stPosition);
    if (nullptr == aIter)
    {
        QCERR("Cannot find circuit");
        throw invalid_argument("Cannot find circuit");
    }
    
    auto pProg = dynamic_cast<AbstractQuantumProgram*>(aIter);
    return pProg->insertQNode(iter, pNode);
}

NodeIter QProg::deleteQNode(NodeIter & iter)
{
    if ((m_stPosition < 0))
    {
        QCERR("Cannot find circuit");
        throw invalid_argument("Cannot find circuit");
    }
    auto aIter = QNodeMap::getInstance().getNode(m_stPosition);
    if (nullptr == aIter)
    {
        QCERR("Cannot find circuit");
        throw invalid_argument("Cannot find circuit");
    }

    auto pProg = dynamic_cast<AbstractQuantumProgram*>(aIter);
    return pProg->deleteQNode(iter);
}

NodeType QProg::getNodeType() const
{
    if (nullptr != m_pQuantumProgram)
        return dynamic_cast<QNode*>(m_pQuantumProgram)->getNodeType();
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

void QProg::clear()
{
    if (nullptr != m_pQuantumProgram)
        return m_pQuantumProgram->clear();
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

qmap_size_t QProg::getPosition() const
{
    return m_stPosition;
}

template <>
QProg & QProg::operator<<<ClassicalCondition>(ClassicalCondition cc)
{
    ClassicalProg temp(cc);
    auto node = dynamic_cast<QNode *>(&temp);
    if (nullptr == node)
    {
        QCERR("node is not base of ClassicalProg");
        throw runtime_error("node is not base of ClassicalProg");
    }
    pushBackNode(node);
    return *this;
}

NodeIter& NodeIter::operator++()
{
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getNext();
    }
    return *this;
}

NodeIter NodeIter::operator++(int)
{
    NodeIter temp(*this);
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getNext();
    }
    return temp;
}

QNode* NodeIter::operator*()
{
    if (nullptr != m_pCur)
    {
        return m_pCur->getNode();
    }
    return nullptr;
}

NodeIter& NodeIter::operator--()
{
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getPre();
    }
    return *this;
}

NodeIter NodeIter::operator--(int i)
{
    NodeIter temp(*this);
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getPre();

    }
    return temp;
}

NodeIter NodeIter::getNextIter()
{
    if (nullptr != m_pCur)
    {
        auto pItem = m_pCur->getNext();
        NodeIter temp(pItem);
        return temp;
    }
    else
    {
        NodeIter temp(nullptr);
        return temp;
    }
}

bool NodeIter::operator!=(NodeIter  iter)
{
    return this->m_pCur != iter.m_pCur;
}

bool NodeIter::operator==(NodeIter iter)
{
    return this->m_pCur == iter.m_pCur;
}

QGate QGateNodeFactory::getGateNode(const string & name, Qubit * qbit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string & name, Qubit * qbit, double angle)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, angle);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string & name, Qubit * controlQBit , Qubit * targetQBit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate  QGateNode(controlQBit, targetQBit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string & name, Qubit * controlQBit, Qubit * targetQBit,double theta)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name,theta);
    QGate  QGateNode(controlQBit, targetQBit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    string name = "U4";
    QuantumGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}
  
QGate QGateNodeFactory::getGateNode(double alpha, double beta, double gamma, double delta, Qubit * controlQBit, Qubit * targetQBit)
{
    string name = "CU";
    QuantumGate * pGate = m_pGateFact->getGateNode(name,alpha, beta, gamma, delta);
    QGate  QGateNode(controlQBit, targetQBit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string &name, QStat matrix, Qubit * controlQBit, Qubit * targetQBit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, matrix);
    QGate  QGateNode(controlQBit, targetQBit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string &name, QStat matrix, Qubit * targetQBit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, matrix);
    QGate  QGateNode(targetQBit, pGate);
    return QGateNode;
}

QGate QPanda::X(Qubit * qbit)
{
    string name = "X";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::X1(Qubit * qbit)
{
    string name = "X1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::RX(Qubit * qbit,double angle)
{
    string name = "RX";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate QPanda::U1(Qubit * qbit, double angle)
{
    string name = "U1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate QPanda::Y(Qubit * qbit)
{
    string name = "Y";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::Y1(Qubit * qbit)
{
    string name = "Y1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::RY(Qubit * qbit, double angle)
{
    string name = "RY";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QGate QPanda::Z(Qubit * qbit)
{
    string name = "Z";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate QPanda::Z1(Qubit * qbit)
{
    string name = "Z1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::RZ(Qubit * qbit, double angle)
{
    string name = "RZ";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt,Qubit * targitBit_second)
{
    string name = "ISWAP";
    return _gs_pGateNodeFactory->getGateNode(name, targitBit_fisrt, targitBit_second);
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second,double theta)
{
    string name = "ISWAP";
    return _gs_pGateNodeFactory->getGateNode(name, 
        targitBit_fisrt,
        targitBit_second,
        theta);
}

QGate QPanda::CR(Qubit * controlBit , Qubit * targitBit, double theta)
{
    string name = "CPhaseGate";
    return _gs_pGateNodeFactory->getGateNode(name, controlBit , targitBit , theta);
}

QGate QPanda::SqiSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "SQISWAP";
    return _gs_pGateNodeFactory->getGateNode(name, 
        targitBit_fisrt,
        targitBit_second);
}

QGate QPanda::S(Qubit * qbit)
{
    string name = "S";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::T(Qubit * qbit)
{
    string name = "T";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate  QPanda::H(Qubit * qbit)
{
    string name = "H";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate  QPanda::CNOT(Qubit * controlQBit , Qubit * targetQBit)
{
    string name = "CNOT";
    return _gs_pGateNodeFactory->getGateNode(name, controlQBit , targetQBit);
}

QGate QPanda::CZ(Qubit * controlQBit , Qubit *targetQBit)
{
    string name = "CZ";
    return _gs_pGateNodeFactory->getGateNode(name, controlQBit , targetQBit);
}

QGate QPanda::U4(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta ,qbit);
}

QGate QPanda::U4(QStat & matrix, Qubit *qubit)
{
    string name = "U4";
    return _gs_pGateNodeFactory->getGateNode(name, matrix, qubit);
}

QGate QPanda::CU(double alpha, double beta, double gamma, double delta, Qubit * controlQBit, Qubit * targetQBit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, controlQBit, targetQBit);
}

QGate QPanda::CU(QStat & matrix, Qubit * controlQBit, Qubit * targetQBit)
{
    string name = "CU";
    return _gs_pGateNodeFactory->getGateNode(name, matrix, controlQBit, targetQBit);
}

QGate QPanda::QDouble(QStat matrix, Qubit * pQubit1, Qubit * pQubit2)
{
    string name = "QDoubleGate";
    return _gs_pGateNodeFactory->getGateNode(name,matrix, pQubit1, pQubit2);
}

OriginProgram::~OriginProgram()
{
    Item *temp;

    while (m_pHead != nullptr)
    {
        m_pHead->setPre(nullptr);
        temp = m_pHead;
        m_pHead = m_pHead->getNext();
        delete temp;
    }
    m_pHead = nullptr;
    m_pEnd = nullptr;
}

OriginProgram::OriginProgram() : m_pHead(nullptr), m_pEnd(nullptr), m_iNodeType(PROG_NODE)
{ }

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
    WriteLock wl(m_sm);
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
    ReadLock * rl = new ReadLock(m_sm);
    Item * pPerItem = perIter.getPCur();
    if (nullptr == pPerItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aiter = this->getFirstNodeIter();

    if (this->getHeadNodeIter() == aiter)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
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
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    delete rl;
    WriteLock wl(m_sm);
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
        m_pEnd = pCurItem;
    }
    NodeIter temp(pCurItem);
    return temp;
}

NodeIter OriginProgram::deleteQNode(NodeIter & targitIter)
{
    ReadLock *rl = new ReadLock(m_sm);
    Item * pTargitItem = targitIter.getPCur();
    if (nullptr == pTargitItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }


    if (nullptr == m_pHead)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
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
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }


    delete rl;
    WriteLock wl(m_sm);

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

    if (m_pEnd == pTargitItem)
    {
        Item * pPerItem = pTargitItem->getPre();
        if (nullptr == pPerItem)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        pPerItem->setNext(nullptr);
        delete(pTargitItem);
        targitIter.setPCur(nullptr);
        NodeIter temp(pPerItem);
        return temp;
    }

    Item * pPerItem = pTargitItem->getPre();
    if (nullptr == pPerItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    pPerItem->setNext(nullptr);
    Item * pNextItem = pTargitItem->getNext();
    if (nullptr == pPerItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    pPerItem->setNext(pNextItem);
    pNextItem->setPre(pPerItem);
    delete pTargitItem;
    targitIter.setPCur(nullptr);

    NodeIter temp(pPerItem);
    return temp;
}

qmap_size_t OriginProgram::getPosition() const
{
    return m_stPosition;
}

void OriginProgram::setPosition(qmap_size_t stPosition)
{
    m_stPosition = stPosition;
}

REGISTER_QPROGRAM(OriginProgram);

void QuantumProgramFactory::registClass(string name, CreateQProgram method)
{
    if ((name.size() <= 0) || (nullptr == method))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    m_QProgMap.insert(pair<string, CreateQProgram>(name, method));
}

AbstractQuantumProgram * QuantumProgramFactory::getQuantumQProg(std::string & name)
{
    if (name.size() <= 0)
    {
        QCERR("param error");
        throw runtime_error("param error");
    }
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
    WriteLock wl(m_sm);
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
        QCERR(memExp.what());
        throw memExp;
    }
}

void OriginCircuit::setDagger(bool isDagger)
{
    m_bIsDagger = isDagger;
}

void OriginCircuit::setControl(vector<Qubit*>& quBitVector )
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
    NodeIter temp(nullptr);
    return temp;
}

NodeIter OriginCircuit::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter OriginCircuit::insertQNode(NodeIter & perIter, QNode * pQNode)
{
    ReadLock * rl = new ReadLock(m_sm);
    Item * pPerItem = perIter.getPCur();
    if (nullptr == pPerItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aiter = this->getFirstNodeIter();

    if (this->getHeadNodeIter() == aiter)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
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
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    delete rl;

    WriteLock wl(m_sm);
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
        m_pEnd = pCurItem;
    }
    NodeIter temp(pCurItem);
    return temp;
}

NodeIter OriginCircuit::deleteQNode(NodeIter & targitIter)
{
    ReadLock * rl = new ReadLock(m_sm);
    Item * pTargitItem= targitIter.getPCur();
    if (nullptr == pTargitItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    if (nullptr == m_pHead)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
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
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    delete rl;

    WriteLock wl(m_sm);
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
    
    if (m_pEnd == pTargitItem)
    {
        Item * pPerItem = pTargitItem->getPre();
        if (nullptr == pPerItem)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        pPerItem->setNext(nullptr);
        delete(pTargitItem);
        targitIter.setPCur(nullptr);
        m_pEnd = pPerItem;
        NodeIter temp(pPerItem);
        return temp;
    }

    Item * pPerItem = pTargitItem->getPre();
    if (nullptr == pPerItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    pPerItem->setNext(nullptr);
    Item * pNextItem = pTargitItem->getNext();
    if (nullptr == pPerItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    pPerItem->setNext(pNextItem);
    pNextItem->setPre(pPerItem);
    delete pTargitItem;
    targitIter.setPCur(nullptr);

    NodeIter temp(pPerItem);
    return temp;
}

qmap_size_t OriginCircuit::getPosition() const
{
    return m_stPosition;
}

void OriginCircuit::setPosition(qmap_size_t stPosition)
{
    m_stPosition = stPosition;
}

void OriginCircuit::clearControl()
{
    m_controlQuBitVector.clear();
    m_controlQuBitVector.resize(0);
}

void QuantumCircuitFactory::registClass(string name, CreateQCircuit method)
{
    if ((name.size() <= 0) || (nullptr == method))
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    m_QCirciutMap.insert(pair<string, CreateQCircuit>(name, method));
}

AbstractQuantumCircuit * QuantumCircuitFactory::getQuantumCircuit(std::string & name)
{
    if (name.size() <= 0)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

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

OriginQGate::OriginQGate(Qubit * qbit, QuantumGate *pQGate) :m_bIsDagger(false)
{
    if (nullptr == pQGate)
    {
        QCERR("pQGate param err");
        throw invalid_argument("pQGate param err");
    }
    if (nullptr == qbit)
    {
        QCERR("qbit param is null");
        throw invalid_argument("qbit param s null");
    }
    m_pQGate = pQGate;
    m_QuBitVector.push_back(qbit);
    m_iNodeType = GATE_NODE;
}

OriginQGate::OriginQGate(Qubit * controlQuBit , Qubit * targetQuBit, QuantumGate * pQGate) :m_bIsDagger(false)
{
    if (nullptr == pQGate)
    {
        QCERR("pQGate param err");
        throw invalid_argument("pQGate param err");
    }
    if (nullptr == targetQuBit)
    {
        QCERR("targetQuBit param is null");
        throw invalid_argument("targetQuBit param s null");
    }
    if (nullptr == controlQuBit)
    {
        QCERR("controlQuBit param is null");
        throw invalid_argument("controlQuBit param s null");
    }
    m_pQGate = pQGate;
    m_QuBitVector.push_back(controlQuBit);
    m_QuBitVector.push_back(targetQuBit);
    m_iNodeType = GATE_NODE;
}

OriginQGate::OriginQGate(vector<Qubit*> &qubit_vector, QuantumGate *pQGate) :m_bIsDagger(false)
{
    if (nullptr == pQGate)
    {
        QCERR("pQGate param err");
        throw invalid_argument("pQGate param err");
    }
    if(0 == qubit_vector.size())
    m_pQGate = pQGate;
    for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end();++aiter)
    {
        m_QuBitVector.push_back(*aiter);
    }
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

Qubit * OriginQGate::popBackQuBit()
{
    auto temp = m_QuBitVector.back();
    m_QuBitVector.pop_back();
    return temp;
}

QuantumGate * OriginQGate::getQGate() const
{
    if (nullptr == m_pQGate)
    {
        QCERR("m_pQGate is null");
        throw runtime_error("m_pQGate is null");
    }
    return m_pQGate;
}

qmap_size_t OriginQGate::getPosition() const
{
    return  m_stPosition;
}
void OriginQGate::setQGate(QuantumGate * pQGate)
{
    m_pQGate = pQGate;
}
void OriginQGate::setPosition(qmap_size_t iPosition)
{

    m_stPosition = iPosition;
}

bool OriginQGate::setDagger(bool isDagger)
{
    m_bIsDagger = isDagger;
    return m_bIsDagger;
}

bool OriginQGate::setControl(QVec quBitVector)
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

void OriginQGate::PushBackQuBit(Qubit * pQubit)
{

    if (nullptr == pQubit)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    m_QuBitVector.push_back(pQubit);

}


HadamardQCircuit::HadamardQCircuit(QVec& pQubitVector)
{
    for (auto aiter :pQubitVector)
    {
        auto  temp = H(aiter);
        m_pQuantumCircuit->pushBackNode((QNode *)&temp);
    }
}

HadamardQCircuit QPanda::CreateHadamardQCircuit(QVec & pQubitVector)
{
    HadamardQCircuit temp(pQubitVector);
    return temp;
}
