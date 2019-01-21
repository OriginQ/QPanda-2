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

#include "QuantumMeasure.h"
#include "Utilities/ConfigMap.h"
USING_QPANDA
using namespace std;
QMeasure  QPanda::Measure(Qubit * targetQuBit,ClassicalCondition  classical_cond)
{
    auto targetCbit = classical_cond.getExprPtr()->getCBit();
    if (nullptr == targetCbit)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
    QMeasure qMeasure(targetQuBit, targetCbit);
    return qMeasure;
}

QMeasure::QMeasure(const QMeasure & oldMeasure)
{
    m_stPosition = oldMeasure.getPosition();    
    auto aiter = QNodeMap::getInstance().getNode(m_stPosition);
    if (aiter != nullptr)
        m_pQuantumMeasure = dynamic_cast<AbstractQuantumMeasure *>(aiter);
    else
    {
        QCERR("there is not target QNode");
        throw runtime_error("there is not target QNode");
    }

    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

QMeasure::QMeasure(Qubit * qbit, CBit * cbit)
{
    auto sClasNname = ConfigMap::getInstance()["QMeasure"];
    auto aMeasure = QuantunMeasureFactory::getInstance().getQuantumMeasure(sClasNname, qbit, cbit);
    auto temp = dynamic_cast<QNode *>(aMeasure);
    m_stPosition = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(m_stPosition);
    if (!QNodeMap::getInstance().addNodeRefer(m_stPosition))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    m_pQuantumMeasure = aMeasure;
}

QMeasure::~QMeasure()
{
    QNodeMap::getInstance().deleteNode(m_stPosition);
}

Qubit * QMeasure::getQuBit() const
{
    if (nullptr == m_pQuantumMeasure)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_pQuantumMeasure->getQuBit();
}

CBit * QMeasure::getCBit() const
{
    if (nullptr == m_pQuantumMeasure)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_pQuantumMeasure->getCBit();
}


NodeType QMeasure::getNodeType() const
{
    if (nullptr == m_pQuantumMeasure)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return (dynamic_cast<QNode *>(m_pQuantumMeasure))->getNodeType();
}

qmap_size_t QMeasure::getPosition() const
{
    return m_stPosition;
}

void QuantunMeasureFactory::registClass(string name, CreateMeasure method)
{
    m_measureMap.insert(pair<string, CreateMeasure>(name, method));
}

AbstractQuantumMeasure * QuantunMeasureFactory::getQuantumMeasure(std::string & classname, Qubit * pQubit, CBit * pCBit)
{
    auto aiter = m_measureMap.find(classname);
    if (aiter != m_measureMap.end())
    {
        return aiter->second(pQubit, pCBit);
    }
    else
    {
        QCERR("can not find targit measure class");
        throw runtime_error("can not find targit measure class");
    }
}

NodeType OriginMeasure::getNodeType() const
{
    return m_iNodeType;
}

OriginMeasure::OriginMeasure(Qubit * pQubit, CBit * pCBit):m_pTargetQubit(pQubit),m_pCBit(pCBit),m_iNodeType(MEASURE_GATE)
{
}

Qubit * OriginMeasure::getQuBit() const
{
    return m_pTargetQubit;
}

CBit * OriginMeasure::getCBit() const
{
    return m_pCBit;
}


qmap_size_t OriginMeasure::getPosition() const
{
    return m_stPosition;
}

void OriginMeasure::setPosition(qmap_size_t stPositio)
{
    m_stPosition = stPositio;
}

REGISTER_MEASURE(OriginMeasure);
