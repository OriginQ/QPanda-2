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
#include "QPanda/QPandaException.h"

QMeasure  Measure(Qubit * targetQuBit, CBit *targetCbit)
{
    QMeasure qMeasure(targetQuBit, targetCbit);
    return qMeasure;
}

QMeasure::QMeasure(const QMeasure & oldMeasure)
{
    m_stPosition = oldMeasure.getPosition();    
    auto aiter = _G_QNodeMap.getNode(m_stPosition);
    if (aiter != nullptr)
        m_pQuantumMeasure = dynamic_cast<AbstractQuantumMeasure *>(aiter);
    else
        throw circuit_not_found_exception("there is not target QNode", false);
    if (!_G_QNodeMap.addNodeRefer(m_stPosition))
        throw exception();
}

QMeasure::QMeasure(Qubit * qbit, CBit * cbit)
{
    string sClasNname = "OriginMeasure";
    auto aMeasure = QuantunMeasureFactory::getInstance().getQuantumMeasure(sClasNname, qbit, cbit);
    auto temp = dynamic_cast<QNode *>(aMeasure);
    m_stPosition = _G_QNodeMap.pushBackNode(temp);
    temp->setPosition(m_stPosition);
    if (!_G_QNodeMap.addNodeRefer(m_stPosition))
        throw exception();
    m_pQuantumMeasure = aMeasure;
}

QMeasure::~QMeasure()
{
    _G_QNodeMap.deleteNode(m_stPosition);
}

Qubit * QMeasure::getQuBit() const
{
    if(nullptr == m_pQuantumMeasure)
        throw  exception();
    return m_pQuantumMeasure->getQuBit();
}

CBit * QMeasure::getCBit() const
{
    if (nullptr == m_pQuantumMeasure)
        throw  exception();
    return m_pQuantumMeasure->getCBit();
}


NodeType QMeasure::getNodeType() const
{
    if (nullptr == m_pQuantumMeasure)
        throw  exception();
    return (dynamic_cast<QNode *>(m_pQuantumMeasure))->getNodeType();
}

QMAP_SIZE QMeasure::getPosition() const
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
        throw circuit_not_found_exception("can not find targit measure class", true);
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


QMAP_SIZE OriginMeasure::getPosition() const
{
    return m_stPosition;
}

void OriginMeasure::setPosition(QMAP_SIZE stPositio)
{
    m_stPosition = stPositio;
}

REGISTER_MEASURE(OriginMeasure);