/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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
#include "Core/Utilities/QProgInfo/ConfigMap.h"
USING_QPANDA
using namespace std;
QMeasure  QPanda::Measure(Qubit * target_qubit,ClassicalCondition  classical_cond)
{
    auto target_cbit = classical_cond.getExprPtr()->getCBit();
    if (nullptr == target_cbit)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
    QMeasure measure(target_qubit, target_cbit);
    return measure;
}

QMeasure QPanda::Measure(int qaddr, int  classical_addr)
{
    auto target_cbit = OriginCMem::get_instance()->get_cbit_by_addr(classical_addr);
    auto target_qubit = OriginQubitPool::get_instance()->get_qubit_by_addr(qaddr);
	if (nullptr == target_cbit
        || nullptr ==  target_qubit)
	{
		QCERR("param error");
		throw invalid_argument("param error");
	}
	QMeasure measure(target_qubit, target_cbit);
	return measure;
}

QMeasure::QMeasure(const QMeasure & old_measure)
{
    m_measure = old_measure.m_measure;
}

QMeasure::QMeasure(std::shared_ptr<AbstractQuantumMeasure> node)
{
    if (!node)
    {
        QCERR("this shared_ptr is null");
        throw invalid_argument("this shared_ptr is null");
    }
    m_measure = node;
}

QMeasure::QMeasure(Qubit * qubit, CBit * cbit)
{
    auto class_name = ConfigMap::getInstance()["QMeasure"];
    auto measure = QuantumMeasureFactory::getInstance().getQuantumMeasure(class_name, qubit, cbit);
    m_measure.reset(measure);
}

std::shared_ptr<AbstractQuantumMeasure> QMeasure::getImplementationPtr()
{
    if (!m_measure)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_measure;
}

QMeasure::~QMeasure()
{
    m_measure.reset();
}

Qubit * QMeasure::getQuBit() const
{
    if (nullptr == m_measure)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_measure->getQuBit();
}

CBit * QMeasure::getCBit() const
{
    if (!m_measure)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_measure->getCBit();
}


NodeType QMeasure::getNodeType() const
{
    if (!m_measure)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return (dynamic_pointer_cast<QNode >(m_measure))->getNodeType();
}

void QuantumMeasureFactory::registClass(string name, CreateMeasure method)
{
    m_measureMap.insert(pair<string, CreateMeasure>(name, method));
}

AbstractQuantumMeasure * QuantumMeasureFactory::getQuantumMeasure(std::string & classname, Qubit * pQubit, CBit * pCBit)
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
    return m_node_type;
}

OriginMeasure::OriginMeasure(Qubit * qubit, CBit * cbit):
    m_target_qubit(qubit),
    m_target_cbit(cbit),
    m_node_type(MEASURE_GATE)
{
}

Qubit * OriginMeasure::getQuBit() const
{
    return m_target_qubit;
}

CBit * OriginMeasure::getCBit() const
{
    return m_target_cbit;
}

REGISTER_MEASURE(OriginMeasure);
