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

#include "Core/Core.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "ClassicalProgram.h"
using namespace QGATE_SPACE;
using namespace std;
USING_QPANDA
QProg  QPanda::CreateEmptyQProg()
{
    QProg temp;
    return temp;
}

QProg QPanda::createEmptyQProg()
{
    QProg temp;
    return temp;
}

QProg::QProg(std::shared_ptr<AbstractQuantumProgram> node)
{
    if (!node)
    {
        QCERR("node is null shared_ptr");
        throw invalid_argument("node is null shared_ptr");
    }

    m_quantum_program = node;
}

QProg::QProg()
{
    auto class_name = ConfigMap::getInstance()["QProg"];
    auto qprog = QuantumProgramFactory::getInstance().getQuantumQProg(class_name);
    m_quantum_program.reset(qprog);
}

QProg::QProg(const QProg &old_qprog)
{
    m_quantum_program = old_qprog.m_quantum_program;
}

QProg::QProg(QProg &other)
{
	m_quantum_program = other.m_quantum_program;
}


QProg::QProg(std::shared_ptr<QNode> pnode)
    :QProg()
{
    if (!pnode)
    {
        throw std::runtime_error("node is null");
    }
    m_quantum_program->pushBackNode(pnode);
}

QProg::QProg(ClassicalCondition &node)
    :QProg()
{
    ClassicalProg tmp(node);
    m_quantum_program->pushBackNode(dynamic_pointer_cast<QNode>(tmp.getImplementationPtr()));
}


QProg::~QProg()
{
    m_quantum_program.reset();
}

std::shared_ptr<AbstractQuantumProgram> QProg::getImplementationPtr()
{
    return m_quantum_program;
}

void QProg::pushBackNode(std::shared_ptr<QNode> node)
{
    if (!node)
    {
        QCERR("node is null");
        throw runtime_error("node is null");
    }
    m_quantum_program->pushBackNode(node);
}

NodeIter QProg::getFirstNodeIter()
{
    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_quantum_program->getFirstNodeIter();
}

NodeIter  QProg::getLastNodeIter()
{
    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_quantum_program->getLastNodeIter();
}

NodeIter QProg::getEndNodeIter()
{
    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_quantum_program->getEndNodeIter();
}

NodeIter QProg::getHeadNodeIter()
{
    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_quantum_program->getHeadNodeIter();
}

NodeIter QProg::insertQNode(const NodeIter & iter,shared_ptr<QNode> node)
{
    if (nullptr == node)
    {
        QCERR("node is nullptr");
        throw runtime_error("node is nullptr");
    }

    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_quantum_program->insertQNode(iter, node);
}

NodeIter QProg::deleteQNode(NodeIter & iter)
{
    return m_quantum_program->deleteQNode(iter);
}

NodeType QProg::getNodeType() const
{
    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return dynamic_pointer_cast<QNode>(m_quantum_program)->getNodeType();
}

void QProg::clear()
{
    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_quantum_program->clear();
}

template <>
QProg & QProg::operator<<<ClassicalCondition>(ClassicalCondition cc)
{
    ClassicalProg temp(cc);
    pushBackNode(dynamic_pointer_cast<QNode>(temp.getImplementationPtr()));
    return *this;
}

OriginProgram::~OriginProgram()
{
}

OriginProgram::OriginProgram()
{ 
}

NodeType OriginProgram::getNodeType() const
{
    return m_node_type;
}

REGISTER_QPROGRAM(OriginProgram);

void QuantumProgramFactory::registClass(string name, CreateQProgram method)
{
    if ((name.size() <= 0) || (nullptr == method))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    m_qprog_map.insert(pair<string, CreateQProgram>(name, method));
}

AbstractQuantumProgram * QuantumProgramFactory::getQuantumQProg(std::string & name)
{
    if (name.size() <= 0)
    {
        QCERR("param error");
        throw runtime_error("param error");
    }
    auto aiter = m_qprog_map.find(name);
    if (aiter != m_qprog_map.end())
    {
        return aiter->second();
    }
    return nullptr;
}

