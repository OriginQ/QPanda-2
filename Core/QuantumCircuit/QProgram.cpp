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

QProg::~QProg()
{
    m_quantum_program.reset();
}

std::shared_ptr<QNode> QProg::getImplementationPtr()
{
    return dynamic_pointer_cast<QNode>(m_quantum_program);
}

void QProg :: pushBackNode(QNode * node)
{
    if (!m_quantum_program)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    m_quantum_program->pushBackNode(node);
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

NodeIter QProg::insertQNode(NodeIter & iter, QNode * node)
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
    auto node = dynamic_cast<QNode *>(&temp);
    if (nullptr == node)
    {
        QCERR("node is not base of ClassicalProg");
        throw qprog_construction_fail("node is not base of ClassicalProg");
    }
    pushBackNode(node);
    return *this;
}


OriginProgram::~OriginProgram()
{
    Item *temp;

    while (m_head != nullptr)
    {
        m_head->setPre(nullptr);
        temp = m_head;
        m_head = m_head->getNext();
        delete temp;
    }
    m_head = nullptr;
    m_end = nullptr;
}

OriginProgram::OriginProgram() : m_head(nullptr), m_end(nullptr), m_node_type(PROG_NODE)
{ }

void OriginProgram::pushBackNode(QNode * node)
{
    if (nullptr == node)
    {
        QCERR("node is null");
        throw runtime_error("node is null");
    }
    auto temp = node->getImplementationPtr();
    pushBackNode(temp);
}

void OriginProgram::pushBackNode(std::shared_ptr<QNode> node)
{
    if (!node)
    {
        QCERR("node is null");
        throw runtime_error("node is null");
    }
    WriteLock wl(m_sm);
    if (nullptr == m_head)
    {
        Item *iter = new OriginItem();
        iter->setNext(nullptr);
        iter->setPre(nullptr);
        iter->setNode(node);
        m_head = iter;
        m_end = iter;
    }
    else
    {
        Item *iter = new OriginItem();
        iter->setNext(nullptr);
        iter->setPre(m_end);
        m_end->setNext(iter);
        m_end = iter;
        iter->setNode(node);
    }
}

NodeIter OriginProgram::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_head);
    return temp;
}

NodeIter OriginProgram::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_end);
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
    return m_node_type;
}

void OriginProgram::clear()
{
    WriteLock wl(m_sm);
    Item *temp;
    if (m_head != nullptr)
    {
        while (m_head != m_end)
        {
            temp = m_head;
            m_head = m_head->getNext();
            m_head->setPre(nullptr);
            delete temp;
        }
        delete m_head;
        m_head = nullptr;
        m_end = nullptr;
    }
}

void OriginProgram::execute(QPUImpl * quantum_gates, QuantumGateParam * param)
{
    auto aiter = getFirstNodeIter();
    if (nullptr == *aiter)
    {
        return ;
    }

    for (; aiter != getEndNodeIter(); ++aiter)
    {
        auto node = *aiter;
        if (nullptr == node)
        {
            QCERR("node is null");
            std::runtime_error("node is null");
        }

        node->execute(quantum_gates, param);
    }
}

NodeIter OriginProgram::insertQNode(NodeIter & perIter, QNode * node)
{
    ReadLock * rl = new ReadLock(m_sm);
    Item * perItem = perIter.getPCur();
    if (nullptr == perItem)
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
        if (perItem == aiter.getPCur())
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
    Item *curItem = new OriginItem();
    auto ptemp = node->getImplementationPtr();
    curItem->setNode(ptemp);

    if (nullptr != perItem->getNext())
    {
        perItem->getNext()->setPre(curItem);
        curItem->setNext(perItem->getNext());
        perItem->setNext(curItem);
        curItem->setPre(perItem);
    }
    else
    {
        perItem->setNext(curItem);
        curItem->setPre(perItem);
        curItem->setNext(nullptr);
        m_end = curItem;
    }
    NodeIter temp(curItem);
    return temp;
}

NodeIter OriginProgram::deleteQNode(NodeIter & target_iter)
{
    ReadLock *rl = new ReadLock(m_sm);
    Item * target_item = target_iter.getPCur();
    if (nullptr == target_item)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }


    if (nullptr == m_head)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aiter = this->getFirstNodeIter();
    for (; aiter != this->getEndNodeIter(); aiter++)
    {
        if (target_item == aiter.getPCur())
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

    if (m_head == target_item)
    {
        if (m_head == m_end)
        {
            delete target_item;
            target_iter.setPCur(nullptr);
            m_head = nullptr;
            m_end = nullptr;
        }
        else
        {
            m_head = target_item->getNext();
            m_head->setPre(nullptr);
            delete target_item;
            target_iter.setPCur(nullptr);
        }

        NodeIter temp(m_head);
        return temp;
    }

    if (m_end == target_item)
    {
        Item * pPerItem = target_item->getPre();
        if (nullptr == pPerItem)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        pPerItem->setNext(nullptr);
        delete(target_item);
        target_iter.setPCur(nullptr);
        NodeIter temp(pPerItem);
        return temp;
    }

    Item * perItem = target_item->getPre();
    if (nullptr == perItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    perItem->setNext(nullptr);
    Item * nextItem = target_item->getNext();
    if (nullptr == perItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    perItem->setNext(nextItem);
    nextItem->setPre(perItem);
    delete target_item;
    target_iter.setPCur(nullptr);

    NodeIter temp(perItem);
    return temp;
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

