/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.

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
#include "ClassicalProgam.h"
#include <type_traits>
#include "Utilities/ConfigMap.h"
USING_QPANDA
using namespace std;
OriginClassicalProg::OriginClassicalProg(ClassicalCondition & classical_prog)
{
    m_node_type = CLASS_COND_NODE;
    m_stPosition = -1;
    m_expr = classical_prog.getExprPtr();
}

OriginClassicalProg::~OriginClassicalProg()
{
    m_expr.reset();
}

NodeType OriginClassicalProg::getNodeType() const 
{
    return m_node_type;
}

qmap_size_t OriginClassicalProg::getPosition() const
{
    return m_stPosition;
}

void OriginClassicalProg::setPosition(qmap_size_t postion)
{
    m_stPosition = postion;
}

cbit_size_t OriginClassicalProg::eval()
{
    if (nullptr == m_expr)
    {
        QCERR("m_expr nullptr");
        throw runtime_error("m_expr nullptr");
    }

    return m_expr->eval();
}


NodeType ClassicalProg::getNodeType() const
{
    auto temp = dynamic_cast<QNode *>(m_node);
    if (nullptr == temp)
    {
        QCERR("m_node type error");
        throw runtime_error("m_node type error");
    }

    return temp->getNodeType();
}

qmap_size_t ClassicalProg::getPosition() const
{
    auto temp = dynamic_cast<QNode *>(m_node);
    if (nullptr == temp)
    {
        QCERR("m_node type error");
        throw runtime_error("m_node type error");
    }

    return temp->getPosition();
}

cbit_size_t ClassicalProg::eval()
{
    if (nullptr == m_node)
    {
        QCERR("m_node nullptr");
        throw runtime_error("m_node nullptr");
    }

    return m_node->eval();
}

ClassicalProg::ClassicalProg(ClassicalCondition & classical_cond)
{
    auto sClasNname = ConfigMap::getInstance()["ClassicalProg"];
    auto aMeasure = ClassicalProgFactory::getInstance().
                    getClassicalProgm(sClasNname,classical_cond);
    auto temp = dynamic_cast<QNode *>(aMeasure);

    auto postion = QNodeMap::getInstance().pushBackNode(temp);
    temp->setPosition(postion);
    if (!QNodeMap::getInstance().addNodeRefer(postion))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }


    m_node = aMeasure;
}

ClassicalProg::ClassicalProg(const ClassicalProg & old_prog)
{
    auto position = old_prog.getPosition();
    auto aiter = QNodeMap::getInstance().getNode(position);
    if (aiter == nullptr)
    {
        QCERR("Cannot find classical prog");
        throw invalid_argument("Cannot find classical prog");
    }


    m_node = dynamic_cast<AbstractClassicalProg *>(aiter);
    if (!QNodeMap::getInstance().addNodeRefer(position))
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

ClassicalProg::~ClassicalProg()
{
    QNodeMap::getInstance().deleteNode(getPosition());
}

void ClassicalProgFactory::registClass(string name, CreateClassicalQProgram method)
{
    if ((name.size() <= 0) || (nullptr == method))
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
    m_ProgMap.insert(pair<string, CreateClassicalQProgram>(name, method));
}

AbstractClassicalProg * ClassicalProgFactory::getClassicalProgm(std::string &name,
    ClassicalCondition & cc)
{
    if (name.size() <= 0)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    auto aiter = m_ProgMap.find(name);
    if (aiter != m_ProgMap.end())
    {
        return aiter->second(cc);
    }
    return nullptr;
}
REGISTER_CLASSICAL_PROGRAM(OriginClassicalProg)
