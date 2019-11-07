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
#include "ClassicalProgram.h"
#include <type_traits>
#include "Core/Utilities/QProgInfo/ConfigMap.h"
USING_QPANDA
using namespace std;
OriginClassicalProg::OriginClassicalProg(ClassicalCondition & classical_prog)
{
    m_node_type = CLASS_COND_NODE;
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
    if (!m_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    auto temp = dynamic_pointer_cast<QNode >(m_node);
    if (nullptr == temp)
    {
        QCERR("m_node type error");
        throw runtime_error("m_node type error");
    }

    return temp->getNodeType();
}


std::shared_ptr<AbstractClassicalProg> ClassicalProg::getImplementationPtr()
{
    if (!m_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_node;
}

cbit_size_t ClassicalProg::eval()
{
    if (!m_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_node->eval();
}

std::shared_ptr<CExpr> ClassicalProg::getExpr()
{
    if (!m_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_node->getExpr();
}

ClassicalProg::ClassicalProg(ClassicalCondition & classical_cond)
{
    auto sClasNname = ConfigMap::getInstance()["ClassicalProg"];
    auto aMeasure = ClassicalProgFactory::getInstance().
                    getClassicalProgm(sClasNname,classical_cond);

    m_node.reset(aMeasure);
}

ClassicalProg::ClassicalProg(shared_ptr<AbstractClassicalProg>  node)
{
    if (!node)
    {
        QCERR("node is null shared_ptr");
        throw invalid_argument("node is null shared_ptr");
    }

    m_node = node;
}

ClassicalProg::ClassicalProg(const ClassicalProg & old_prog)
{
    m_node = old_prog.m_node;
}

ClassicalProg::~ClassicalProg()
{
    m_node.reset();
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
