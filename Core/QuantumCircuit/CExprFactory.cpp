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

#include "CExprFactory.h"
#include "OriginClassicalExpression.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "OriginQuantumMachine.h"

USING_QPANDA
using namespace std;
CExprFactory::CExprFactory()
{
}

CExprFactory&
CExprFactory::GetFactoryInstance()
{
    static CExprFactory fac;
    return fac;
}


CExpr * CExprFactory::GetCExprByCBit(CBit *bit)
{
    if (_CExpr_CBit_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = ConfigMap::getInstance()["CExpr"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CExpr_CBit_Constructor.find(sClassName);

    if (aiter == _CExpr_CBit_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(bit);

}

CExpr * CExprFactory::GetCExprByValue(cbit_size_t value)
{
    if (_CExpr_CBit_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = ConfigMap::getInstance()["CExpr"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CExpr_Value_Constructor.find(sClassName);

    if (aiter == _CExpr_Value_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(value);
}

void CExprFactory::registerclass_Value_(string & sClassName,
    value_constructor_t _Constructor)
{
    _CExpr_Value_Constructor.insert(make_pair(sClassName, _Constructor));
}

CExpr * CExprFactory::GetCExprByOperation(
    CExpr *leftexpr,
    CExpr *rightexpr,
    int op)
{
    if (_CExpr_Operator_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = ConfigMap::getInstance()["CExpr"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CExpr_Operator_Constructor.find(sClassName);

    if (aiter == _CExpr_Operator_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(leftexpr, rightexpr, op);

}

void CExprFactory::registerclass_CBit_(
    string &sClassName, cbit_constructor_t _Constructor
)
{
    _CExpr_CBit_Constructor.insert(make_pair(sClassName, _Constructor));
}

CExprFactoryHelper::CExprFactoryHelper(
    string sClassName,
    cbit_constructor_t _Constructor
)
{
    auto &fac =
        CExprFactory::GetFactoryInstance();
    fac.registerclass_CBit_(sClassName, _Constructor);
}

CExprFactoryHelper::CExprFactoryHelper(string sClassName,
    value_constructor_t _Constructor)
{
    auto &fac =
        CExprFactory::GetFactoryInstance();
    fac.registerclass_Value_(sClassName, _Constructor);
}

void CExprFactory::registerclass_operator_
(string &sClassName, operator_constructor_t _Constructor)
{
    _CExpr_Operator_Constructor.insert(make_pair(sClassName, _Constructor));
}

CExprFactoryHelper::CExprFactoryHelper(
    string sClassName,
    operator_constructor_t _Constructor
)
{
    auto &fac =
        CExprFactory::GetFactoryInstance();
    fac.registerclass_operator_(sClassName, _Constructor);
}

REGISTER_CEXPR(OriginCExpr);




