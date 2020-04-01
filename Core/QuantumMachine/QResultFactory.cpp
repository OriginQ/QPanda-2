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
#include "QResultFactory.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "OriginQuantumMachine.h"
using namespace std;
USING_QPANDA
/* QResult Factory */
QResultFactory::
QResultFactory()
{ }

QResult * QResultFactory::GetEmptyQResult()
{
    if (_QResult_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = ConfigMap::getInstance()["QResult"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _QResult_Constructor.find(sClassName);

    if (aiter == _QResult_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second();
}

void QResultFactory::registerclass(string& sClassName, constructor_t
    _Constructor)
{
    _QResult_Constructor.insert(make_pair(sClassName, _Constructor));
}

QResultFactory& QResultFactory::GetFactoryInstance()
{
    static QResultFactory fac;
    return fac;
}

QResultFactoryHelper::
QResultFactoryHelper(string sClassName, constructor_t _Constructor)
{
    auto &fac = QResultFactory::GetFactoryInstance();
    fac.registerclass(sClassName, _Constructor);
}
REGISTER_QRES_NAME(OriginQResult);
