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
#include "QubitPoolFactory.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "OriginQuantumMachine.h"
USING_QPANDA
using namespace std;
/* Qubit Pool Factory */
QubitPoolFactory::QubitPoolFactory()
{
}

QubitPoolFactory &
QubitPoolFactory::GetFactoryInstance()
{
    static QubitPoolFactory fac;
    return fac;
}

/* Factory Get Instance */
QubitPool * QubitPoolFactory::GetPoolWithoutTopology(size_t size)
{
    // I think the last Constructor that user push
    // to the top of the stack is the user defined
    // class. So only Get the top is ok

    if (_Qubit_Pool_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = ConfigMap::getInstance()["QubitPool"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _Qubit_Pool_Constructor.find(sClassName);

    if (aiter == _Qubit_Pool_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(size);
}

void QubitPoolFactory::registerclass_size_(string & sClassName,
    size_constructor_t constructor)
{
    _Qubit_Pool_Constructor.insert(make_pair(sClassName, constructor));
}

/* Factory Helper Class */
QubitPoolFactoryHelper::QubitPoolFactoryHelper(string  sClassName,
    size_constructor_t constructor)
{
    // Only push the Constructor to the top
    auto &fac = QubitPoolFactory::GetFactoryInstance();
    fac.registerclass_size_(sClassName, constructor);
}

REGISTER_QUBIT_POOL_SIZE_(OriginQubitPool);
REGISTER_QUBIT_POOL_SIZE_(OriginQubitPoolv2);
