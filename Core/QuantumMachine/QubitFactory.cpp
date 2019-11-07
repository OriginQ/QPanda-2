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
#include "QubitFactory.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "OriginQuantumMachine.h"
using namespace std;
USING_QPANDA
/* 5. Qubit Factory */
QubitFactory::
QubitFactory()
{
}

/* Factory Helper Class */
QubitFactoryHelper::QubitFactoryHelper(string sClassName, constructor_t _Constructor)
{
    auto &fac = QubitFactory::GetFactoryInstance();
    fac.registerclass(sClassName, _Constructor);
}

QubitFactory & QubitFactory::GetFactoryInstance()
{
    static QubitFactory fac;
    return fac;
}

/* Factory Get Instance */
Qubit * QubitFactory::GetInstance(PhysicalQubit *_physQ)
{
    // I think the last Constructor that user push
    // to the top of the stack is the user defined
    // class. So only Get the top is ok

    auto sClassName = ConfigMap::getInstance()["Qubit"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _Qubit_Constructor.find(sClassName);

    if (aiter == _Qubit_Constructor.end())
    {
        return nullptr;
    }

    return aiter->second(_physQ);
}

void QubitFactory::registerclass(string &sClassName, constructor_t constructor)
{
    // Only push the Constructor to the top
    _Qubit_Constructor.insert(make_pair(sClassName, constructor));
}

REGISTER_QUBIT(OriginQubit);





