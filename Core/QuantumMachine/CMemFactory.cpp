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
#include "CMemFactory.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "OriginQuantumMachine.h"
using namespace std;
USING_QPANDA
/* CMem Factory*/
CMemFactory::CMemFactory()
{
}

CMem * CMemFactory::GetInstanceFromSize(size_t _Size)
{
    if (_CMem_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = ConfigMap::getInstance()["CMem"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CMem_Constructor.find(sClassName);

    if (aiter == _CMem_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(_Size);
}

CMemFactoryHelper::CMemFactoryHelper
(string sClassName, size_constructor_t _Constructor)
{
    auto &fac = CMemFactory::GetFactoryInstance();
    fac.registerclass_size_(sClassName, _Constructor);
}

void CMemFactory::
registerclass_size_(string & sClassName, size_constructor_t constructor)
{
    _CMem_Constructor.insert(make_pair(sClassName, constructor));
}

CMemFactory&
CMemFactory::GetFactoryInstance()
{
    static CMemFactory fac;
    return fac;
}
REGISTER_CMEM_SIZE_(OriginCMem);