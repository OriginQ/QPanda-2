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

#include "OriginQuantumMachine.h"
#include "PhysicalQubitFactory.h"
#include "QubitFactory.h"
#include <algorithm>
USING_QPANDA
using namespace std;
OriginQubitPool::OriginQubitPool(size_t maxQubit)
{
    for (auto i = 0U; i < maxQubit; ++i)
    {
        auto _New_Physical_Qubit =
            PhysicalQubitFactory::GetFactoryInstance().
            GetInstance();
        vecQubit.push_back(_New_Physical_Qubit);
        _New_Physical_Qubit->setQubitAddr(i);
    }
}

void OriginQubitPool::clearAll()
{
    for (auto iter = vecQubit.begin(); iter != vecQubit.end();)
    {
        delete *iter;
        iter = vecQubit.erase(iter);
    }
}

size_t OriginQubitPool::getMaxQubit() const
{
    return vecQubit.size();
}

size_t OriginQubitPool::getIdleQubit() const
{
    size_t retIdle = 0;
    for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
    {
        if (!(*iter)->getOccupancy())
        {
            retIdle++;
        }
    }
    return retIdle;
}

Qubit * OriginQubitPool::allocateQubit()
{
    for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
    {
        if (!(*iter)->getOccupancy())
        {
            (*iter)->setOccupancy(true);
            return QubitFactory::GetFactoryInstance().
                GetInstance(*iter);
            
        }
    }
    return nullptr;
}

Qubit * OriginQubitPool::allocateQubitThroughPhyAddress(size_t stQubitAddr)
{
    for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
    {
        if ((*iter)->getQubitAddr() == stQubitAddr)
        {
            (*iter)->setOccupancy(true);
            return QubitFactory::GetFactoryInstance().
                GetInstance(*iter);
        }
    }
    return nullptr;
}

Qubit * OriginQubitPool::allocateQubitThroughVirAddress(size_t qubit_num)
{
    if (qubit_num >= vecQubit.size())
    {
        return nullptr;
    }

	vecQubit[qubit_num]->setOccupancy(true);
    return  QubitFactory::GetFactoryInstance().
        GetInstance(vecQubit[qubit_num]);
}

void OriginQubitPool::Free_Qubit(Qubit* _Qubit)
{
    auto ptPhysQ = _Qubit->getPhysicalQubitPtr();
    auto iter = find(vecQubit.begin(), vecQubit.end(), ptPhysQ);
    if (iter == vecQubit.end())
    {
        QCERR("QubitPool duplicate free");
        throw runtime_error("QubitPool duplicate free");
    }
    else
    {
        (*iter)->setOccupancy(false);
    }
}

size_t OriginQubitPool::getPhysicalQubitAddr(Qubit *qubit)
{
    if (nullptr == qubit)
    {
        QCERR("qubit is nullptr");
        throw invalid_argument("qubit is nullptr");
    }
    for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
    {
        auto physAddr = qubit->getPhysicalQubitPtr();
        if (*iter == physAddr)
        {
            return (*iter)->getQubitAddr();
        }
    }
    QCERR("qubit argument error");
    throw invalid_argument("qubit argument error");
}

size_t OriginQubitPool::getVirtualQubitAddress(Qubit* qubit) const
{
    if (nullptr == qubit)
    {
        QCERR("qubit is nullptr");
        throw invalid_argument("qubit is nullptr");
    }

    for (size_t i = 0; i<vecQubit.size(); ++i)
    {
        auto physAddr = qubit->getPhysicalQubitPtr();
        if (vecQubit[i] == physAddr)
        {
            return i;
        }
    }
    QCERR("qubit argument error");
    throw invalid_argument("qubit argument error");
}

OriginQubitPool::~OriginQubitPool()
{
    for_each(vecQubit.begin(), vecQubit.end(), [](PhysicalQubit* _s) {delete _s; });
}




