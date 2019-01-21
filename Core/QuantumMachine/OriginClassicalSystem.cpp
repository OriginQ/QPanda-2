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
#include <sstream>
using namespace std;

OriginCBit::OriginCBit(string name)
    :name(name),
    bOccupancy(false)
{
}

template<typename _Ty>
static string obj2str(_Ty& obj)
{
    stringstream ss;
    ss << obj;
    return ss.str();
}

OriginCMem::OriginCMem(size_t maxMem)
{    
    for (auto i = 0U; i < maxMem; ++i)
    {        
        auto _New_CBit =
            CBitFactory::GetFactoryInstance().
            CreateCBitFromName("c"+obj2str(i));
        vecBit.push_back(_New_CBit);
    }
}

CBit *OriginCMem::Allocate_CBit()
{
    for (auto iter = vecBit.begin(); iter!=vecBit.end(); ++iter)
    {
        if (!(*iter)->getOccupancy())
        {
            (*iter)->setOccupancy(true);
            return *iter;
        }
    }
    return nullptr;
}

CBit *OriginCMem::Allocate_CBit(size_t stCBitNum)
{
    for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
    {
        string sCBitName = "c" + to_string(stCBitNum);
        if (sCBitName == (*iter)->getName())
        {
            (*iter)->setOccupancy(true);
            return *iter;
        }
    }
    return nullptr;
}

size_t OriginCMem::getMaxMem() const
{
    return vecBit.size();
}

size_t OriginCMem::getIdleMem() const
{
    size_t idlenum = 0;
    for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
    {
        if (!(*iter)->getOccupancy())
        {
            idlenum++;
        }
    }
    return idlenum;
}

void OriginCMem::Free_CBit(CBit *cbit)
{
    for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
    {
        if ((*iter)==cbit)
        {
            if (!(*iter)->getOccupancy())
            {
                QCERR("CMem duplicate free");
                throw runtime_error("CMem duplicate free");
            }
            else
            {
                ((*iter)->setOccupancy(false));
                return;
            }
        }
    }
    // if not any matched
    QCERR("Cbit argument error");
    throw invalid_argument("Cbit argument error");
}

void OriginCMem::clearAll()
{
    for (auto iter = vecBit.begin();
        iter != vecBit.end();
        ++iter
        )
    {
        delete (*iter);
        *iter = nullptr;
    }
}
OriginCMem::~OriginCMem()
{
    for_each(vecBit.begin(), vecBit.end(), [](CBit* _s) {delete _s; });
}

