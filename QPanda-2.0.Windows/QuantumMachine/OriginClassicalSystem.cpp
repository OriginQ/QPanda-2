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
	:name(name)
{
	bOccupancy = false;
}

//REGISTER_CBIT_NAME_(OriginCBit)

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
			Factory::CBitFactory::GetFactoryInstance().
			CreateCBitFromName("c"+obj2str(i));
		vecBit.push_back(_New_CBit);
	}
}

IMPLEMENTATION CBit * OriginCMem::Allocate_CBit()
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

IMPLEMENTATION size_t OriginCMem::getMaxMem() const
{
	return vecBit.size();
}

IMPLEMENTATION size_t OriginCMem::getIdleMem() const
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

IMPLEMENTATION void OriginCMem::Free_CBit(CBit *cbit)
{
	for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
	{
		if ((*iter)==cbit)
		{
			if (!(*iter)->getOccupancy())
			{
				// if duplicate free
				throw(exception());
			}
			else
			{
				((*iter)->setOccupancy(false));
                return;
			}
		}
	}
	// if not any matched
	throw(exception());
}

IMPLEMENTATION void OriginCMem::clearAll()
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
//REGISTER_CMEM_SIZE_(OriginCMem)

