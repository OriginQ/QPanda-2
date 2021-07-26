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

#include "OriginQuantumMachine.h"
#include <sstream>
using namespace std;
USING_QPANDA
OriginCBit::OriginCBit(string name)
    :name(name),
    bOccupancy(false)
{
	set_val(atoll(name.c_str() + 1));
}

template<typename _Ty>
static string obj2str(_Ty& obj)
{
    stringstream ss;
    ss << obj;
    return ss.str();
}

OriginCMem::OriginCMem()
{    
	size_t maxMem = 29;
    for (auto i = 0U; i < maxMem; ++i)
    {        
        auto _New_CBit =
            CBitFactory::GetFactoryInstance().
            CreateCBitFromName("c"+obj2str(i));
        vecBit.push_back(_New_CBit);
    }
}

size_t OriginCMem::get_capacity()
{
	return  vecBit.size();
}

void OriginCMem::set_capacity(size_t capacity_num)
{
	auto cur_cap = vecBit.size();

	if (capacity_num < cur_cap)
	{
		vecBit.erase(vecBit.begin() + capacity_num, vecBit.end());
	}
	else if(capacity_num > cur_cap)
	{
		for (size_t i = cur_cap; i < capacity_num; ++i)
		{
			auto _New_CBit =
				CBitFactory::GetFactoryInstance().
				CreateCBitFromName("c" + obj2str(i));
			vecBit.push_back(_New_CBit);
		}
	}
	else
	{
		//do nothing
	}
}

CBit* OriginCMem::get_cbit_by_addr(size_t caddr)
{
	std::string str_cbit = "c" + to_string(caddr);
	for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
	{
		if (str_cbit == (*iter)->getName() &&  (*iter)->getOccupancy())
			return CBitFactory::GetFactoryInstance().CreateCBitFromName(str_cbit);
	}

	QCERR("get cbit by address error");
	throw invalid_argument("get cbit by address error");
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

size_t OriginCMem::get_allocate_cbits(std::vector<CBit *>& cbit_vect)
{
	size_t allocate_size = 0;
	for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
	{
		if ((*iter)->getOccupancy())
		{
            allocate_size++;
			auto new_cbit = CBitFactory::GetFactoryInstance().CreateCBitFromName((*iter)->getName());
            cbit_vect.push_back(new_cbit);
		}
	}
	return allocate_size;
}

CBit* OriginCMem::cAlloc()
{
    return Allocate_CBit();
}

CBit* OriginCMem::cAlloc(size_t cbit_num)
{
	return Allocate_CBit(cbit_num);
}

std::vector<ClassicalCondition> OriginCMem::cAllocMany(size_t count)
{
	if (count > getIdleMem())
	{
		QCERR("count > getIdleMem()");
		throw(calloc_fail("count > getIdleMem()"));
	}
	try
	{
		std::vector<ClassicalCondition> cbits_vect;
		for (size_t i = 0; i < count; i++)
		{
			auto cbit = Allocate_CBit();
            cbits_vect.push_back(cbit);
		}
		return cbits_vect;
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw(calloc_fail(e.what()));
	}

}

void OriginCMem::cFree(CBit* cbit)
{
    Free_CBit(cbit);
}

void OriginCMem::cFreeAll(std::vector< CBit* >& cbits_vect)
{
    for (auto it : cbits_vect)
    {
		Free_CBit(it);
    }
}

OriginCMem::~OriginCMem()
{
    for_each(vecBit.begin(), vecBit.end(), [](CBit* _s) {delete _s; });
}



OriginCMemv2::OriginCMemv2(size_t maxMem)
{
	for (auto i = 0U; i < maxMem; ++i)
	{
		auto _New_CBit =
			CBitFactory::GetFactoryInstance().
			CreateCBitFromName("c" + obj2str(i));
		vecBit.push_back(_New_CBit);
	}
}

CBit* OriginCMemv2::Allocate_CBit()
{
	for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
	{
		if (!(*iter)->getOccupancy())
		{
			(*iter)->setOccupancy(true);
			return *iter;
		}
	}
	return nullptr;
}

CBit* OriginCMemv2::Allocate_CBit(size_t stCBitNum)
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

size_t OriginCMemv2::getMaxMem() const
{
	return vecBit.size();
}

size_t OriginCMemv2::getIdleMem() const
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

void OriginCMemv2::Free_CBit(CBit* cbit)
{
	for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
	{
		if ((*iter) == cbit)
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

void OriginCMemv2::clearAll()
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

size_t OriginCMemv2::get_allocate_cbits(std::vector<CBit*>& cbit_vect)
{
	size_t allocate_size = 0;
	for (auto iter = vecBit.begin(); iter != vecBit.end(); ++iter)
	{
		if ((*iter)->getOccupancy())
		{
			allocate_size++;
			auto new_cbit = CBitFactory::GetFactoryInstance().CreateCBitFromName((*iter)->getName());
			cbit_vect.push_back(new_cbit);
		}
	}
	return allocate_size;
}

OriginCMemv2::~OriginCMemv2()
{
	for_each(vecBit.begin(), vecBit.end(), [](CBit* _s) {delete _s; });
}
