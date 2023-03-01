/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

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
OriginQubitPool::OriginQubitPool()
{
	size_t m_maxQubit = 29;
	for (auto i = 0U; i < m_maxQubit; ++i)
	{
		auto _New_Physical_Qubit =
			PhysicalQubitFactory::GetFactoryInstance().
			GetInstance();
		vecQubit.push_back(_New_Physical_Qubit);
		_New_Physical_Qubit->setQubitAddr(i);
	}
}


size_t OriginQubitPool::get_capacity()
{
	return  vecQubit.size();
}

void OriginQubitPool::set_capacity(size_t capacity_num)
{
	auto cur_cap = vecQubit.size();

	if (capacity_num < cur_cap)
	{
		vecQubit.erase(vecQubit.begin() + capacity_num, vecQubit.end());
	}
	else if(capacity_num > cur_cap)
	{
		for (size_t i = cur_cap; i < capacity_num; ++i)
		{
			auto _New_Physical_Qubit =
				PhysicalQubitFactory::GetFactoryInstance().
				GetInstance();
			vecQubit.push_back(_New_Physical_Qubit);
			_New_Physical_Qubit->setQubitAddr(i);
		}
	}
	else
	{
		// do nothing
	}
}


void OriginQubitPool::clearAll()
{
	for (auto iter = vecQubit.begin(); iter != vecQubit.end();)
	{
		delete* iter;
		iter = vecQubit.erase(iter);
	}
}

size_t OriginQubitPool::getMaxQubit() const
{
	return vecQubit.size();
}

size_t OriginQubitPool::get_max_usedqubit_addr() const
{
	size_t max_addr = 0;
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if ((*iter)->getOccupancy())
		{
			int addr = (*iter)->getQubitAddr();
			if (max_addr < addr)
				max_addr = addr;
		}
	}
	return max_addr;
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

Qubit* OriginQubitPool::allocateQubit()
{
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if (!(*iter)->getOccupancy())
		{
			(*iter)->setOccupancy(true);
			return QubitFactory::GetFactoryInstance().GetInstance(*iter);
		}
	}
	return nullptr;
}

Qubit* OriginQubitPool::allocateQubitThroughPhyAddress(size_t stQubitAddr)
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

Qubit* OriginQubitPool::allocateQubitThroughVirAddress(size_t qubit_num)
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

size_t OriginQubitPool::getPhysicalQubitAddr(Qubit* qubit)
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

	for (size_t i = 0; i < vecQubit.size(); ++i)
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

size_t OriginQubitPool::get_allocate_qubits(std::vector<Qubit*>& qubit_vect) const
{
	size_t allocate_size = 0;
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if ((*iter)->getOccupancy())
		{
			qubit_vect.push_back(QubitFactory::GetFactoryInstance().GetInstance(*iter));
			allocate_size++;
		}
	}
	return allocate_size;
}

Qubit* OriginQubitPool::qAlloc()
{
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if (!(*iter)->getOccupancy())
		{
			(*iter)->setOccupancy(true);
			return QubitFactory::GetFactoryInstance().GetInstance(*iter);
		}
	}
	return nullptr;
}

QVec  OriginQubitPool::qAllocMany(size_t qubit_num)
{
	if (qubit_num > getIdleQubit())
	{
		QCERR("qubit_num > idle_qubit");
		throw(qalloc_fail("qubit_num > idle_qubit"));
	}

	QVec qubits_vect;
	for (size_t i = 0; i < qubit_num; i++)
	{
		qubits_vect.push_back(qAlloc());
	}
	return qubits_vect;
}

void OriginQubitPool::qFree(Qubit* qubit)
{
	if (nullptr == qubit)
	{
		QCERR("qubit is nullptr");
		throw invalid_argument("qubit is nullptr");
	}

	auto phy_qubit = qubit->getPhysicalQubitPtr();
	auto iter = find(vecQubit.begin(), vecQubit.end(), phy_qubit);
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

void OriginQubitPool::qFreeAll(QVec& qubits_vect)
{
    for (auto &iter : qubits_vect)
	{
		qFree(iter);
    }
}

void OriginQubitPool::qFreeAll()
{
    for (auto &q : vecQubit)
    {
        if (q->getOccupancy())
        {
            q->setOccupancy(false);
        }
    }
    return ;
}

Qubit * OriginQubitPool::get_qubit_by_addr(size_t qaddr)
{
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		auto phy_addr = (*iter)->getQubitAddr();
		if (phy_addr == qaddr 
			&& (*iter)->getOccupancy())
		{
			return QubitFactory::GetFactoryInstance().GetInstance(*iter);
		}
	}

	QCERR("get qubit by physical address error");
	throw invalid_argument("get qubit by physical address error");
}



OriginQubitPool::~OriginQubitPool()
{
	for_each(vecQubit.begin(), vecQubit.end(), [](PhysicalQubit* _s) {delete _s; });
}



OriginQubitPoolv1::OriginQubitPoolv1(size_t maxQubit)
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

void OriginQubitPoolv1::clearAll()
{
	for (auto iter = vecQubit.begin(); iter != vecQubit.end();)
	{
		delete* iter;
		iter = vecQubit.erase(iter);
	}
}

size_t OriginQubitPoolv1::getMaxQubit() const
{
	return vecQubit.size();
}

size_t OriginQubitPoolv1::get_max_usedqubit_addr() const
{
	size_t max_addr = 0;
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if ((*iter)->getOccupancy())
		{
			int addr = (*iter)->getQubitAddr();
			if (max_addr < addr)
				max_addr = addr;
		}
	}
	return max_addr;
}

size_t OriginQubitPoolv1::getIdleQubit() const
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

Qubit* OriginQubitPoolv1::allocateQubit()
{
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if (!(*iter)->getOccupancy())
		{
			(*iter)->setOccupancy(true);
			return QubitFactory::GetFactoryInstance().GetInstance(*iter);
		}
	}
	return nullptr;
}

Qubit* OriginQubitPoolv1::allocateQubitThroughPhyAddress(size_t stQubitAddr)
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

Qubit* OriginQubitPoolv1::allocateQubitThroughVirAddress(size_t qubit_num)
{
	if (qubit_num >= vecQubit.size())
	{
		return nullptr;
	}

	vecQubit[qubit_num]->setOccupancy(true);
	return  QubitFactory::GetFactoryInstance().
		GetInstance(vecQubit[qubit_num]);
}

void OriginQubitPoolv1::Free_Qubit(Qubit* _Qubit)
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

size_t OriginQubitPoolv1::getPhysicalQubitAddr(Qubit* qubit)
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

size_t OriginQubitPoolv1::getVirtualQubitAddress(Qubit* qubit) const
{
	if (nullptr == qubit)
	{
		QCERR("qubit is nullptr");
		throw invalid_argument("qubit is nullptr");
	}

	for (size_t i = 0; i < vecQubit.size(); ++i)
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

size_t OriginQubitPoolv1::get_allocate_qubits(std::vector<Qubit*>& qubit_vect) const
{
	size_t allocate_size = 0;
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if ((*iter)->getOccupancy())
		{
			qubit_vect.push_back(QubitFactory::GetFactoryInstance().GetInstance(*iter));
			allocate_size++;
		}
	}
	return allocate_size;
}

OriginQubitPoolv1::~OriginQubitPoolv1()
{
	for_each(vecQubit.begin(), vecQubit.end(), [](PhysicalQubit* _s) {delete _s; });
}

OriginQubitPoolv2::OriginQubitPoolv2(size_t maxQubit) {
	for (auto i = 0U; i < maxQubit; ++i)
	{
		auto _New_Physical_Qubit =
			PhysicalQubitFactory::GetFactoryInstance().
			GetInstance();
		vecQubit.push_back(_New_Physical_Qubit);
		_New_Physical_Qubit->setQubitAddr(i);
	}
}

OriginQubitPoolv2::~OriginQubitPoolv2()
{
	clearAll();
}

void OriginQubitPoolv2::clearAll()
{
	for (auto pq : vecQubit) {
		delete pq;
	}
	vecQubit.clear();
	for (auto q : allocated_qubit) {
		delete q.first;
	}
	allocated_qubit.clear();
}

size_t OriginQubitPoolv2::getMaxQubit() const
{
	return vecQubit.size();
}

size_t OriginQubitPoolv2::get_max_usedqubit_addr() const
{
	size_t max_addr = 0;
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if ((*iter)->getOccupancy())
		{
			int addr = (*iter)->getQubitAddr();
			if (max_addr < addr)
				max_addr = addr;
		}
	}
	return max_addr;
}

size_t OriginQubitPoolv2::getIdleQubit() const
{
	return vecQubit.size() - allocated_qubit.size();
}

Qubit* OriginQubitPoolv2::allocateQubit() {

	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if (!(*iter)->getOccupancy())
		{
			PhysicalQubit* valid_qubit = *iter;
			valid_qubit->setOccupancy(true);
			// find if it is allocated
			for (auto& q : allocated_qubit) {
				// allocated qubit
				if (q.first->getPhysicalQubitPtr() == valid_qubit) {
					q.second++;
					return q.first;
				}
			}
			// not allocated
			auto qubit_ptr = QubitFactory::GetFactoryInstance().
				GetInstance(*iter);
			allocated_qubit.insert(make_pair(qubit_ptr, 1));
			return qubit_ptr;
		}
	}
	return nullptr;
}

Qubit* OriginQubitPoolv2::allocateQubitThroughPhyAddress(size_t pos)
{
	// if allocated then return
	for (auto& q : allocated_qubit) {
		if (q.first->getPhysicalQubitPtr()->getQubitAddr() == pos) {
			q.first->getPhysicalQubitPtr()->setOccupancy(true);
			q.second++;
			return q.first;
		}
	}

	// not allocated
	for (auto iter = vecQubit.begin(); iter != vecQubit.end(); ++iter)
	{
		if ((*iter)->getQubitAddr() == pos)
		{
			PhysicalQubit* valid_qubit = *iter;
			valid_qubit->setOccupancy(true);
			auto qubit_ptr = QubitFactory::GetFactoryInstance().
				GetInstance(*iter);
			allocated_qubit.insert(make_pair(qubit_ptr, 1));
			return qubit_ptr;
		}
	}
	return nullptr;
}

Qubit* OriginQubitPoolv2::allocateQubitThroughVirAddress(size_t qubit_num)
{
	if (qubit_num >= vecQubit.size())
	{
		return nullptr;
	}
	PhysicalQubit* valid_qubit = vecQubit[qubit_num];
	valid_qubit->setOccupancy(true);

	for (auto& q : allocated_qubit) {
		if (q.first->getPhysicalQubitPtr() == valid_qubit) {
			q.second++;
			return q.first;
		}
	}
	auto qubit_ptr = QubitFactory::GetFactoryInstance().
		GetInstance(valid_qubit);
	allocated_qubit.insert(make_pair(qubit_ptr, 1));
	return qubit_ptr;

}

void OriginQubitPoolv2::Free_Qubit(Qubit* _Qubit)
{
	if (nullptr == _Qubit)
	{
		QCERR("qubit ptr is null");
		throw runtime_error("qubit ptr is null");
	}
	auto allocated_iter = allocated_qubit.begin();
	for (; allocated_iter != allocated_qubit.end(); ++allocated_iter)
	{
		if (allocated_iter->first == _Qubit)
		{
			if (allocated_iter->second <= 0)
			{
				QCERR("QubitPool duplicate free");
				throw runtime_error("QubitPool duplicate free");
			}
			else
			{
				allocated_iter->second--;
				break;
			}
		}
	}
	if (allocated_iter != allocated_qubit.end())
	{
		if (allocated_iter->second == 0)
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
				allocated_qubit.erase(allocated_iter);
				delete _Qubit;
			}
		}
	}
	else
	{
		QCERR("QubitPool duplicate free");
		throw runtime_error("QubitPool duplicate free");
	}
}

size_t OriginQubitPoolv2::getPhysicalQubitAddr(Qubit* q)
{
	if (nullptr == q)
	{
		QCERR("qubit is nullptr");
		throw invalid_argument("qubit is nullptr");
	}
	for (auto iter = allocated_qubit.begin(); iter != allocated_qubit.end(); ++iter)
	{
		if (iter->first == q)
		{
			return q->getPhysicalQubitPtr()->getQubitAddr();
		}
	}
	QCERR("qubit argument error");
	throw invalid_argument("qubit argument error");
}

size_t OriginQubitPoolv2::getVirtualQubitAddress(Qubit* vq) const
{
	if (nullptr == vq)
	{
		QCERR("qubit is nullptr");
		throw invalid_argument("qubit is nullptr");
	}

	for (size_t i = 0; i < vecQubit.size(); ++i)
	{
		auto physAddr = vq->getPhysicalQubitPtr();
		if (vecQubit[i] == physAddr)
		{
			return i;
		}
	}
	QCERR("qubit argument error");
	throw invalid_argument("qubit argument error");
}

size_t OriginQubitPoolv2::get_allocate_qubits(std::vector<Qubit*>& qubit_vect) const
{
	for (auto iter : allocated_qubit)
	{
		qubit_vect.push_back(iter.first);
	}
	return allocated_qubit.size();
}