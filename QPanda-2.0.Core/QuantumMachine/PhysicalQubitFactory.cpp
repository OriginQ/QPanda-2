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
#include "PhysicalQubitFactory.h"
#include "QPanda/ConfigMap.h"
#include "OriginQuantumMachine.h"
/* Physical Qubit Factory*/
PhysicalQubitFactory::PhysicalQubitFactory()
{

}

PhysicalQubitFactory&
PhysicalQubitFactory::GetFactoryInstance()
{
	static PhysicalQubitFactory fac;
	return fac;
}

/* Factory Get Instance */
PhysicalQubit * PhysicalQubitFactory::GetInstance()
{
	// I think the last Constructor that user push
	// to the top of the stack is the user defined
	// class. So only Get the top is ok

	auto sClassName = ConfigMap::getInstance()["PhysicalQubit"];
	if (sClassName.size() <= 0)
	{
		return nullptr;
	}
	auto aiter = _Physical_Qubit_Constructor.find(sClassName);

	if (aiter == _Physical_Qubit_Constructor.end())
	{
		return nullptr;
	}

	return aiter->second();
}

void PhysicalQubitFactory::registerclass(string & sClassName, constructor_t constructor)
{
	// Only push the Constructor to the top
	_Physical_Qubit_Constructor.insert(make_pair(sClassName, constructor));
}

/* Factory Helper Class */
PhysicalQubitFactoryHelper::PhysicalQubitFactoryHelper(string  sClassName, constructor_t _Constructor)
{
	auto &fac = PhysicalQubitFactory::GetFactoryInstance();
	fac.registerclass(sClassName, _Constructor);
}

REGISTER_PHYSICAL_QUBIT(OriginPhysicalQubit);






