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

#ifndef PHYSICAL_QUBIT_FACTORY_H
#define PHYSICAL_QUBIT_FACTORY_H

#include "QuantumMachineInterface.h"
#include "QPanda/QPandaException.h"
#include <functional>
#include <stack>
using namespace std;

#define REGISTER_PHYSICAL_QUBIT(classname)  \
PhysicalQubit * classname##_Constructor()\
{\
    return new classname();\
}\
 PhysicalQubitFactoryHelper  _Physical_Qubit_Factory_Helper_##classname(\
    #classname, \
    classname##_Constructor \
) 

/*  Physical Qubit Factory*/
class PhysicalQubitFactory
{
	// Factory for class PhysicalQubit
	PhysicalQubitFactory();
public:

	static PhysicalQubitFactory & GetFactoryInstance();

	PhysicalQubit* GetInstance();
	typedef function<PhysicalQubit*()> constructor_t;
	typedef map<string, constructor_t> constructor_Map_t;
	void registerclass(string &, constructor_t constructor);
	// the constructor stack
	constructor_Map_t _Physical_Qubit_Constructor;
};

class PhysicalQubitFactoryHelper
{
	typedef PhysicalQubitFactory::constructor_t constructor_t;
public:
	PhysicalQubitFactoryHelper(string, constructor_t);
};

#endif