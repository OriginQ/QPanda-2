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

#ifndef QUBIT_POOL_FACTORY_H
#define QUBIT_POOL_FACTORY_H

#include "QuantumMachineInterface.h"
#include "QPanda/QPandaException.h"
#include <functional>
#include <stack>
using namespace std;

#define REGISTER_QUBIT_POOL_SIZE_(classname) \
QubitPool* classname##_Constructor(size_t size)\
{\
    return new classname(size);\
}\
static QubitPoolFactoryHelper _Qubit_Pool_Factory_Helper_##classname(\
    #classname,\
	classname##_Constructor\
)

/* Qubit Pool Factory */

class QubitPoolFactory
{
	QubitPoolFactory();
public:
	static QubitPoolFactory& GetFactoryInstance();
	typedef function<QubitPool*(size_t)> size_constructor_t;
	typedef map<string, size_constructor_t> size_constructor_stack_t;
	size_constructor_stack_t _Qubit_Pool_Constructor;
	QubitPool* GetPoolWithoutTopology(size_t);
	void registerclass_size_(string &, size_constructor_t constructor);
};

class QubitPoolFactoryHelper
{
	typedef QubitPoolFactory::
		size_constructor_t size_constructor_t;
public:
	QubitPoolFactoryHelper(string, size_constructor_t);
};

#endif