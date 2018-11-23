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

#ifndef CBIT_FACTORY_H
#define CBIT_FACTORY_H

#include "QuantumMachineInterface.h"
#include "QPanda/QPandaException.h"
#include <functional>
#include <stack>
using namespace std;

#define REGISTER_CBIT_NAME_(classname) \
CBit* classname##_Constructor(string name)\
{\
    return new classname(name);\
}\
static CBitFactoryHelper _CBit_Factory_Helper_##classname(\
    #classname, \
	classname##_Constructor\
)

/* CBit Factory */
class CBitFactory
{
	CBitFactory();
public:
	static CBitFactory & GetFactoryInstance();
	typedef function<CBit*(string)> name_constructor_t;
	typedef map<string, name_constructor_t> name_constructor_stack_t;
	name_constructor_stack_t _CBit_Constructor;
	void registerclass_name_(string&, name_constructor_t constructor);
	CBit* CreateCBitFromName(string);
};

class CBitFactoryHelper
{
	typedef CBitFactory::name_constructor_t
		name_constructor_t;
public:
	CBitFactoryHelper(string, name_constructor_t);
};

#endif