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

#ifndef QRESULT_FACTORY_H
#define QRESULT_FACTORY_H

#include "QuantumMachineInterface.h"
#include "QPanda/QPandaException.h"
#include <functional>
#include <stack>
using namespace std;


#define REGISTER_QRES_NAME(classname) \
QResult* classname##_Constructor()\
{\
    return new classname();\
}\
static QResultFactoryHelper _QRes_Factory_Helper_##classname(\
    #classname,\
	classname##_Constructor\
)

/* QResult Factory */
class QResultFactory
{
	QResultFactory();
public:
	typedef function<QResult*()> constructor_t;
	typedef map<string, constructor_t> constructor_Map_t;
	constructor_Map_t _QResult_Constructor;
	QResult* GetEmptyQResult();
	void registerclass(string &, constructor_t);
	static QResultFactory& GetFactoryInstance();
};

class QResultFactoryHelper
{
	typedef QResultFactory::constructor_t
		constructor_t;
public:
	QResultFactoryHelper(string, constructor_t _Constructor);
};


#endif