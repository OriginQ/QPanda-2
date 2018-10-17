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

#ifndef CEXPR_FACTORY_H
#define CEXPR_FACTORY_H

#include "ClassicalConditionInterface.h"
#include "QPanda/QPandaException.h"
#include <functional>
#include <stack>
using namespace std;

#define REGISTER_CEXPR(classname)\
CExpr* classname##_CBit_Constructor(CBit* cbit)\
{\
	return new classname(cbit);\
}\
CExpr* classname##_Operator_Constructor(\
	CExpr* leftexpr,\
	CExpr* rightexpr,\
	int op\
)\
{\
	return new classname(leftexpr, rightexpr, op);\
}\
static CExprFactoryHelper _CBit_Constructor_Helper_##classname( \
    #classname,\
    classname##_CBit_Constructor\
);\
static CExprFactoryHelper \
_Operator_Constructor_Helper_##classname\
(\
#classname,\
classname##_Operator_Constructor)

/* 11. CExpr Factory */
class CExprFactory
{
	CExprFactory();
public:

	static CExprFactory & GetFactoryInstance();

	typedef function<CExpr*(CBit*)> cbit_constructor_t;
	typedef map<string, cbit_constructor_t> cbit_constructor_map_t;
	cbit_constructor_map_t _CExpr_CBit_Constructor;
	CExpr* GetCExprByCBit(CBit*);
	void registerclass_CBit_(string &, cbit_constructor_t);

	typedef function<CExpr*(CExpr*, CExpr*, int)> operator_constructor_t;
	typedef map<string, operator_constructor_t> operator_constructor_map_t;
	operator_constructor_map_t _CExpr_Operator_Constructor;
	CExpr* GetCExprByOperation(CExpr*, CExpr*, int);
	void registerclass_operator_(string &, operator_constructor_t);
};


class CExprFactoryHelper
{
	typedef CExprFactory::cbit_constructor_t
		cbit_constructor_t;
	typedef CExprFactory::operator_constructor_t
		operator_constructor_t;
public:
	CExprFactoryHelper(string, cbit_constructor_t _Constructor);
	CExprFactoryHelper(string, operator_constructor_t _Constructor);
};

#endif