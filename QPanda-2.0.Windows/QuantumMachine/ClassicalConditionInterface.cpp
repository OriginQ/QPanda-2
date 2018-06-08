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

#include "ClassicalConditionInterface.h"
#include "Factory.h"

bool ClassicalCondition::eval(map<string, bool>
_Value_Map)
{
    if(nullptr == expr)
    {
        throw exception();
    }
	return expr->eval(_Value_Map);
}

bool ClassicalCondition::checkValidity() const
{
	return expr->checkValidity();
}

ClassicalCondition::ClassicalCondition(CBit *cbit)
{
	auto &fac = Factory::CExprFactory::GetFactoryInstance();
	expr=fac.GetCExprByCBit(cbit);
	if (expr == nullptr)
	{
		throw factory_get_instance_fail(
			"CExpr"
		);
	}
}

ClassicalCondition::ClassicalCondition(CExpr *_Expr)
{
	expr = _Expr;
}

ClassicalCondition& 
ClassicalCondition::operator=(ClassicalCondition newcond)
{
	delete expr;
	expr = newcond.expr->deepcopy();
	return *this;
}

ClassicalCondition::ClassicalCondition
(const ClassicalCondition &cc)
{
	expr = cc.expr->deepcopy();
}

ClassicalCondition::~ClassicalCondition()
{
    delete expr;
}

ClassicalCondition operator+(
	ClassicalCondition leftcc,
	ClassicalCondition rightcc)
{
	return
		Factory::CExprFactory::
		GetFactoryInstance().GetCExprByOperation
		(
			leftcc.expr->deepcopy(),
			rightcc.expr->deepcopy(),
			PLUS
		);
}

ClassicalCondition operator-(
	ClassicalCondition leftcc,
	ClassicalCondition rightcc)
{
	return
		Factory::CExprFactory::
		GetFactoryInstance().GetCExprByOperation
		(
			leftcc.expr->deepcopy(),
			rightcc.expr->deepcopy(),
			MINUS
		);
}


ClassicalCondition operator&&(
	ClassicalCondition leftcc,
	ClassicalCondition rightcc)
{
	return
		Factory::CExprFactory::
		GetFactoryInstance().GetCExprByOperation
		(
			leftcc.expr->deepcopy(),
			rightcc.expr->deepcopy(),
			AND
		);
}

ClassicalCondition operator||(
	ClassicalCondition leftcc,
	ClassicalCondition rightcc)
{
	return
		Factory::CExprFactory::
		GetFactoryInstance().GetCExprByOperation
		(
			leftcc.expr->deepcopy(),
			rightcc.expr->deepcopy(),
			OR
		);
}

ClassicalCondition operator!(
	ClassicalCondition leftcc)
{
	return
		Factory::CExprFactory::
		GetFactoryInstance().GetCExprByOperation
		(
			leftcc.expr->deepcopy(),
			nullptr,
			NOT
		);
}
