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

#ifndef CLASSICAL_CONDITION_INTERFACE_H
#define CLASSICAL_CONDITION_INTERFACE_H

#include <string>
#include "QuantumMachineInterface.h"

using namespace std;

enum ContentSpecifier
{
	CBIT,
	OPERATOR
};

enum OperatorSpecifier
{
	PLUS,
	MINUS,
	AND,
	OR,
	NOT,
};

class CExpr
{
	// classical expression interface	
public:
    
	virtual CExpr* getLeftExpr() const = 0;
	virtual CExpr* getRightExpr() const = 0;
	virtual void setLeftExpr(CExpr*) = 0;
	virtual void setRightExpr(CExpr*) = 0;
	virtual string getName() const = 0;
	virtual CBit* getCBit() const = 0;
    virtual bool checkValidity() const = 0;
	virtual ~CExpr() {}
    virtual bool eval(map<string , bool>) const = 0;
	virtual CExpr* deepcopy() const = 0;
};

class ClassicalCondition
{
	CExpr* expr;
public:
	inline CExpr * getExprPtr() const { return expr; }
	bool eval(map<string, bool>);
    bool checkValidity() const;
	ClassicalCondition(CBit*);
	ClassicalCondition(CExpr*);
	ClassicalCondition(const ClassicalCondition&);
	friend ClassicalCondition operator+(ClassicalCondition, ClassicalCondition);
	friend ClassicalCondition operator-(ClassicalCondition, ClassicalCondition);
	friend ClassicalCondition operator&&(ClassicalCondition, ClassicalCondition);
	friend ClassicalCondition operator||(ClassicalCondition, ClassicalCondition);
	friend ClassicalCondition operator!(ClassicalCondition);
    ~ClassicalCondition();
};

#endif