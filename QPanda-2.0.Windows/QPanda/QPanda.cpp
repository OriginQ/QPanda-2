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

#include "QPanda.h"
#include "../QuantumMachine/Factory.h"

QuantumMachine* qvm = Factory::QuantumMachineFactory
::GetFactoryInstance().CreateByName("OriginQVM");// global

void init()
{
	qvm->init();
}

void finalize()
{
	qvm->finalize();
}

Qubit * qAlloc()
{
	return qvm->Allocate_Qubit();
}

void qFree(Qubit * q)
{
	qvm->Free_Qubit(q);
}

CBit * cAlloc()
{
	return qvm->Allocate_CBit();
}

void cFree(CBit * c)
{
	qvm->Free_CBit(c);
}

void load(QuantumProgram & q)
{
	qvm->load(q);
}

void append(QuantumProgram & q)
{
	qvm->append(q);
}

QMachineStatus* getstat()
{
	return qvm->getStatus();
}

QResult* getResult()
{
	return qvm->getResult();
}

ClassicalCondition bind_a_cbit(CBit * c)
{
	return ClassicalCondition(c);
}

void run()
{
	qvm->run();
}

map<string, bool> getResultMap()
{
	return qvm->getResult()->getResultMap();
}

bool getCBitValue(CBit * cbit)
{
	auto resmap = getResultMap();
	return resmap[cbit->getName()];
}




