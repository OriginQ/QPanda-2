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

#include "Utilities.h"

void Reset_Qubit(Qubit* q, bool setVal)
{
	auto cbit = cAlloc();
    auto aTmep = Reset_Qubit_Circuit(q, cbit, setVal);
	append(aTmep);
	cFree(cbit);
}

QuantumProgram  Reset_Qubit_Circuit(Qubit *q, CBit* cbit, bool setVal)
{
	//auto cbit = cAlloc();
	auto prog = CreateEmptyQProg();
	prog << Measure(q, cbit);
	auto cond = bind_a_cbit(cbit);
	auto resetcircuit = CreateEmptyCircuit();
	resetcircuit << RX(q);
	auto no_reset = CreateEmptyCircuit();
	if (setVal==false)
		prog << CreateIfProg(cond, &resetcircuit, &no_reset);
	else
		prog << CreateIfProg(cond, &no_reset, &resetcircuit);
	return prog;
}

void Reset_All(vector<Qubit*> qubit_vector, bool setVal)
{
	for_each(qubit_vector.begin(),
		qubit_vector.end(),
		[setVal](Qubit* q) {Reset_Qubit(q, setVal); });
}
