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
#include "Core/Core.h"
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/Grover_Algorithm/Grover_Algorithm.h"
USING_QPANDA
using namespace std;

QCircuit diffusion_operator(vector<Qubit*> qvec) {
	vector<Qubit*> controller(qvec.begin(), qvec.end() - 1);
	QCircuit c;
	c << apply_QGate(qvec, H);
	c << apply_QGate(qvec, X);
	c << Z(qvec.back()).control(controller);
	c << apply_QGate(qvec, X);
	c << apply_QGate(qvec, H);
	
	return c;
}

QProg Grover_algorithm(vector<Qubit*> working_qubit,
	Qubit* ancilla,
	vector<ClassicalCondition> cvec,
	grover_oracle oracle,
	QuantumMachine *qvm,
	uint64_t repeat = 0) {
	QProg prog;
	prog << X(ancilla);
	prog << apply_QGate(working_qubit, H);
	prog << H(ancilla);

	// if repeat is not specified, choose a sufficient large repeat times.
	// repeat = (default) 100*sqrt(N)
	if (repeat == 0) {
		uint64_t sqrtN = 1ull << (working_qubit.size() / 2);
		repeat = 100 * sqrtN;
	}
	
	for (auto i = 0ull; i < repeat; ++i) {
		prog << oracle(working_qubit, ancilla);
		prog << diffusion_operator(working_qubit);
	}

	prog << MeasureAll(working_qubit, cvec);
	return prog;
}

QProg QPanda::groverAlgorithm(size_t target,
	size_t search_range,
	QuantumMachine * qvm ,
	grover_oracle oracle)
{
	if (0 == search_range)
	{
		QCERR("search_range equal 0");
		throw invalid_argument("search_range equal 0");
	}
	size_t qubit_number = (size_t)(log(search_range)/log(2));
	if (target > target)
	{
		QCERR("target > search_range");
		throw invalid_argument("target > search_range");
	}

	if (qubit_number == 0)
		return QProg();
	vector<Qubit*> working_qubit = qvm->allocateQubits(qubit_number);
	Qubit* ancilla = qvm->allocateQubit();

	int cbitnum = qubit_number;
	vector<ClassicalCondition> cvec = qvm->allocateCBits(cbitnum);

	auto prog = Grover_algorithm(working_qubit, ancilla, cvec, oracle, qvm, 1);

	return prog;
	
}

