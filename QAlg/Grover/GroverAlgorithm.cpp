/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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
#include "QAlg/Grover/GroverAlgorithm.h"

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

	std::cout << "before oracle" << endl;
	std::cout << prog << endl;
	auto mat = getCircuitMatrix(prog);
	std::cout << mat << endl;

	// if repeat is not specified, choose a sufficient large repeat times.
	// repeat = (default) 100*sqrt(N)
	if (repeat == 0) {
		uint64_t sqrtN = 1ull << (working_qubit.size() / 2);
		repeat = 100 * sqrtN;
	}
	
	for (auto i = 0ull; i < repeat; ++i) {
		prog << oracle(working_qubit, ancilla);

		std::cout << "oracle" << endl;
		std::cout << prog << endl;
		auto mat2 = getCircuitMatrix(prog);
		std::cout << mat2 << endl;

		prog << diffusion_operator(working_qubit);
	}

	std::cout << "last prog" << endl;
	std::cout << prog << endl;
	auto mat3 = getCircuitMatrix(prog);
	std::cout << mat3 << endl;

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

std::vector<size_t> QPanda::search_target_from_measure_result(const prob_dict& measure_result)
{
	std::vector<size_t> target_index;
	double total_val = 0.0;
	size_t data_cnt = 0;
	for (const auto& var : measure_result) {
		total_val += var.second;
		++data_cnt;
	}

	double average_probability = total_val / data_cnt;
	size_t possible_solutions_cnt = 0;
	double possible_solutions_sum = 0.0;
	//printf("measure result:\n");
	for (const auto aiter : measure_result)
	{
		//printf("%s:%5f\n", aiter.first.c_str(), aiter.second);
		if (aiter.second > average_probability)
		{
			++possible_solutions_cnt;
			possible_solutions_sum += aiter.second;
		}
	}

	//printf("first average_probability: %f\n", average_probability);
	average_probability = ((0.15 * possible_solutions_sum) / possible_solutions_cnt) + (0.85 * average_probability); /**< Weighted Sum */
	//printf("second average_probability: %f\n", average_probability);
	size_t search_result_index = 0;
	for (const auto aiter : measure_result)
	{
		if (aiter.second > average_probability) {
			target_index.push_back(search_result_index);
		}

		++search_result_index;
	}

	return target_index;
}

