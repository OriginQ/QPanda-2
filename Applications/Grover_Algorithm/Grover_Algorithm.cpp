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
#include "Core/QPanda.h"
#include "Utils/Utilities.h"

USING_QPANDA
using namespace std;

using grover_oracle = Oracle<QVec, Qubit*>;

grover_oracle generate_3_qubit_oracle(int target) {
	return [target](QVec qvec, Qubit* qubit) {
		QCircuit oracle;
		switch (target)
		{
		case 0:
			oracle << X(qvec[0]) << X(qvec[1]) << Toffoli(qvec[0], qvec[1], qubit) << X(qvec[0]) << X(qvec[1]);
			break;
		case 1:
			oracle << X(qvec[0]) << Toffoli(qvec[0], qvec[1], qubit) << X(qvec[0]);
			break;
		case 2:
			oracle << X(qvec[1]) << Toffoli(qvec[0], qvec[1], qubit) << X(qvec[1]);
			break;
		case 3:
			oracle << Toffoli(qvec[0], qvec[1], qubit);
			break;
		}
		return oracle;
	};
}

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

QProg Grover_algorithm(vector<Qubit*> working_qubit, Qubit* ancilla, vector<ClassicalCondition> cvec, grover_oracle oracle, uint64_t repeat = 0) {
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

int main()
{
	while (1) {
		int target;
		cout << "input the input function" << endl
			<< "The function has a boolean input" << endl
			<< "and has a boolean output" << endl
			<< "target=(0/1/2/3)?";
		cin >> target;
		cout << "Programming the oracle..." << endl;
		grover_oracle oracle = generate_3_qubit_oracle(target);

		init(QMachineType::CPU_SINGLE_THREAD);

		int qubit_number = 3;
		vector<Qubit*> working_qubit = qAllocMany(qubit_number - 1);
		Qubit* ancilla = qAlloc();

		int cbitnum = 2;
		vector<ClassicalCondition> cvec = cAllocMany(cbitnum);

		auto prog = Grover_algorithm(working_qubit, ancilla, cvec, oracle, 1);

		/* To Print The Circuit */

		extern QuantumMachine* global_quantum_machine;
		cout << transformQProgToQRunes(prog, global_quantum_machine) << endl;


		auto resultMap = directlyRun(prog);
		if (resultMap["c0"])
		{
			if (resultMap["c1"])
			{
				cout << "target number is 3 !";
			}
			else
			{
				cout << "target number is 2 !";
			}
		}
		else
		{
			if (resultMap["c1"])
			{
				cout << "target number is 1 !";
			}
			else
			{
				cout << "target number is 0 !";
			}
		}
		finalize();
	}
	return 0;
}

