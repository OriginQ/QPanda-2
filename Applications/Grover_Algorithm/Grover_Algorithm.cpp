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
#include "Core/Utilities/Utilities.h"
#include "QAlg/Grover_Algorithm/Grover_Algorithm.h"

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

int main()
{

	while (1) {
		auto qvm = initQuantumMachine(QMachineType::CPU_SINGLE_THREAD);
		int target;
		cout << "input the input function" << endl
			<< "The function has a boolean input" << endl
			<< "and has a boolean output" << endl
			<< "target=(0/1/2/3)?";
		cin >> target;
		cout << "Programming the oracle..." << endl;
		grover_oracle oracle = generate_3_qubit_oracle(target);

		auto prog = groverAlgorithm(target,4, qvm, oracle);
		cout << transformQProgToOriginIR(prog, qvm) << endl;

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
		destroyQuantumMachine(qvm);
	}

	return 0;
}

