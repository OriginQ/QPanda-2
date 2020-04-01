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

#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/DJ_Algorithm/DJ_Algorithm.h"
#include "QPandaNamespace.h"
#include "Core/Core.h"
#include <vector>

using namespace std;
USING_QPANDA

DJ_Oracle generate_two_qubit_oracle(vector<bool> oracle_function) {
    return [oracle_function](QVec qubit1, Qubit* qubit2) {
        QCircuit prog;
        if (oracle_function[0] == false &&
            oracle_function[1] == true)
        {
            // f(x) = x;
            prog << CNOT(qubit1[0], qubit2);
        }
        else if (oracle_function[0] == true &&
            oracle_function[1] == false)
        {
            // f(x) = x + 1;
            prog << CNOT(qubit1[0], qubit2)
                << X(qubit2);
        }
        else if (oracle_function[0] == true &&
            oracle_function[1] == true)
        {
            // f(x) = 1
            prog << X(qubit2);
        }
        else
        {
            // f(x) = 0, do nothing  
        }
        return prog;
    };
}

void two_qubit_deutsch_jozsa_algorithm(vector<bool> boolean_function)
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
    auto oracle = generate_two_qubit_oracle(boolean_function);
	
	auto prog = deutschJozsaAlgorithm(boolean_function, qvm, oracle);
	auto result = qvm ->directlyRun(prog);
	if (result["c0"] == false)
	{
		cout << "Constant function!" << endl;
	}
	else if (result["c0"] == true)
	{
		cout << "Balanced function!" << endl;
	}
	destroyQuantumMachine(qvm);
}

int main()
{
	while (1) {
		bool fx0 = 0, fx1 = 0;
		cout << "input the input function" << endl
			<< "The function has a boolean input" << endl
			<< "and has a boolean output" << endl
			<< "f(0)= (0/1)?";
		cin >> fx0;
		cout << "f(1)=(0/1)?";
		cin >> fx1;
		std::vector<bool> oracle_function({ fx0,fx1 });
		cout << "Programming the circuit..." << endl;
		two_qubit_deutsch_jozsa_algorithm(oracle_function);
	}
	return 0;
}
