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

#include "Utilities/Utilities.h"
#include "Utils/Utilities.h"
#include "QPandaNamespace.h"
#include "Core/QPanda.h"
#include <vector>

using namespace std;
USING_QPANDA

using DJ_Oracle = Oracle<QVec, Qubit*>;

DJ_Oracle generate_two_qubit_oracle(vector<bool> oracle_function);

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

QProg Deutsch_Jozsa_algorithm(vector<Qubit*> qubit1, Qubit* qubit2, vector<ClassicalCondition> cbit, DJ_Oracle oracle) {

    auto prog = CreateEmptyQProg();
    //Firstly, create a circuit container

	prog << X(qubit2);
    prog << apply_QGate(qubit1, H) << H(qubit2);
    // Perform Hadamard gate on all qubits

    prog << oracle(qubit1, qubit2);

    // Finally, Hadamard the first qubit and measure it
    prog << apply_QGate(qubit1, H) << MeasureAll(qubit1, cbit);
    return prog;
}

void two_qubit_deutsch_jozsa_algorithm(vector<bool> boolean_function)
{
	init(QMachineType::CPU);
	auto qvec = qAllocMany(2);
	auto c = cAlloc();
    if (qvec.size() != 2)
    {
        QCERR("qvec size error£¬the size of qvec must be 2");
        throw invalid_argument("qvec size error£¬the size of qvec must be 2");
    }

    auto oracle = generate_two_qubit_oracle(boolean_function);
	QProg prog;
	prog << Deutsch_Jozsa_algorithm({ qvec[0] }, qvec[1], { c }, oracle);

	/* To Print The Circuit */
	/*
	extern QuantumMachine* global_quantum_machine;
	cout << transformQProgToQRunes(prog, global_quantum_machine) << endl;
	*/

	directlyRun(prog);
	if (c.eval() == false)
	{
		cout << "Constant function!" << endl;
	}
	else if (c.eval() == true)
	{
		cout << "Balanced function!" << endl;
	}
	finalize();
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
