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
#include <bitset>
#include <vector>

using namespace std;
using namespace QPanda;



int main()
{
	string filename = "testfile.txt";
	std::ofstream os(filename);
	os << R"(QINIT 4
		CREG 2
		RY q[3], (1.570796)
		DAGGER
		H q[1]
		H q[2]
		RZ q[2], (2.356194)
		CU q[2], q[3], (3.141593, 4.712389, 1.570796, -1.570796)
		RZ q[1], (4.712389)
		CU q[1], q[3], (3.141593, 4.712389, 1.570796, -3.141593)
		CNOT q[1], q[2]
		CNOT q[2], q[1]
		CNOT q[1], q[2]
		H q[2]
		CU q[2], q[1], (-0.785398, -1.570796, 0.000000, 0.000000)
		H q[1]
		SWAP q[1], q[2]
		ENDDAGGER
		DAGGER
		X q[1]
		CONTROL q[1], q[2]
		RY q[0], (3.141593)
		ENDCONTROL
		X q[1]
		X q[2]
		CONTROL q[1], q[2]
		RY q[0], (1.047198)
		ENDCONTROL
		X q[2]
		CONTROL q[1], q[2]
		RY q[0], (0.679674)
		ENDCONTROL
		ENDDAGGER
		MEASURE q[0], c[0]
		QIF c[0]
		DAGGER
		H q[1]
		H q[2]
		RZ q[2], (2.356194)
		CU q[2], q[3], (3.141593, 4.712389, 1.570796, -1.570796)
		RZ q[1], (4.712389)
		CU q[1], q[3], (3.141593, 4.712389, 1.570796, -3.141593)
		CNOT q[1], q[2]
		CNOT q[2], q[1]
		CNOT q[1], q[2]
		H q[2]
		CU q[2], q[1], (-0.785398, -1.570796, 0.000000, 0.000000)
		H q[1]
		ENDDAGGER 
        ENDQIF

		)"
		;

	os.close();

	init();
	extern QuantumMachine* global_quantum_machine;
	QProg prog = QPanda::transformOriginIRToQProg(filename, global_quantum_machine);

	cout << transformQProgToOriginIR(prog, global_quantum_machine);

	getchar();

	return 0;
}
