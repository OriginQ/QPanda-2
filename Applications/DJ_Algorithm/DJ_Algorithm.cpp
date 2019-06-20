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

using QGEN = function<QCircuit(Qubit*, Qubit*)>;

QGEN The_Two_Qubit_Oracle(vector<bool> oracle_function) {
    return [oracle_function](Qubit* qubit1, Qubit* qubit2) {
        QCircuit prog;
        if (oracle_function[0] == false &&
            oracle_function[1] == true)
        {
            // f(x) = x;
            prog << CNOT(qubit1, qubit2);
        }
        else if (oracle_function[0] == true &&
            oracle_function[1] == false)
        {
            // f(x) = x + 1;
            prog << X(qubit2)
                << CNOT(qubit1, qubit2)
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


QProg Two_Qubit_DJ_With_Oracle(Qubit* qubit1, Qubit* qubit2, ClassicalCondition & cbit, QCircuit(*oracle)(Qubit* qubit1, Qubit* qubit2)) {

    auto prog = CreateEmptyQProg();
    //Firstly, create a circuit container

    prog << H(qubit1) << H(qubit2);
    // Perform Hadamard gate on all qubits

    prog << oracle(qubit1, qubit2);

    // Finally, Hadamard the first qubit and measure it
    prog << H(qubit1) << Measure(qubit1, cbit);
    return prog;
}

QProg  Two_Qubit_DJ_Algorithm_Circuit(
    Qubit * qubit1,
    Qubit * qubit2,
    ClassicalCondition & cbit,
    QGEN oracle)
{
    auto prog = CreateEmptyQProg();
    //Firstly, create a circuit container

    prog << H(qubit1) << H(qubit2);
    // Perform Hadamard gate on all qubits
    prog << oracle(qubit1, qubit2);

    // Finally, Hadamard the first qubit and measure it
    prog << H(qubit1) << Measure(qubit1, cbit);
    return prog;
}


QProg DJ_Algorithm(QVec & qvec, ClassicalCondition & c)
{
    if (qvec.size() != 2)
    {
        QCERR("qvec size error£¬the size of qvec must be 2");
        throw invalid_argument("qvec size error£¬the size of qvec must be 2");
    }

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

    auto temp = Reset_Qubit(qvec[0], false);
    temp << Reset_Qubit(qvec[1], true);
    auto oracle = The_Two_Qubit_Oracle(oracle_function);
    temp << Two_Qubit_DJ_Algorithm_Circuit(qvec[0], qvec[1], c, oracle);
    return temp;
}

int main()
{
    init(QMachineType::CPU);
    auto qvec = qAllocMany(2);
    auto c = cAlloc();
    auto prog = DJ_Algorithm(qvec, c);
    directlyRun(prog);
    if (c.eval() == false)
    {
        cout << "Constant function!";
    }
    else if (c.eval() == true)
    {
        cout << "Balanced function!";
    }
    finalize();
}
