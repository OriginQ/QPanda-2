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

#include "DJ_Algorithm.h"
#include "Utilities/Utilities.h"

namespace QPanda
{

    void DJ_Algorithm()
    {
        bool fx0 = 0, fx1 = 0;
        cout << "input the input function" << endl
            << "The function has a boolean input" << endl
            << "and has a boolean output" << endl
            << "f(0)= (0/1)?";
        cin >> fx0;
        cout << "f(1)=(0/1)?";
        cin >> fx1;
        vector<bool> oracle_function({ fx0,fx1 });
        cout << "Programming the circuit..." << endl;
        init();
        auto q1 = qAlloc();
        auto q2 = qAlloc();
        auto c1 = cAlloc();

        Reset_Qubit(q1, false);
        Reset_Qubit(q2, true);

        auto temp = Two_Qubit_DJ_Algorithm_Circuit(q1, q2, c1, oracle_function);
        append(temp);

        run();

        //auto resultMap = getResultMap();
        if (getCBitValue(c1) == false)
        {
            cout << "Constant function!";
        }
        else if (getCBitValue(c1) == true)
        {
            cout << "Balanced function!";
        }
    }

    QProg  Two_Qubit_DJ_Algorithm_Circuit(
        Qubit * qubit1,
        Qubit * qubit2,
        CBit * cbit,
        vector<bool> oracle_function)

    {
        auto prog = CreateEmptyQProg();
        //Firstly, create a circuit container

        prog << H(qubit1) << H(qubit2);
        // Perform Hadamard gate on all qubits

        if (oracle_function[0] == false
            &&
            oracle_function[1] == false)
            // different oracle leads to different circuit
            // f(x) = oracle_function[x]
        {
            // f(x) = 0, do nothing
        }
        else if (oracle_function[0] == false
            &&
            oracle_function[1] == true
            )
        {
            // f(x) = x;
            prog << CNOT(qubit1, qubit2);
        }
        else if (oracle_function[0] == true
            &&
            oracle_function[1] == false
            )
        {
            // f(x) = x + 1;
            prog << X(qubit2)
                << CNOT(qubit1, qubit2)
                << X(qubit2);
        }
        else if (oracle_function[0] == true
            &&
            oracle_function[1] == true
            )
        {
            // f(x) = 1
            prog << X(qubit2);
        }

        // Finally, Hadamard the first qubit and measure it
        prog << H(qubit1) << Measure(qubit1, cbit);
        return prog;
    }

}