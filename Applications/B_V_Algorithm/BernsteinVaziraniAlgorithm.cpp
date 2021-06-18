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
#include "QPanda.h"
#include "Utilities/Tools/Utils.h"
#include "QAlg/B_V_Algorithm/BernsteinVaziraniAlgorithm.h"
#include <bitset>

using namespace std;
using namespace QPanda;

using BV_Oracle = Oracle<QVec, Qubit *>;

BV_Oracle generate_bv_oracle(vector<bool> oracle_function) {
    return [oracle_function](QVec qVec, Qubit* qubit) {
        QCircuit bv_qprog;
        auto length = oracle_function.size();
        for (auto i = 0; i < length; i++)
        {
            if (oracle_function[i])
            {
                bv_qprog << CNOT(qVec[i], qubit);
            }
        }
        return bv_qprog;
    };
}

int main()
{
    while (1)
    {
        auto qvm = initQuantumMachine();
        cout << "Bernstein Vazirani Algorithm\n" << endl;
        cout << "f(x)=a*x+b\n" << endl;
        cout << "input a" << endl;
        string stra;
        cin >> stra;
        vector<bool> a;
        for (auto iter = stra.begin(); iter != stra.end(); iter++)
        {
            if (*iter == '0')
            {
                a.push_back(0);
            }
            else
            {
                a.push_back(1);
            }
        }

        cout << "input b" << endl;
        bool b;
        cin >> b;
        cout << "a=\t" << stra << endl;
        cout << "b=\t" << b << endl;
        cout << " Programming the circuit..." << endl;
        size_t qubitnum = a.size();

        auto oracle = generate_bv_oracle(a);
        auto bvAlgorithm = bernsteinVaziraniAlgorithm(stra, b, qvm, oracle);
        auto result = qvm->directlyRun(bvAlgorithm);

        for (auto aiter : result)
        {
            std::cout << aiter.first << " : " << aiter.second << std::endl;
        }
        destroyQuantumMachine(qvm);
    }

    return 0;
}
