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
#include "Utilities/Utilities.h"
#include "Utils/Utilities.h"
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


QProg BV_QProg(QVec qVec, vector<ClassicalCondition> cVec, vector<bool>& a,
    BV_Oracle & oracle)
{

    if (qVec.size() != (a.size()+1))
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
    size_t length = qVec.size();
    QProg  bv_qprog = CreateEmptyQProg();
    bv_qprog << X(qVec[length - 1])
             << apply_QGate(qVec, H)
             << oracle(qVec, qVec[length - 1]);
             
    qVec.pop_back();

    bv_qprog << apply_QGate(qVec, H) << MeasureAll(qVec, cVec);
    return bv_qprog;
}

int main()
{
    while (1)
    {
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
        init();
        vector<Qubit*> qVec = qAllocMany(qubitnum + 1);
        auto cVec = cAllocMany(qubitnum);
        auto oracle = generate_bv_oracle(a);
        auto bvAlgorithm = BV_QProg(qVec, cVec, a, oracle);
        directlyRun(bvAlgorithm);
        string measure;
        cout << "a=\t";
        for (auto iter = cVec.begin(); iter != cVec.end(); iter++)
        {
            cout << (*iter).eval();
        }
        cout <<endl;
        finalize();
    }
    
}
