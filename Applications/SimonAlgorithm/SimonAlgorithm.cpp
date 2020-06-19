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
#include <bitset>
#include <vector>

using namespace std;
using namespace QPanda;

QCircuit controlfunc(vector<Qubit*> qVec, size_t index, int value)
{
    auto length = qVec.size() / 2;
    auto  cfunc = CreateEmptyCircuit();
    vector<Qubit*> qvtemp;
    qvtemp.insert(qvtemp.begin(), qVec.begin(), qVec.begin() + length);
    if (index == 1)
    {
        cfunc << X(qVec[0]);
    }
    else if (index == 2)
    {
        cfunc << X(qVec[1]);
    }
    else if (index == 0)
    {
        cfunc << X(qVec[0]);
        cfunc << X(qVec[1]);
    }
    if (value == 1)
    {
        QGate temp = X(qVec[3]);
        temp.setControl(qvtemp);
        cfunc << temp;
    }
    else if (value == 2)
    {
        QGate temp1 = X(qVec[2]);
        temp1.setControl(qvtemp);
        cfunc << temp1;
    }
    else if (value == 3)
    {
        QGate temp2 = X(qVec[2]);
        temp2.setControl(qvtemp);
        cfunc << temp2;
        QGate temp3 = X(qVec[3]);
        temp3.setControl(qvtemp);
        cfunc << temp3;
    }
    if (index == 1)
    {
        cfunc << X(qVec[0]);
    }
    else if (index == 2)
    {
        cfunc << X(qVec[1]);
    }
    else if (index == 0)
    {
        cfunc << X(qVec[0]);
        cfunc << X(qVec[1]);
    }
    return cfunc;
}


//f(x),x is 2bits variable
QCircuit oraclefunc(vector<Qubit*> qVec, vector<int> funvalue)
{
    auto length = qVec.size() / 2;
    auto  func = CreateEmptyCircuit();
    for (auto i = 0; i < 4; i++)
    {
        func << controlfunc(qVec, i, funvalue[i]);
    }
    return func;
}

QProg Simon_QProg(vector<Qubit*> qVec, vector<ClassicalCondition> cVec, vector<int> funvalue)
{
    size_t length = cVec.size();
    auto simon_qprog = CreateEmptyQProg();
    for (auto i = 0; i < length; i++)
    {
        simon_qprog << H(qVec[i]);
    }
    simon_qprog << oraclefunc(qVec,funvalue);
    for (auto i = 0; i < length; i++)
    {
        simon_qprog << H(qVec[i]);
    }
    for (auto i = 0; i < length; i++)
    {
        simon_qprog << Measure(qVec[i],cVec[i]);
    }
    return simon_qprog;
}

int main()
{
    cout << "4-qubit Simon Algorithm\n" << endl;
    cout << "f(x)=f(y)\t x+y=s" << endl;
    cout << "input f(x),f(x):[0,3]" << endl;
    vector<int> funcvalue(4, 0);
    cout << "input f(0):" << endl;
    cin >> funcvalue[0];
    cout << "input f(1):" << endl;
    cin >> funcvalue[1];
    cout << "input f(2):" << endl;
    cin >> funcvalue[2];
    cout << "input f(3):" << endl;
    cin >> funcvalue[3];
    cout << "f(0)=" << funcvalue[0] << endl;
    cout << "f(1)=" << funcvalue[1] << endl;
    cout << "f(2)=" << funcvalue[2] << endl;
    cout << "f(3)=" << funcvalue[3] << endl;
    cout << " Programming the circuit..." << endl;
    init(QMachineType::CPU);
    int qubit_num = 4;
    int cbit_nun = 2;
    vector<Qubit*> qVec = qAllocMany(4);
    vector<ClassicalCondition> cVec = cAllocMany(2);
    QProg  simonAlgorithm = Simon_QProg(qVec, cVec, funcvalue);
    vector<int> result(20);

    for (auto i = 0; i < 20; i++)
    {
        directlyRun(simonAlgorithm);
        result[i] = cVec[0].get_val() * 2 + cVec[1].get_val();
    }
    if (find(result.begin(), result.end(), 3) != result.end())
    {
        if (find(result.begin(), result.end(), 2) != result.end())
        {
            cout << "s=00" << endl;
        }
        else
        {
            cout << "s=11" << endl;
        }

    }
    else if (find(result.begin(), result.end(), 2) != result.end())
    {
        cout << "s=01" << endl;
    }
    else if (find(result.begin(), result.end(), 1) != result.end())
    {
        cout << "s=10" << endl;
    }
    finalize();
}