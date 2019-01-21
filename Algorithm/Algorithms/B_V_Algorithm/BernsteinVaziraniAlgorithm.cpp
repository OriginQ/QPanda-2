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

#include "BernsteinVaziraniAlgorithm.h"
#include <bitset>

USING_QPANDA
QProg BV_QProg(vector<Qubit*> qVec, vector<CBit*> cVec, vector<bool>& a, bool b)
{
    if (qVec.size() != (a.size()+1))
    {
        throw exception();
    }
    size_t length = qVec.size();
    QProg  bv_qprog = CreateEmptyQProg();
    bv_qprog << X(qVec[length - 1]);
    for (auto iter = qVec.begin(); iter != qVec.end(); iter++)
    {
        bv_qprog << H(*iter);
    }
    for (auto i=0;i<length-1;i++)
    {
        if (a[i])
        {
            bv_qprog << CNOT(qVec[i], qVec[length - 1]);
        }
    }
    for (auto i = 0; i < length - 1; i++)
    {
        bv_qprog << H(qVec[i]);
    }
    for (auto i = 0; i < length - 1; i++)
    {
        bv_qprog << Measure(qVec[i], cVec[i]);
    }  
    return bv_qprog;
}

void BernsteinVaziraniAlgorithm()
{
    cout << "Bernstein Vazirani Algorithm\n" << endl;
    cout << "f(x)=a*x+b\n" << endl;
    cout << "input a" << endl;
    //size_t b;
    string stra;
    cin >> stra;
    //size_t a;
   // vector<bool> a(b);
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
    cout<<" Programming the circuit..." << endl;
    size_t qbitnum = a.size();
    init();
    vector<Qubit*> qVec;
    vector<CBit*> cVec;
    for (auto i = 0; i < qbitnum ; i++)
    {
        qVec.push_back(qAlloc());
        cVec.push_back(cAlloc());
    }
    qVec.push_back(qAlloc());
    auto bvAlgorithm = BV_QProg(qVec, cVec, a, b);
    append(bvAlgorithm);
    run();
    string measure;
    cout << "a=\t";
    for (auto iter = cVec.begin(); iter != cVec.end(); iter++)
    {
        cout << getCBitValue(*iter);
    }
    cout <<  "\n"<<"b=\t" << b << endl;
    return;
}
