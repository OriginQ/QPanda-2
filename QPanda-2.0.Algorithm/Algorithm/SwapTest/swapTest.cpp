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

#include "swapTest.h"
#include<bitset>

//swap test,phi and psi are 1-qubit state
//phi contains theta1,alpha1,theta2,alpha2
//two ststes are :
//cos(theta1/2)|0>+exp(i*alpha1)sin(theta1/2)|1>
//cos(theta2/2)|0>+exp(i*alpha2)sin(theta2/2)|1>
QProg swaptest_QProg(vector<Qubit*> qVec, vector<CBit*> cVec, vector<double>& phi)
{

    QProg  swaptest_qprog = CreateEmptyQProg();
    swaptest_qprog << H(qVec[0]);
    //initial state
    swaptest_qprog << RY(qVec[1], phi[0])<<RZ(qVec[1], phi[1]);
    swaptest_qprog << RY(qVec[2], phi[2]) << RZ(qVec[2], phi[3]);
    //control swap
    QCircuit controlswap = CreateEmptyCircuit();
    controlswap << CNOT(qVec[1], qVec[2])<< CNOT(qVec[2], qVec[1])<<CNOT(qVec[1], qVec[2]);
    vector<Qubit*> qvtemp;
    qvtemp.push_back(qVec[0]);
    controlswap.setControl(qvtemp);
    swaptest_qprog << controlswap;
    swaptest_qprog <<H(qVec[0])<< Measure(qVec[0], cVec[0]);
    return swaptest_qprog;
}

void swaptest()
{
    cout << "Swap Test Algorithm\n" << endl;
    cout << "Initialize phi" << endl;
    double theta1;
    double alpha1;
    double theta2;
    double alpha2;
    vector<double> phi;
    cout << "input theta1:" << endl;
    cin >> theta1;
    cout << "input alpha1:" << endl;
    cin >> alpha1;
    cout << "input theta2:" << endl;
    cin >> theta2;
    cout << "input alpha2:" << endl;
    cin >> alpha2;
    cout << "phi=" << cos(theta1 / 2) << "*|0>+" << exp(1i*alpha1)*sin(theta1 / 2) << "|1>" << endl;
    cout << "psi=" << cos(theta2 / 2) << "*|0>+" << exp(1i*alpha2)*sin(theta2 / 2) << "|1>" << endl;
    phi.push_back(theta1);
    phi.push_back(alpha1);
    phi.push_back(theta2);
    phi.push_back(alpha2);

    cout<<" Programming the circuit..." << endl;
    init();
    vector<Qubit*> qVec;
    vector<CBit*> cVec;
    for (auto i = 0; i < 3 ; i++)
    {
        qVec.push_back(qAlloc());
    }
    cVec.push_back(cAlloc());

    double prob;
    size_t times=0;
    for (auto i = 0; i < 1000; i++)
    {
        if (swaptest1(qVec, cVec, phi))
        {
            times++;
        }
    }
    prob = times*0.001;
    cout << "|<phi|psi>|^2=" << 1 - 2 * prob << endl;
    return;
}
bool swaptest1(vector<Qubit*> qVec, vector<CBit*> cVec, vector<double>& phi)
{
    init();
    auto bvAlgorithm = swaptest_QProg(qVec, cVec, phi);
    append(bvAlgorithm);
    run();
    return getCBitValue(cVec[0]);
}
