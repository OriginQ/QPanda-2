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
using namespace std;
using namespace QPanda;

//operation sequence P->Q->P,P is qVec[1],Q is qVec[0]
QProg CoinFlip_Algorithm(vector<Qubit*> qVec, vector<ClassicalCondition> cVec,bool fx)
{
    auto coinflip = CreateEmptyQProg();
    //initial state: |10>-|01>
    coinflip << X(qVec[0]) <<H(qVec[0])<<X(qVec[1])<< CNOT(qVec[0], qVec[1]);
    //P's first operation
    coinflip << H(qVec[1]);
    //Q's operation
    if (fx )
    {
        coinflip << X(qVec[0]);
    }

    //P's second operation
    coinflip << H(qVec[0]);
    //Joint measurement
    coinflip << CNOT(qVec[0], qVec[1]) << H(qVec[0]);
    coinflip << Measure(qVec[0], cVec[0]) << Measure(qVec[1], cVec[1]);
    return QProg();
}

int CoinFlip_Prog(QProg & prog, vector<Qubit*> qVec, vector<ClassicalCondition> cVec, bool fx)
{
    auto temp = CoinFlip_Algorithm(qVec, cVec, fx);
    prog << temp;
    directlyRun(prog);
    return ((1 << cVec[1].get_val()) + (int)cVec[0].get_val());
}

int main()
{
    bool fx = 0;
    cout << "Entanglement Flip Game\n" << endl
        << "\n" << endl
        << "Input choice of Q:(0/1)\n";
    cin >> fx;
    cout << "Programming the circuit..." << endl;
    int outcome=0;
    init(QMachineType::CPU);
    vector<Qubit*> qVec = qAllocMany(2);
    vector<ClassicalCondition> cVec = cAllocMany(2);
    QProg prog;
    auto temp = CoinFlip_Prog(prog,qVec, cVec, fx);
    for (auto i = 0; i < 10; i++)
    {
        outcome = CoinFlip_Prog(prog,qVec,cVec,fx);
        if (temp != outcome)
        {
            cout << "Q wins!\n" << endl;
            return 0;
        }
    }
    cout << "max entanglement!" << endl;
    cout << "P wins!\n" << endl;
}
