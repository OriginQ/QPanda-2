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
using namespace std;
using namespace QPanda;
QCircuit addOne(vector<Qubit*> qVec)
{
    auto qCircuit = CreateEmptyCircuit();
    vector<Qubit*> vControlQbit;
    vControlQbit.insert(vControlQbit.begin(), qVec.begin() + 1, qVec.end()-1);
    for (auto iter = qVec.begin(); iter != qVec.end() - 1; iter++)
    {
        auto gat = X(*iter);
        gat.setControl(vControlQbit);
        qCircuit << gat;
        if (vControlQbit.size() >= 1)
        {
            vControlQbit.erase(vControlQbit.begin(), vControlQbit.begin()+1);
        }
        
    }
    return qCircuit;
}
QCircuit walkOneStep(vector<Qubit*> qVec)
{
    int iLength = (int)qVec.size();
    auto qCircuit = CreateEmptyCircuit();
    qCircuit << X(qVec[iLength - 1]);
    vector<Qubit*> vControlQbit;
    vControlQbit.insert(vControlQbit.begin(), qVec.begin() + 1, qVec.end());
    auto qCircuit1 = addOne(qVec);
    auto qCircuit2 = addOne(qVec);
    qCircuit2.setDagger(true);
    qCircuit << qCircuit1;
    qCircuit << X(qVec[iLength - 1]);
    qCircuit << qCircuit2;
    return qCircuit;
}

QProg quantumWalk(vector<Qubit*> qVec, vector<ClassicalCondition> cVec)
{
    
    size_t length = qVec.size();
    QProg  QuantumWalkProg = CreateEmptyQProg();
    QuantumWalkProg << X(qVec[length - 2]) << X(qVec[length - 2]);
    for (auto i = 0; i < ((1 << length)-1); i++)
    {
        QuantumWalkProg << H(qVec[length - 1]);
        QuantumWalkProg << walkOneStep(qVec);
    }

    return QuantumWalkProg;
}
int main()
{
    int qubitnum = 0;
    cout << "welcome to Quantum walk\n" << endl
        << "\n" << endl
        << "please input qubit num\n";
    cin >> qubitnum;

    if ((qubitnum < 0) || (qubitnum > 24))
    {
        QCERR("qubitnum need > 0 and <24");
        exit(1);
    }
    init(QMachineType::CPU);
    vector<Qubit*> qVec = qAllocMany(qubitnum);
    vector<ClassicalCondition> cVec = cAllocMany(qubitnum);
    qVec.push_back(qAlloc());
    auto qwAlgorithm = quantumWalk(qVec, cVec);
    auto reuslt = directlyRun(qwAlgorithm);

    for(auto var : reuslt)
    {
        cout << var.first << " : " << var.second << endl;
    }
    finalize();
}

