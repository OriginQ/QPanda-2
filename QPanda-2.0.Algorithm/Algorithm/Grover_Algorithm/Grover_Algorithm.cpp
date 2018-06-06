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

#include "Grover_Algorithm.h"


/*****************************************************************************************************************
Name:        Grover
Description: program of 3-qubit Grover Search Algorithm
Argin:       qVec          pointer vector of the qubits
             cVec          pointer vector of the cbits
return:      grover        program of Grover Search Algorithm
*****************************************************************************************************************/
QProg& Grover(vector<Qubit*> qVec, vector<CBit*> cVec, int target)
{
    QProg & grover = CreateEmptyQProg();
    OriginQCircuit & init = CreateEmptyCircuit();
    OriginQCircuit & oracle = CreateEmptyCircuit();
    OriginQCircuit & reverse = CreateEmptyCircuit();
    init << H(qVec[0]) << H(qVec[1]) << RX(qVec[2]) << H(qVec[2]);
    vector<Qubit *> controlVector;
    controlVector.push_back(qVec[0]);
    controlVector.push_back(qVec[1]);
    OriginQGateNode  &toff = RX(qVec[2]);
    toff.setControl(controlVector);
    switch (target)
    {
    case 0:
        oracle << RX(qVec[0]) << RX(qVec[1]) << toff << RX(qVec[0]) << RX(qVec[1]);
        break;
    case 1:
        oracle << RX(qVec[0]) << toff << RX(qVec[0]);
        break;
    case 2:
        oracle << RX(qVec[1]) << toff << RX(qVec[1]);
        break;
    case 3:
        oracle << toff;
        break;
    }
    reverse << H(qVec[0]) << H(qVec[1]) << RX(qVec[0]) << RX(qVec[1])
        << H(qVec[1]) << CNOT(qVec[0], qVec[1]);
    reverse << H(qVec[1]) << RX(qVec[0]) << RX(qVec[1]) << H(qVec[0]) << H(qVec[1]) << RX(qVec[2]);
    grover << init << oracle << reverse << Measure(qVec[0], cVec[0]) << Measure(qVec[1], cVec[1]);
    return grover;
}


/*****************************************************************************************************************
Name:        Grover_Algorithm
Description: 3-qubit Grover Algorithm,2 working qubits and 1 ancilla qubit.
             find target number in the region of [0,3],user inputs the target
             number,different target number corresponds to differnent oracle 
             in Grover Algorithm.
*****************************************************************************************************************/
void Grover_Algorithm()
{
    int target;
    cout << "input the target number" << endl
        << "The region is [0,3]" << endl
        << "target= ";
    cin >> target;
    cout << "Programming the circuit..." << endl;
    init();
    vector<Qubit*> qv;
    int qubitnum = 3;
    for (size_t i = 0; i < qubitnum; i++)
    {
        qv.push_back(qAlloc());
    }
    vector<CBit*> cv;
    int cbitnum = 2;
    for (size_t i = 0; i < cbitnum; i++)
    {
        cv.push_back(cAlloc());
    }
    auto &groverprog = Grover(qv, cv, target);
    load(groverprog);
    run();
    auto resultMap = getResultMap();
    if (resultMap["c0"])
    {
        if (resultMap["c1"])
        {
            cout << "target number is 3 !";
        }
        else
        {
            cout << "target number is 2 !";
        }
    }
    if (!resultMap["c0"])
    {
        if (resultMap["c1"])
        {
            cout << "target number is 1 !";
        }
        else
        {
            cout << "target number is 0 !";
        }
    }
    finalize();
}
