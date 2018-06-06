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

#include "HHL_Algorithm.h"

/*****************************************************************************************************************
Name:        HHL_Algorithm
Description: HHL algorithm program test
Argin:       times       execution times of HHL algorithm
return:      none
*****************************************************************************************************************/
void HHL_Algorithm( size_t times)
{
    map<string, bool> temp;
    int x0 = 0;
    int x1 = 0;
    for (size_t i = 0; i < times;i++)
    {
        temp = hhlalgorithm();
        if (temp["c0"])
        {
            if (temp["c1"])
            {
                x1++;
            }
            else
            {
                x0++;
            }
        }
    }
    int sum = x0 + x1;
    cout << "prob0:" << x0*1.0/sum << endl;
    cout << "prob1:" << x1*1.0/sum << endl;
}

/*****************************************************************************************************************
Name:        hhlalgorithm
Description: execution of HHL algorithm
Argin:       none
return:      resultMap                 execution outcome of HHL algorithm
*****************************************************************************************************************/
map<string, bool> hhlalgorithm()
{
    init();
    int qubitnum = 4;
    vector<Qubit*> qv;
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
    auto &hhlprog =CreateEmptyQProg();
    hhlprog << RY(qv[3], PI / 2);       //  change vecotr b in equation Ax=b
    hhlprog << hhl(qv, cv);
    load(hhlprog);
    run();
    auto resultMap = getResultMap();
    finalize();
    return resultMap;
}

/*****************************************************************************************************************
Name:        hhl
Description: program of HHL Algorithm, the linear equations are Ax=b, 
             A=[1.5 0.5;0.5 1.5],b=[1/sqrt(2);1/sqrt(2)] after encoding.
Argin:       qVec          pointer vector of the qubit
             cVec           pointer vector of the cbit
return:      hhlProg                 program of HHL Algorithm
*****************************************************************************************************************/
QProg& hhl(vector<Qubit*> qVec, vector<CBit*> cVec)
{
    ClassicalCondition *cc0=bind_a_cbit(cVec[0]);

    OriginQCircuit & ifcircuit = CreateEmptyCircuit();
    OriginQCircuit & PSEcircuit = hhlPse(qVec);                           //PSE circuit of HHL algorithm
    OriginQCircuit & CRot = CRotate(qVec);                                //controled rotation circuit of HHL algorithm
    OriginQCircuit & PSEcircuitdag = hhlPse(qVec).dagger();               //PSE dagger circuit
    QProg & PSEdagger = CreateEmptyQProg();
    PSEdagger << PSEcircuitdag << Measure(qVec[3], cVec[1]);
    QIfNode & ifnode = CreateIfProg(cc0, &PSEdagger);
    QProg & hhlProg = CreateEmptyQProg();
    hhlProg << PSEcircuit << CRot << Measure(qVec[0], cVec[0]) << ifnode; //whole HHL program
    return hhlProg;
}

/*****************************************************************************************************************
Name:        hhlPse
Description: Quantum Phase Estimation circuit of HHL algorithm
Argin:       qVec    pointer vector of the qubit
return:      PSEcircuit                 quantum phase estimation circuit
*****************************************************************************************************************/
OriginQCircuit& hhlPse(vector<Qubit*> qVec)
{
    OriginQCircuit & PSEcircuit = CreateEmptyCircuit();
    PSEcircuit << H(qVec[1]) << H(qVec[2]) << RZ(qVec[2], 0.75*PI);
    OriginQGateNode & gat1 = QDouble(PI, 1.5*PI, -0.5*PI, PI / 2, qVec[2], qVec[3]);
    OriginQGateNode & gat2 = QDouble(PI, 1.5*PI, -PI, PI / 2, qVec[1], qVec[3]);
    PSEcircuit << gat1 << RZ(qVec[1], 1.5*PI) << gat2;
    PSEcircuit << CNOT(qVec[1], qVec[2]) << CNOT(qVec[2], qVec[1]) << CNOT(qVec[1], qVec[2]);
    OriginQGateNode & gat3 = QDouble(-0.25*PI, -0.5*PI, 0, 0, qVec[2], qVec[1]);
    PSEcircuit << H(qVec[2]) << gat3 << H(qVec[1]);
    return PSEcircuit;
}

/*****************************************************************************************************************
Name:        CRotate
Description: controled rotation circuit of HHL algorithm
Argin:       qVec    pointer vector of the qubit
return:      CRot                 controled rotation circuit of HHL algorithm
*****************************************************************************************************************/
OriginQCircuit& CRotate(vector<Qubit*> qVec)
{
    OriginQCircuit & CRot = CreateEmptyCircuit();
    vector<Qubit *> controlVector;
    controlVector.push_back(qVec[1]);
    controlVector.push_back(qVec[2]);
    OriginQGateNode & gat4 = RY(qVec[0], PI);
    gat4.setControl(controlVector);
    OriginQGateNode & gat5 = RY(qVec[0], PI / 3);
    gat5.setControl(controlVector);
    OriginQGateNode & gat6 = RY(qVec[0], 0.679673818908);                 //arcsin(1/3)
    gat6.setControl(controlVector);
    CRot << RX(qVec[1]) << gat4 << RX(qVec[1]) << RX(qVec[2]) << gat5 << RX(qVec[2]) << gat6;
    return CRot;
}