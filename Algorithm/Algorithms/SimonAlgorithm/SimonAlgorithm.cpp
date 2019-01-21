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

#include "SimonAlgorithm.h"
#include <bitset>
#include <vector>














//QProg Simon_QProg(vector<Qubit*> qVec, vector<CBit*> cVec, vvb& func)
//{
//    
//    size_t length = cVec.size();
//    QProg  simon_qprog = CreateEmptyQProg();
//    vector<Qubit*> qvtemp;
//    qvtemp.insert(qvtemp.begin(), qVec.begin(), qVec.begin()+length);
//    vector<Qubit*> qvtemp1;
//    qvtemp1.insert(qvtemp1.begin(), qVec.begin() + length, qVec.end());
//    for (auto i = 0; i < length; i++)
//    {
//        simon_qprog << H(qVec[i]);
//    }
//
//    for (auto iter = func.begin(); iter != func.end(); iter++)
//    {
//        auto orac = SimonOracle(qvtemp1, *iter);
//        orac.setControl(qvtemp);
//        simon_qprog << orac;
//    }
//    for (auto i = 0; i < length; i++)
//    {
//        simon_qprog << H(qVec[i]);
//    }
//    return simon_qprog;
//}
//
//
//void SimonAlgorithm()
//{
//    cout << "Simon Algorithm\n" << endl;
//    cout << "f(x)=f(y)\t x+y=s" << endl;
//    cout << "input dimension of f(x)" << endl;
//    //size_t b;
//    size_t dimension;
//    cin >> dimension;
//    vector<string> vstr(dimension);
//    //vvfunc:definition of f(x)
//    vvb vvfunc(dimension);
//
//    for (auto i = 0; i < (1 << dimension); i++)
//    {
//        cout << "f(" << dec << i << ")=" << endl;
//        cin >> vstr[i];
//        for (auto iter = vstr[i].begin(); iter != vstr[i].end(); iter++)
//        {
//            if (*iter == '0')
//            {
//                vvfunc[i].push_back(0);
//            }
//            else
//            {
//                vvfunc[i].push_back(1);
//            }
//        }
//    }
//
//    cout << "definition of f(x)" << endl;
//    for (auto i = 0; i < (1<<dimension); i++)
//    {
//        cout << "f(" << i << ")=" << vstr[i] << endl;
//    }
//    string stemp = vstr[0];
//    vector<bool>  vbtemp(dimension,0);
//    for (auto i = 1; i < (1<<dimension); i++)
//    {
//        if (stemp == vstr[i])
//        {
//            for (auto j = 0; j < dimension; j++)
//            {
//                vbtemp[j] = vvfunc[0][j] | vvfunc[i][j];
//            }
//        }
//        
//    }
//    cout << "s=";
//    for (auto iter = vbtemp.begin(); iter != vbtemp.end(); iter++)
//    {
//        cout << *iter;
//    }
//    cout << endl;
//
//    cout<<" Programming the circuit..." << endl;
//    
//
//    vector<Qubit*> qVec;
//    vector<CBit*> cVec;
//    for (auto i = 0; i <2*dimension ; i++)
//    {
//        qVec.push_back(qAlloc());
//        if (i % 2 == 0)
//        {
//            cVec.push_back(cAlloc());
//        }
//    }
//    vector<bool> result(dimension);     //save value of s
//    bool findresult = 0;
//    auto simonAlgorithm = Simon_QProg(qVec, cVec, vvfunc);
//    append(simonAlgorithm);
//    
//    vvb vvby(dimension);
//    for (auto i = 0; i < dimension; i++)
//    {
//        init();
//        run();
//        for (auto j = 0; j < dimension; j++)
//        {
//            vvby[i].push_back(getCBitValue(cVec[j]));
//        }
//        
//    }
//    bool btemp;
//    for (auto i = 1; i < (1 << dimension); i++)
//    {
//        btemp = 1;
//        for (auto iter = vvby.begin(); iter != vvby.end(); iter++)
//        {
//            btemp = btemp & isSatisfied(*iter, i);
//        }
//         
//    }
//    
//    
//
//
//
//    run();
//    
//    for (auto iter = cVec.begin(); iter != cVec.end(); iter++)
//    {
//        cout << getCBitValue(*iter);
//    }
//    cout <<  "\n"<<"b=\t" << b << endl;
//    return;
//}
//
//
//QCircuit SimonOracle(vector<Qubit*> qVec, vector<bool> oracle)
//{
//    auto oraclecircuit = CreateEmptyCircuit();
//    for (auto i=0;i<qVec.size();i++)
//    {
//        if (oracle[i])
//        {
//            oraclecircuit << X(qVec[i]);
//        }
//    }
//    return QCircuit();
//}
//
////y*s=0
//bool isSatisfied(vector<bool>& vb, size_t num)
//{
//    size_t stemp = 0;
//    for (auto i=0;i<vb.size();i++)
//    {
//        stemp += ((num << i) | vb[i]);
//    }
//    return stemp % 2;
//}
void simonTest()
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
    init();
    vector<Qubit*> qVec;
    vector<CBit*> cVec;
    for (auto i = 0; i < 4 ; i++)
    {
        qVec.push_back(qAlloc());
        if (i % 2 == 0)
        {
            cVec.push_back(cAlloc());
        }
    }
    //auto controlcircuit = oraclefunc(qVec, funcvalue);
    //auto simonAlgorithm = CreateEmptyQProg();
    QProg  simonAlgorithm = Simon_QProg(qVec, cVec, funcvalue);
    //auto simonAlgorithm = CreateEmptyQProg();
    //simonAlgorithm << controlcircuit;
    append(simonAlgorithm);
    vector<int> result(20);
    for (auto i = 0; i < 20; i++)
    {
        //init();
        run();
        result[i] = getCBitValue(cVec[0])*2 + getCBitValue(cVec[1]);
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
}
//int simonExecution(vector<Qubit*> qVec, vector<CBit*> cVec,QProg pro)
//{
//    init();
//    append(pro);
//    run();
//    return 1<<getCBitValue(cVec[0]) + getCBitValue(cVec[1]);
//}
QProg Simon_QProg(vector<Qubit*> qVec, vector<CBit*> cVec, vector<int> funvalue)
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

//f(x),x is 2bits variable
QCircuit oraclefunc(vector<Qubit*> qVec, vector<int> funvalue)
{
    auto length = qVec.size() / 2;
    auto  func = CreateEmptyCircuit();
    for (auto i = 0; i < 4; i++)
    {
        func << controlfunc(qVec,i, funvalue[i]);
    }
    return func;
}
QCircuit controlfunc(vector<Qubit*> qVec,size_t index, int value)
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
