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

#ifndef _QUANTUM_GATE_PARAM_H
#define _QUANTUM_GATE_PARAM_H
#include <mutex>
#include <iostream>
#include <map>
#include <complex>
#include <algorithm>
#include <vector>
using std::map;
using std::vector;
using std::complex;
using std::string;
using std::pair;
using std::size_t;
typedef complex <double> COMPLEX;
typedef vector<size_t>  Qnum;
typedef vector <complex<double>> QStat;
#define iunit COMPLEX(0,1)
/*****************************************************************************************************************
QuantumGateParam:quantum gate param 
*****************************************************************************************************************/
class QuantumGateParam
{
    typedef std::vector <std::complex<double>> QStat;
 public:
    int                           mQuantumBitNumber;                    /* quantum bit number                   */

    std::map<string , bool>              mReturnValue;               /* MonteCarlo result                    */


    /*************************************************************************************************************
    Name:        setQBitNum
    Description: set quantum number
    Argin:       iQuantumBitNum      quantum bit number
    Argout:      None
    return:      true or false
    *************************************************************************************************************/
    inline bool setQBitNum(int iQuantumBitNum)
    {
        mQuantumBitNumber = iQuantumBitNum;
        return true;
    }
};
class QGateParam
{

public:
    QGateParam() {};
    QGateParam(int qn) :qVec(qn, 0), qstate(1 << qn, 0), qubitnumber(qn)
    {
        for (auto i = 0; i < qubitnumber; i++)
        {
            qVec[i] = i;
        }
        qstate[0] = 1;
    }
    Qnum qVec;
    QStat qstate;
    int qubitnumber;
    bool enable = true;

};
#endif
