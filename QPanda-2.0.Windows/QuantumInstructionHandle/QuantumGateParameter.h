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
using std::string;
/*****************************************************************************************************************
QuantumGateParam:quantum gate param 
*****************************************************************************************************************/
class QuantumGateParam
{
    typedef std::vector <std::complex<double>> QStat;
 public:
    int                           mPMeasureSize;                        /* PMeasure bit size                    */
    int                           mQuantumBitNumber;                    /* quantum bit number                   */

    std::map<string , bool>              mReturnValue;               /* MonteCarlo result                    */
    std::vector<std::pair<size_t, double>>  mPMeasure;                  /* Pmeasure result                      */

    QuantumGateParam() : mPMeasureSize(0),mQuantumBitNumber(0){};
    ~QuantumGateParam()
    {
        mReturnValue.clear();
        mPMeasure.clear();
    };

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

#endif
