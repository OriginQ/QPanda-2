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

#ifndef _QUANTUM_GATE_H
#define _QUANTUM_GATE_H
#include <iostream>
#include <stdio.h>
#include <vector>
#include <complex>
#include <map>
#include "QuantumGateParameter.h"
#include "QError.h"

typedef std::vector<QGateParam> vQParam;


/*****************************************************************************************************************
QuantumGates:quantum gate
*****************************************************************************************************************/
class QuantumGates
{

public:
    QuantumGates();
    virtual ~QuantumGates() = 0;

    /*************************************************************************************************************
    Name:        getQState
    Description: get quantum state
    Argin:       pQuantumProParam      quantum program param.
    Argout:      sState                state string
    return:      quantum error
    *************************************************************************************************************/
    //virtual bool getQState(string & sState, QuantumGateParam *pQuantumProParam) = 0;

    //virtual QError Hadamard(size_t qn, double error_rate) = 0;
    virtual QError Hadamard(size_t qn, bool isConjugate, 
                        double error_rate) = 0;
    virtual QError Hadamard(size_t qn, Qnum& vControlBit, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError X(size_t qn, bool isConjugate, 
                        double error_rate) = 0;
    virtual QError X(size_t qn, Qnum& vControlBit, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError Y(size_t qn, bool isConjugate, 
                        double error_rate) = 0;
    virtual QError Y(size_t qn, Qnum& vControlBit, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError Z(size_t qn, bool isConjugate, 
                        double error_rate) = 0;
    virtual QError Z(size_t qn, Qnum& vControlBit, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError T(size_t qn, bool isConjugate, 
                        double error_rate) = 0;
    virtual QError T(size_t qn, Qnum& vControlBit, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError S(size_t qn, bool isConjugate, 
                        double error_rate) = 0;
    virtual QError S(size_t qn, Qnum& vControlBit, 
                        bool isConjugate, double error_rate) = 0;

    virtual QError RX_GATE(size_t qn, double theta,
                        bool isConjugate, double error_rate) = 0;
    virtual QError RX_GATE(size_t qn, double theta,
                        Qnum& vControlBit,
                        bool isConjugate,
                        double error_rate) = 0;
    virtual QError RY_GATE(size_t qn, double theta, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError RY_GATE(size_t qn, double theta,
                        Qnum& vControlBit,
                        bool isConjugate,
                        double error_rate) = 0;
    virtual QError RZ_GATE(size_t qn, double theta,
                        bool isConjugate, double error_rate) = 0;
    virtual QError RZ_GATE(size_t qn, double theta,
                        Qnum& vControlBit,
                        bool isConjugate,
                        double error_rate) = 0;
    virtual QError CNOT(size_t qn_0, size_t qn_1,
                        bool isConjugate, double error_rate) = 0;
    virtual QError CNOT(size_t qn_0, size_t qn_1,
                        Qnum& vControlBit,
                        bool isConjugate,
                        double error_rate) = 0;
    virtual QError CR(size_t qn_0, size_t qn_1, double theta, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError CR(size_t qn_0, size_t qn_1,
                        Qnum& vControlBit,
                        double theta,
                        bool isConjugate,
                        double error_rate) = 0;

    virtual QError CZ(size_t qn_0, size_t qn_1, 
                        bool isConjugate, double error_rate) = 0;
    virtual QError CZ(size_t qn_0, size_t qn_1,
                        Qnum& vControlBit,
                        bool isConjugate,
                        double error_rate) = 0;

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
                        double theta,
                        bool isConjugate,
                        double error_rate) = 0;
    virtual QError iSWAP(size_t qn_0, size_t qn_1,
                        Qnum& vControlBit,
                        double theta,
                        bool isConjugate,
                        double error_rate) = 0;

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
                        bool isConjugate,
                        double error_rate) = 0;
    virtual QError iSWAP(size_t qn_0, size_t qn_1,
                        Qnum& vControlBit,
                        bool isConjugate,
                        double error_rate) = 0;

    virtual QError SqiSWAP(size_t qn_0, size_t qn_1,
                        bool isConjugate,
                        double error_rate) = 0;
    virtual QError SqiSWAP(size_t qn_0, size_t qn_1,
                        Qnum& vControlBit,
                        bool isConjugate,
                        double error_rate) = 0;

    virtual QError Reset(size_t qn) = 0;
    virtual bool qubitMeasure(size_t qn) = 0;
    virtual QError pMeasure(Qnum& qnum, std::vector<std::pair<size_t, double>> &mResult, 
                        int select_max=-1) = 0;

    virtual QError pMeasure(Qnum& qnum, std::vector<double> &mResult) = 0;

    virtual QError initState(QuantumGateParam *) = 0;

    virtual QError endGate(QuantumGateParam *pQuantumProParam, 
                        QuantumGates * pQGate) = 0;
    
    virtual QError unitarySingleQubitGate(size_t qn, QStat& matrix, 
                        bool isConjugate, 
                        double error_rate) = 0;

    virtual QError controlunitarySingleQubitGate(size_t qn, Qnum& qnum,
                        QStat& matrix, 
                        bool isConjugate,
                        double error_rate) = 0;
    
    virtual QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
                        QStat& matrix,
                        bool isConjugate,
                        double error_rate) = 0;

    virtual QError controlunitaryDoubleQubitGate(size_t qn_0,
                        size_t qn_1,
                        Qnum& qnum,
                        QStat& matrix,
                        bool isConjugate,
                        double error_rate) = 0;

protected:
    //string sCalculationUnitType;
    /*************************************************************************************************************
    Name:        randGenerator
    Description: 16807 random number generator
    Argin:       None
    Argout:      None
    return:      random number in the region of [0,1]
    *************************************************************************************************************/
    double randGenerator();
};

#endif
