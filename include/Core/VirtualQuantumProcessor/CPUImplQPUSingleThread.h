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

#ifndef CPU_QUANTUM_GATE_SINGLE_THREAD_H
#define CPU_QUANTUM_GATE_SINGLE_THREAD_H

#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#ifndef SQ2
#define SQ2 (1 / 1.4142135623731)
#endif

#ifndef PI
#define PI 3.14159265358979
#endif

class CPUImplQPUSingleThread : public QPUImpl
{
public:
    vQParam qubit2stat;
    QGateParam &findgroup(size_t qn);
    CPUImplQPUSingleThread();
    CPUImplQPUSingleThread(size_t);
    ~CPUImplQPUSingleThread();

    bool TensorProduct(QGateParam& qgroup0, QGateParam& qgroup1);

    QError unitarySingleQubitGate(size_t qn,
        QStat& matrix,
        bool isConjugate,
        double error_rate,
        GateType);

    QError controlunitarySingleQubitGate(size_t qn,
        Qnum& vControlBit,
        QStat& matrix, 
        bool isConjugate,
        double error_rate,
        GateType);
    
    QError unitaryDoubleQubitGate(size_t qn_0,
        size_t qn_1, 
        QStat& matrix, 
        bool isConjugate, 
        double error_rate,
        GateType);
   
    QError controlunitaryDoubleQubitGate(size_t qn_0,
        size_t qn_1,
        Qnum& vControlBit,
        QStat& matrix,
        bool isConjugate,
        double error_rate,
        GateType);

    virtual QError P0(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError P0(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    virtual QError P1(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError P1(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) ;


    virtual QError Hadamard(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError Hadamard(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    virtual QError X(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError X(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    virtual QError Y(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError Y(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    virtual QError Z(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError Z(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    virtual QError T(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError T(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    virtual QError S(size_t qn, bool isConjugate,
        double error_rate);

    virtual QError S(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    virtual QError U1_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);

    virtual QError RX_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);

    virtual QError RX_GATE(size_t qn, double theta,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate);

    virtual QError RY_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);

    virtual QError RY_GATE(size_t qn, double theta,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate);

    virtual QError RZ_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);

    virtual QError RZ_GATE(size_t qn, double theta,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate);

    virtual QError CNOT(size_t qn_0, size_t qn_1,
        bool isConjugate, double error_rate);

    virtual QError CNOT(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate);

    virtual QError CR(size_t qn_0, size_t qn_1, double theta,
        bool isConjugate, double error_rate);

    virtual QError CR(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        double theta,
        bool isConjugate,
        double error_rate);

    virtual QError CZ(size_t qn_0, size_t qn_1,
        bool isConjugate, double error_rate);

    virtual QError CZ(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate);

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        double theta,
        bool isConjugate,
        double error_rate);

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        double theta,
        bool isConjugate,
        double error_rate);

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        bool isConjugate,
        double error_rate);

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate);

    virtual QError SqiSWAP(size_t qn_0, size_t qn_1,
        bool isConjugate,
        double error_rate);

    virtual QError SqiSWAP(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate);

/*
add multi-qubit gate which has diagonal matrix
2019/03/15 
*/

    virtual QError DiagonalGate(Qnum& vQubit, QStat & matrix,
        bool isConjugate, double error_rate);
    virtual QError controlDiagonalGate(Qnum& vQubit, QStat & matrix, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    QStat getQState();
    QError Reset(size_t qn);
    bool qubitMeasure(size_t qn);
    QError pMeasure(Qnum& qnum, std::vector<std::pair<size_t, double>> &mResult, int select_max=-1);
    QError pMeasure(Qnum& qnum, std::vector<double> &mResult);
    QError initState(QuantumGateParam *);
    QError endGate(QuantumGateParam *pQuantumProParam, QPUImpl * pQGate);
};

#endif
