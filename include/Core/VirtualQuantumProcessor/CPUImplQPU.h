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

#ifndef CPU_QUANTUM_GATE_H
#define CPU_QUANTUM_GATE_H

#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/Utilities/Utilities.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#ifndef SQ2
#define SQ2 (1 / 1.4142135623731)
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

#define DECL_GATE_MATRIX(NAME)\
extern const qcomplex_t NAME##00;\
extern const qcomplex_t NAME##01;\
extern const qcomplex_t NAME##10;\
extern const qcomplex_t NAME##11;
#define DECL_ANGLE_GATE_MATRIX(NAME)\
extern const double NAME##_Nx;\
extern const double NAME##_Ny;\
extern const double NAME##_Nz;\

#define REGISTER_GATE_MATRIX(NAME,U00,U01,U10,U11)\
extern const qcomplex_t NAME##00 = U00;\
extern const qcomplex_t NAME##01 = U01;\
extern const qcomplex_t NAME##10 = U10;\
extern const qcomplex_t NAME##11 = U11;

#define REGISTER_ANGLE_GATE_MATRIX(NAME,Nx,Ny,Nz)\
extern const double NAME##_Nx = Nx;\
extern const double NAME##_Ny = Ny;\
extern const double NAME##_Nz = Nz;\

#define CONST_GATE(NAME) \
QError                                          \
NAME(size_t qn, bool isConjugate, double error_rate)\
{                                                    \
    const_single_qubit_gate(NAME, qn,isConjugate,error_rate);\
    return  qErrorNone;                      \
}
#define CONTROL_CONST_GATE(NAME) \
QError                                          \
NAME(size_t qn, Qnum& vControlBit,bool isConjugate , double error_rate)\
{                                                    \
    control_const_single_qubit_gate(NAME, qn,vControlBit,isConjugate,error_rate);\
    return  qErrorNone;                      \
}

#define SINGLE_ANGLE_GATE(NAME) \
QError                                          \
NAME(size_t qn,double theta,bool isConjugate, double error_rate)\
{                                                    \
    single_qubit_angle_gate(NAME, qn,theta,isConjugate,error_rate);\
    return  qErrorNone;                      \
}

#define CONTROL_SINGLE_ANGLE_GATE(NAME)    \
QError                                          \
NAME(size_t qn, double theta,Qnum& vControlBit,bool isConjugate, double error_rate)\
{                                                    \
    control_single_qubit_angle_gate(NAME, qn, theta,vControlBit,isConjugate, error_rate); \
    return  qErrorNone;                      \
}

#define REGISTER_DOUBLE_GATE_MATRIX(NAME,matrix) \
extern const QStat NAME##_Matrix=matrix;

#define const_single_qubit_gate(GATE_NAME,qn,isConjugate,error_rate) \
single_gate<GATE_NAME##00,GATE_NAME##01,GATE_NAME##10,GATE_NAME##11>(qn,isConjugate,error_rate)

#define control_const_single_qubit_gate(GATE_NAME,qn,vControlBit,isConjugate,error_rate) \
control_single_gate<GATE_NAME##00,GATE_NAME##01,GATE_NAME##10,GATE_NAME##11>\
(qn,vControlBit,isConjugate,error_rate)

#define single_qubit_angle_gate(GATE_NAME,qn,theta,isConjugate,error_rate) \
single_angle_gate<GATE_NAME##_Nx,GATE_NAME##_Ny,GATE_NAME##_Nz>(qn,theta,isConjugate,error_rate)

#define control_single_qubit_angle_gate(GATE_NAME,qn,theta,vControlBit,isConjugate,error_rate) \
control_single_angle_gate<GATE_NAME##_Nx,GATE_NAME##_Ny,GATE_NAME##_Nz>     \
(qn,theta,vControlBit,isConjugate,error_rate)

DECL_GATE_MATRIX(Hadamard)
DECL_GATE_MATRIX(X)
DECL_GATE_MATRIX(Y)
DECL_GATE_MATRIX(Z)
DECL_GATE_MATRIX(T)
DECL_GATE_MATRIX(S)
DECL_GATE_MATRIX(P0)
DECL_GATE_MATRIX(P1)
DECL_ANGLE_GATE_MATRIX(RX_GATE)
DECL_ANGLE_GATE_MATRIX(RY_GATE)
DECL_ANGLE_GATE_MATRIX(RZ_GATE)

class CPUImplQPU : public QPUImpl
{
public:
    vQParam qubit2stat;
    QGateParam & findgroup(size_t qn);
    CPUImplQPU();
    CPUImplQPU(size_t);
    ~CPUImplQPU();

    inline bool TensorProduct(QGateParam& qgroup0, QGateParam& qgroup1)
    {
        if (qgroup0.qVec[0] == qgroup1.qVec[0])
        {
            return false;
        }
        size_t length = qgroup0.qstate.size();
        size_t slabel = qgroup0.qVec[0];
        for (auto iter0 = qgroup1.qstate.begin(); iter0 != qgroup1.qstate.end(); iter0++)
        {
            for (auto i = 0; i < length; i++)
            {
                //*iter1 *= *iter;
                qgroup0.qstate.push_back(qgroup0.qstate[i] * (*iter0));
            }
        }
        qgroup0.qstate.erase(qgroup0.qstate.begin(), qgroup0.qstate.begin() + length);
        qgroup0.qVec.insert(qgroup0.qVec.end(), qgroup1.qVec.begin(), qgroup1.qVec.end());
        qgroup1.enable = false;
        return true;
    }

    template<const qcomplex_t& U00, const qcomplex_t& U01, const qcomplex_t& U10, const qcomplex_t& U11>
    QError single_gate(size_t qn, bool isConjugate, double error_rate)
    {
        qcomplex_t alpha;
        qcomplex_t beta;
        QGateParam& qgroup = findgroup(qn);
        size_t j;
        size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
        qcomplex_t C00 = U00;
        qcomplex_t C01 = U01;
        qcomplex_t C10 = U10;
        qcomplex_t C11 = U11;
        if (isConjugate)
        {
            qcomplex_t temp;
            C00 = qcomplex_t(C00.real(), -C00.imag());
            C01 = qcomplex_t(C01.real(), -C01.imag());
            C10 = qcomplex_t(C10.real(), -C10.imag());
            C11 = qcomplex_t(C11.real(), -C11.imag());
            temp = C01;;
            C01 = U10;
            C10 = temp;
        }
        //#pragma omp parallel for private(j,alpha,beta)
        for (size_t i = 0; i < qgroup.qstate.size(); i += ststep * 2)
        {
            for (j = i; j<i + ststep; j++)
            {
                alpha = qgroup.qstate[j];
                beta = qgroup.qstate[j + ststep];
                qgroup.qstate[j] = C00 * alpha + C01 * beta;         /* in j,the goal qubit is in |0>        */
                qgroup.qstate[j + ststep] = C10 * alpha + C11 * beta;         /* in j+ststep,the goal qubit is in |1> */
            }
        }
        return qErrorNone;
    }


    QError U1_GATE(size_t qn, double theta,bool isConjugate,double error_rate)
    {
        QGateParam& qgroup = findgroup(qn);
        size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
        qcomplex_t C00 = (1,0);
        qcomplex_t C01 = (0,0);
        qcomplex_t C10 = (0,0);
        qcomplex_t C11 = isConjugate? qcomplex_t(cos(-theta), sin(-theta)) :qcomplex_t(cos(theta),sin(theta));
        for (size_t i = 0; i < qgroup.qstate.size(); i += ststep * 2)
        {
            for (size_t j = i; j < i + ststep; ++j)
            {
                qgroup.qstate[j + ststep] = C11 * qgroup.qstate[j + ststep];
            }
        }
        return qErrorNone;
    }


    template<const double& Nx, const double& Ny, const double& Nz>
    QError single_angle_gate(size_t qn, double theta, bool isConjugate, double error_rate)
    {
        qcomplex_t alpha;
        qcomplex_t beta;
        qcomplex_t U00(cos(theta / 2), -sin(theta / 2)*Nz);
        qcomplex_t U01(-sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U10(sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U11(cos(theta / 2), sin(theta / 2)*Nz);
        if (isConjugate)
        {
            qcomplex_t temp;
            U00 = qcomplex_t(U00.real(), -U00.imag());
            U01 = qcomplex_t(U01.real(), -U01.imag());
            U10 = qcomplex_t(U10.real(), -U10.imag());
            U11 = qcomplex_t(U11.real(), -U11.imag());
            temp = U01;
            U01 = U10;
            U10 = temp;
        }
        QGateParam& qgroup = findgroup(qn);
        size_t j;
        size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
        //#pragma omp parallel for private(j,alpha,beta)
        for (size_t i = 0; i < qgroup.qstate.size(); i += ststep * 2)
        {
            for (j = i; j<i + ststep; j++)
            {
                alpha = qgroup.qstate[j];
                beta = qgroup.qstate[j + ststep];
                qgroup.qstate[j] = U00 * alpha + U01 * beta;         /* in j,the goal qubit is in |0>        */
                qgroup.qstate[j + ststep] = U10 * alpha + U11 * beta;         /* in j+ststep,the goal qubit is in |1> */
            }
        }
        return qErrorNone;
    }

    template<const double& Nx, const double& Ny, const double& Nz>
    QError control_single_angle_gate(size_t qn,
        double theta,
        Qnum vControlBit,
        bool isConjugate,
        double error_rate)
    {
        if (QPanda::RandomNumberGenerator() > error_rate)
        {
            QGateParam& qgroup0 = findgroup(qn);
            for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
            {
                TensorProduct(qgroup0, findgroup(*iter));
            }
            size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());
            size_t x;

            size_t n = qgroup0.qVec.size();
            size_t ststep = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn)
                - qgroup0.qVec.begin());
            size_t index = 0;
            size_t block = 0;

            qcomplex_t alpha, beta;
            qcomplex_t U00(cos(theta / 2), -sin(theta / 2)*Nz);
            qcomplex_t U01(-sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
            qcomplex_t U10(sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
            qcomplex_t U11(cos(theta / 2), sin(theta / 2)*Nz);
            if (isConjugate)
            {
                qcomplex_t temp;
                U00 = qcomplex_t(U00.real(), -U00.imag());
                U01 = qcomplex_t(U01.real(), -U01.imag());
                U10 = qcomplex_t(U10.real(), -U10.imag());
                U11 = qcomplex_t(U11.real(), -U11.imag());
                temp = U01;
                U01 = U10;
                U10 = temp;
            }
            Qnum qvtemp;
            for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
            {
                size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
                    - qgroup0.qVec.begin());
                block += 1ull << stemp;
                qvtemp.push_back(stemp);
            }
            sort(qvtemp.begin(), qvtemp.end());
            Qnum::iterator qiter;
            size_t j;
            //#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
            for (size_t i = 0; i < M; i++)
            {
                index = 0;
                x = i;
                qiter = qvtemp.begin();

                for (j = 0; j < n; j++)
                {
                    while (qiter != qvtemp.end() && *qiter == j)
                    {
                        qiter++;
                        j++;
                    }
                    //index += ((x % 2)*(1ull << j));
                    index += ((x & 1) << j);
                    x >>= 1;
                }

                /*
                * control qubits are 1,target qubit is 0
                */
                index = index + block - ststep;
                alpha = qgroup0.qstate[index];
                beta = qgroup0.qstate[index + ststep];
                qgroup0.qstate[index] = alpha * U00 + beta * U01;
                qgroup0.qstate[index + ststep] = alpha * U10 + beta * U11;
            }
        }
        return qErrorNone;
    }

    template<const qcomplex_t& U00,
        const qcomplex_t& U01,
        const qcomplex_t& U10,
        const qcomplex_t& U11>
        QError control_single_gate(
            size_t qn,
            Qnum  vControlBit,
            bool isConjugate,
            double error_rate)
    {
        if (QPanda::RandomNumberGenerator() > error_rate)
        {
            QGateParam& qgroup0 = findgroup(qn);
            for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
            {
                TensorProduct(qgroup0, findgroup(*iter));
            }
            size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());
            size_t x;

            size_t n = qgroup0.qVec.size();
            size_t ststep = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn)
                - qgroup0.qVec.begin());
            size_t index = 0;
            size_t block = 0;

            qcomplex_t alpha, beta;

            qcomplex_t C00 = U00;
            qcomplex_t C01 = U01;
            qcomplex_t C10 = U10;
            qcomplex_t C11 = U11;
            if (isConjugate)
            {
                qcomplex_t temp;
                C00 = qcomplex_t(C00.real(), -C00.imag());
                C01 = qcomplex_t(C01.real(), -C01.imag());
                C10 = qcomplex_t(C10.real(), -C10.imag());
                C11 = qcomplex_t(C11.real(), -C11.imag());
                temp = C01;
                C01 = U10;
                C10 = temp;
            }
            Qnum qvtemp;
            for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
            {
                size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
                    - qgroup0.qVec.begin());
                block += 1ull << stemp;
                qvtemp.push_back(stemp);
            }
            sort(qvtemp.begin(), qvtemp.end());
            Qnum::iterator qiter;
            size_t j;
            //#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
            for (size_t i = 0; i < M; i++)
            {
                index = 0;
                x = i;
                qiter = qvtemp.begin();

                for (j = 0; j < n; j++)
                {
                    while (qiter != qvtemp.end() && *qiter == j)
                    {
                        qiter++;
                        j++;
                    }
                    //index += ((x % 2)*(1ull << j));
                    index += ((x & 1) << j);
                    x >>= 1;
                }

                /*
                * control qubits are 1,target qubit is 0
                */
                index = index + block - ststep;
                alpha = qgroup0.qstate[index];
                beta = qgroup0.qstate[index + ststep];
                qgroup0.qstate[index] = alpha * C00 + beta * C01;
                qgroup0.qstate[index + ststep] = alpha * C10 + beta * C11;
            }
        }
        return qErrorNone;
    }

    //single qubit gate and control-single qubit gate
    CONST_GATE(P0);
    CONST_GATE(P1);
    CONST_GATE(X);
    CONST_GATE(Y);
    CONST_GATE(Z);
    CONST_GATE(Hadamard);
    CONST_GATE(T);
    CONST_GATE(S);
    SINGLE_ANGLE_GATE(RX_GATE);
    SINGLE_ANGLE_GATE(RY_GATE);
    SINGLE_ANGLE_GATE(RZ_GATE);
    CONTROL_SINGLE_ANGLE_GATE(RX_GATE);
    CONTROL_SINGLE_ANGLE_GATE(RY_GATE);
    CONTROL_SINGLE_ANGLE_GATE(RZ_GATE);
    CONTROL_CONST_GATE(Hadamard);
    CONTROL_CONST_GATE(X);             //CCCC-NOT
    CONTROL_CONST_GATE(Y);
    CONTROL_CONST_GATE(Z);
    CONTROL_CONST_GATE(T);
    CONTROL_CONST_GATE(S);
    CONTROL_CONST_GATE(P0);
    CONTROL_CONST_GATE(P1);

    //define const CNOT,CZ,ISWAP,SQISWAP
    inline QError CNOT(size_t qn_0, size_t qn_1, 
		bool isConjugate, double error_rate)
    {
        Qnum qvtemp;
        qvtemp.push_back(qn_0);
        qvtemp.push_back(qn_1);
        X(qn_1, qvtemp, isConjugate, error_rate);           //qn_1 is target
        return qErrorNone;
    }
    inline QError CNOT(size_t qn_0, size_t qn_1, Qnum& vControlBit, 
		bool isConjugate, double error_rate)
    {
        X(qn_1, vControlBit, isConjugate, error_rate);      //qn_1 is target
        return qErrorNone;
    }

	QError iSWAP(size_t qn_0, size_t qn_1, double theta, 
		bool isConjugate, double);
	QError iSWAP(size_t qn_0, size_t qn_1, Qnum& vControlBit, 
		double theta, bool isConjugate, double);

    inline QError iSWAP(size_t qn_0, size_t qn_1, 
		bool isConjugate, double error_rate)
    {
        iSWAP(qn_0, qn_1, PI / 2, isConjugate, error_rate);
        return qErrorNone;
    }
    inline QError iSWAP(size_t qn_0, size_t qn_1, Qnum& vControlBit, 
		bool isConjugate, double error_rate)
    {
        iSWAP(qn_0, qn_1, vControlBit, PI / 2, isConjugate, error_rate);
        return qErrorNone;
    }

    inline QError SqiSWAP(size_t qn_0, size_t qn_1, 
		bool isConjugate, double error_rate)
    {
        iSWAP(qn_0, qn_1, PI / 4, isConjugate, error_rate);
        return qErrorNone;
    }
    inline QError SqiSWAP(size_t qn_0, size_t qn_1, Qnum& vControlBit, 
		bool isConjugate, double error_rate)
    {
        iSWAP(qn_0, qn_1, vControlBit, PI / 4, isConjugate, error_rate);
        return qErrorNone;
    }

	QError CR(size_t qn_0, size_t qn_1, 
		double theta, bool isConjugate, double error_rate);
	QError CR(size_t qn_0, size_t qn_1, Qnum& vControlBit, 
		double theta, bool isConjugate, double error_rate);

    inline QError CZ(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
    {
        CR(qn_0, qn_1, PI, isConjugate, error_rate);
        return qErrorNone;
    }
    inline QError CZ(size_t qn_0, size_t qn_1, Qnum& vControlBit, bool isConjugate, double error_rate)
    {
        CR(qn_0, qn_1, vControlBit, PI, isConjugate, error_rate);
        return qErrorNone;
    }

    //define unitary single/double quantum gate
    QError unitarySingleQubitGate(size_t qn, 
		QStat& matrix, bool isConjugate, double error_rate, GateType);
    QError controlunitarySingleQubitGate(size_t qn, Qnum& vControlBit, 
		QStat& matrix, bool isConjugate, double error_rate, GateType);
    QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1, 
		QStat& matrix, bool isConjugate, double error_rate, GateType);
    QError controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum& vControlBit, 
		QStat& matrix, bool isConjugate, double error_rate, GateType);
    QError DiagonalGate(Qnum& vQubit, QStat & matrix,
        bool isConjugate, double error_rate);
    QError controlDiagonalGate(Qnum& vQubit, QStat & matrix, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QStat getQState();
    QError Reset(size_t qn);
    bool qubitMeasure(size_t qn);
    QError pMeasure(Qnum& qnum, std::vector<std::pair<size_t, double>> &mResult, 
		int select_max=-1);
    QError pMeasure(Qnum& qnum, std::vector<double> &mResult);
    QError initState(QuantumGateParam *);    
    QError endGate(QuantumGateParam *pQuantumProParam, QPUImpl * pQGate);
};

#endif
