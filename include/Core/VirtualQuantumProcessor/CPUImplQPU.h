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

#ifndef CPU_QUANTUM_GATE_H
#define CPU_QUANTUM_GATE_H

#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/Utilities/Tools/Utils.h"
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


/**
* @brief QPU implementation by  CPU model
* @ingroup VirtualQuantumProcessor
*/
class CPUImplQPU : public QPUImpl
{
public:
    CPUImplQPU();
    CPUImplQPU(size_t qubit_num);
    ~CPUImplQPU();

    template<const qcomplex_t& U00, const qcomplex_t& U01, const qcomplex_t& U10, const qcomplex_t& U11>
    QError single_gate(size_t qn, bool is_dagger, double error_rate)
    {
        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, matrix, is_dagger);

        return qErrorNone;
    }


    QError U1_GATE(size_t qn, double theta, bool is_dagger, double error_rate)
    {
        QStat matrix = { 1, 0, 0, qcomplex_t(cos(theta),sin(theta)) };
        _U1(qn, matrix, is_dagger);
        return qErrorNone;
    }

	QError P_GATE(size_t qn, double theta, bool is_dagger, double error_rate)
	{
		QStat matrix = { 1, 0, 0, qcomplex_t(cos(theta),sin(theta)) };
		_U1(qn, matrix, is_dagger);
		return qErrorNone;
	}


    template<const double& Nx, const double& Ny, const double& Nz>
    QError single_angle_gate(size_t qn, double theta, bool is_dagger, double error_rate)
    {
        qcomplex_t U00(cos(theta / 2), -sin(theta / 2)*Nz);
        qcomplex_t U01(-sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U10(sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U11(cos(theta / 2), sin(theta / 2)*Nz);

        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, matrix, is_dagger);

        return qErrorNone;
    }

    template<const double& Nx, const double& Ny, const double& Nz>
    QError control_single_angle_gate(size_t qn,
        double theta,
        Qnum vControlBit,
        bool is_dagger,
        double error_rate)
    {
        qcomplex_t U00(cos(theta / 2), -sin(theta / 2)*Nz);
        qcomplex_t U01(-sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U10(sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U11(cos(theta / 2), sin(theta / 2)*Nz);

        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, vControlBit, matrix, is_dagger);
        return qErrorNone;
    }

    template<const qcomplex_t& U00,
        const qcomplex_t& U01,
        const qcomplex_t& U10,
        const qcomplex_t& U11>
        QError control_single_gate(
            size_t qn,
            Qnum  vControlBit,
            bool is_dagger,
            double error_rate)
    {
        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, vControlBit, matrix, is_dagger);
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
        _CNOT(qn_0, qn_1);
        return qErrorNone;
    }

    inline QError CNOT(size_t qn_0, size_t qn_1, Qnum& vControlBit,
        bool isConjugate, double error_rate)
    {
        _CNOT(qn_0, qn_1, vControlBit);
        return qErrorNone;
    }

    inline QError iSWAP(size_t qn_0, size_t qn_1, double theta,
        bool isConjugate, double)
    {
        QStat matrix = { 1, 0, 0, 0,
                        0, std::cos(theta), qcomplex_t(0,-std::sin(theta)), 0,
                        0, qcomplex_t(0,-std::sin(theta)), std::cos(theta), 0,
                        0, 0, 0, 1 };
        _iSWAP_theta(qn_0, qn_1, matrix, isConjugate);
        return qErrorNone;
    }


    inline QError iSWAP(size_t qn_0, size_t qn_1, Qnum& vControlBit,
        double theta, bool isConjugate, double)
    {
        QStat matrix = { 1, 0, 0, 0,
                        0, std::cos(theta), qcomplex_t(0,-std::sin(theta)), 0,
                        0, qcomplex_t(0,-std::sin(theta)), std::cos(theta), 0,
                        0, 0, 0, 1 };
        _iSWAP_theta(qn_0, qn_1, matrix, isConjugate, vControlBit);
        return qErrorNone;
    }

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

    inline QError CR(size_t qn_0, size_t qn_1,
        double theta, bool isConjugate, double error_rate)
    {
        QStat matrix = { 1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, qcomplex_t(std::cos(theta), std::sin(theta)) };
        _CR(qn_0, qn_1, matrix, isConjugate);
        return qErrorNone;
    }

    inline QError CR(size_t qn_0, size_t qn_1, Qnum& vControlBit,
        double theta, bool isConjugate, double error_rate)
    {
        QStat matrix = { 1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, qcomplex_t(std::cos(theta), std::sin(theta)) };
        _CR(qn_0, qn_1, matrix, isConjugate, vControlBit);
        return qErrorNone;
    }

	inline QError CP(size_t qn_0, size_t qn_1,
		double theta, bool isConjugate, double error_rate)
	{
		QStat matrix = { 1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						0, 0, 0, qcomplex_t(std::cos(theta), std::sin(theta)) };
		_CP(qn_0, qn_1, matrix, isConjugate);
		return qErrorNone;
	}

	inline QError CP(size_t qn_0, size_t qn_1, Qnum& vControlBit,
		double theta, bool isConjugate, double error_rate)
	{
		QStat matrix = { 1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						0, 0, 0, qcomplex_t(std::cos(theta), std::sin(theta)) };
		_CP(qn_0, qn_1, matrix, isConjugate, vControlBit);
		return qErrorNone;
	}

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
        QStat& matrix, bool is_dagger,
        GateType);
    QError controlunitarySingleQubitGate(size_t qn, Qnum& controls,
        QStat& matrix, bool is_dagger,
        GateType type);
    QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
        QStat& matrix, bool is_dagger,
        GateType);
    QError controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum& controls,
        QStat& matrix, bool is_dagger,
        GateType);
    QError DiagonalGate(Qnum& vQubit, QStat & matrix,
        bool isConjugate, double error_rate);
    QError controlDiagonalGate(Qnum& vQubit, QStat & matrix, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QStat getQState();
    QError Reset(size_t qn);
    bool qubitMeasure(size_t qn);
    QError pMeasure(Qnum& qnum, prob_tuple &probs,
        int select_max = -1);
    QError pMeasure(Qnum& qnum, prob_vec &probs);
    QError initState(size_t head_rank, size_t rank_size, size_t qubit_num);
    QError initState(size_t qubit_num, const QStat &state = {});

    inline QError P00(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
    {
        QStat P00_matrix = { 1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,0 };
        return unitaryDoubleQubitGate(qn_0, qn_1, P00_matrix, isConjugate, GateType::P00_GATE);
    }

    inline QError SWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
    {
        QStat P00_matrix = { 1,0,0,0,
            0,0,1,0,
            0,1,0,0,
            0,0,0,1 };
        return unitaryDoubleQubitGate(qn_0, qn_1, P00_matrix, isConjugate, GateType::SWAP_GATE);
    }

    inline QError P11(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
    {
        QStat P11_matrix = { 0,0,0,0,
            0,0,0,0,
            0,0,0,0,
            0,0,0,1 };
        return unitaryDoubleQubitGate(qn_0, qn_1, P11_matrix, isConjugate, GateType::P11_GATE);
    }

protected:
    QError _single_qubit_normal_unitary(size_t qn, QStat& matrix, bool is_dagger);
    QError _single_qubit_normal_unitary(size_t qn, Qnum& controls, QStat& matrix, bool is_dagger);

    QError _double_qubit_normal_unitary(size_t qn_0, size_t qn_1, QStat& matrix, bool is_dagger);
    QError _double_qubit_normal_unitary(size_t qn_0, size_t qn_1, Qnum& controls, QStat& matrix, bool is_dagger);

    QError _X(size_t qn);
    QError _Y(size_t qn);
    QError _Z(size_t qn);
    QError _S(size_t qn, bool is_dagger);
    QError _U1(size_t qn, QStat &matrix, bool is_dagger);
	QError _P(size_t qn, QStat &matrix, bool is_dagger);
    QError _RZ(size_t qn, QStat &matrix, bool is_dagger);
    QError _H(size_t qn, QStat &matrix);

    QError _CNOT(size_t qn_0, size_t qn_1);
    QError _CZ(size_t qn_0, size_t qn_1);
    QError _CR(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
	QError _CP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
    QError _SWAP(size_t qn_0, size_t qn_1);
    QError _iSWAP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
    QError _iSWAP_theta(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
    QError _CU(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);

    QError _X(size_t qn, Qnum &controls);
    QError _Y(size_t qn, Qnum &controls);
    QError _Z(size_t qn, Qnum &controls);
    QError _S(size_t qn, bool is_dagger, Qnum &controls);
    QError _U1(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls);
	QError _P(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _RZ(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _H(size_t qn, QStat &matrix, Qnum &controls);

    QError _CNOT(size_t qn_0, size_t qn_1, Qnum &controls);
    QError _CZ(size_t qn_0, size_t qn_1, Qnum &controls);
    QError _CR(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
	QError _CP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _SWAP(size_t qn_0, size_t qn_1, Qnum &controls);
    QError _iSWAP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _iSWAP_theta(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _CU(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);

    inline int64_t _insert(int64_t value, size_t n1, size_t n2)
    {
        if (n1 > n2)
        {
            std::swap(n1, n2);
        }

        int64_t mask1 = (1ll << n1) - 1;
        int64_t mask2 = (1ll << (n2 - 1)) - 1;
        int64_t z = value & mask1;
        int64_t y = ~mask1 & value & mask2;
        int64_t x = ~mask2 & value;

        return ((x << 2) | (y << 1) | z);
    }

    inline int64_t _insert(int64_t value, size_t n)
    {
        int64_t number = 1ll << n;
        if (value < number)
        {
            return value;
        }

        int64_t mask = number - 1;
        int64_t x = mask & value;
        int64_t y = ~mask & value;
        return ((y << 1) | x);
    }
    void _verify_state(const QStat &state);
    inline int _omp_thread_num(size_t size);
private:
    bool m_is_init_state{false};
    QStat m_state;
    QStat m_init_state;
    size_t m_qubit_num;
    const int64_t m_threshold = 1ll << 9;
};

class CPUImplQPUWithOracle : public CPUImplQPU {
public:
    QError controlOracularGate(std::vector<size_t> bits,
        std::vector<size_t> controlbits,
        bool is_dagger,
        std::string name);
};

#endif
