/******************************************************************************
Copyright (c) 2017-2020 Origin Quantum Computing Co., Ltd.. All Rights Reserved.



Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language
governing permissions and
limitations under the License.

Author:Dou Menghan
Date:2017-11-10
Description:gpu quantum logic gates class
*****************************************************************************************************************/
#include "GPUImplQPU.h"
#ifdef USE_CUDA
using namespace std;
#include "QPandaNamespace.h"
#include "GPUGatesWrapper.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>

using std::stringstream;

GPUImplQPU::~GPUImplQPU()
{
    if (miQbitNum > 0)
        GATEGPU::destroyState(mvCPUQuantumStat, mvQuantumStat, miQbitNum);

    if (m_probgpu != nullptr)
    {
        GATEGPU::gpuFree(m_probgpu);
    }
    if (m_resultgpu != nullptr)
    {
        GATEGPU::gpuFree(m_resultgpu);
    }
}

size_t GPUImplQPU::getQStateSize()
{
    if (!mbIsInitQState)
        return 0;
    else
    {
        return 1ull << miQbitNum;
    }
}

/*****************************************************************************************************************
Name:        initState
Description: initialize the quantum state
Argin:       stNumber  Quantum number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError GPUImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{
    if (!GATEGPU::initstate(mvCPUQuantumStat, mvQuantumStat, qubit_num))
    {
        return undefineError;
    }
    miQbitNum = qubit_num;
    mbIsInitQState = true;
    return qErrorNone;
}


/*****************************************************************************************************************
Name:        getQState
Description: get quantum state
Argin:       pQuantumProParam       quantum program prarm pointer
Argout:      sState                 string state
return:      quantum error
*****************************************************************************************************************/
QStat GPUImplQPU::getQState()
{
	if (miQbitNum <= 0)
	{
        return QStat();
	}

    GATEGPU::getState(mvCPUQuantumStat, mvQuantumStat, miQbitNum);
    size_t uiDim = 1ull << miQbitNum;
    stringstream ssTemp;
	QStat temp;

    for (size_t i = 0; i < uiDim; i++)
    {
        qcomplex_t qstate = { mvCPUQuantumStat.real[i] ,mvCPUQuantumStat.imag[i] };
        temp.push_back(qstate);
    }
    return temp;
}

/*****************************************************************************************************************
Name:        endGate
Description: end gate
Argin:       pQuantumProParam       quantum program param pointer
pQGate                 quantum gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError GPUImplQPU::endGate(QPanda::TraversalConfig *pQuantumProParam, QPUImpl * pQGate)
{
    return qErrorNone;
}

QError GPUImplQPU::Hadamard(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::Hadamard(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::X(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::X(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::Y(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::Y(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::Z(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::Z(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::S(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::S(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::T(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::T(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}


QError GPUImplQPU::RX_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::RX_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::RY_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::RY_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}


QError GPUImplQPU::RZ_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::RZ_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}



//double quantum gate
QError GPUImplQPU::CNOT(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::CNOT(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::CZ(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::CZ(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::CR(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::CR(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    double theta,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::iSWAP(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::iSWAP(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    double theta,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

//pi/2 iSWAP
QError GPUImplQPU::iSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::iSWAP(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}


//pi/4 SqiSWAP
QError GPUImplQPU::SqiSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::SqiSWAP(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::U1_GATE(size_t qn, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::U1_GATE(size_t qn, Qnum & vControlBit, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::U2_GATE(size_t qn, double phi, double lambda, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::U2_GATE(size_t qn, Qnum & vControlBit, double phi, double lambda, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::U3_GATE(size_t qn, double theta, double phi, double lambda, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::U3_GATE(size_t qn, Qnum & vControlBit, double theta, double phi, double lambda, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::unitarySingleQubitGate(size_t qn,
    QStat & matrix,
    bool isConjugate,
    GateType type)
{
    gpu_qstate_t matrix_real[4] = { matrix[0].real(),matrix[1].real(), matrix[2].real(), matrix[3].real()};
    gpu_qstate_t matrix_imag[4] = { matrix[0].imag(),matrix[1].imag(), matrix[2].imag(), matrix[3].imag() };
    GATEGPU::QState gpu_matrix;
    gpu_matrix.real = matrix_real;
    gpu_matrix.imag = matrix_imag;
    if (!GATEGPU::unitarysingle(mvQuantumStat, qn, gpu_matrix, isConjugate))
    {
        return undefineError;
    }

    return qErrorNone;
}

QError GPUImplQPU::controlunitarySingleQubitGate(size_t qn,
    Qnum& qnum,
    QStat& matrix,
    bool isConjugate,
    GateType type)
{
    gpu_qstate_t matrix_real[4] = { matrix[0].real(),matrix[1].real(), matrix[2].real(), matrix[3].real() };
    gpu_qstate_t matrix_imag[4] = { matrix[0].imag(),matrix[1].imag(), matrix[2].imag(), matrix[3].imag() };
    GATEGPU::QState gpu_matrix;
    gpu_matrix.real = matrix_real;
    gpu_matrix.imag = matrix_imag;
    if (!GATEGPU::controlunitarysingle(mvQuantumStat, qnum, gpu_matrix, isConjugate))
    {
        return undefineError;
    }

    return qErrorNone;
}

QError GPUImplQPU::unitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool isConjugate,
    GateType type)
{
    gpu_qstate_t matrix_real[16];
    gpu_qstate_t matrix_imag[16];
    for (int i = 0; i < 16; i++)
    {
        matrix_real[i] = matrix[i].real();
        matrix_imag[i] = matrix[i].imag();
    }
    GATEGPU::QState gpu_matrix;
    gpu_matrix.real = matrix_real;
    gpu_matrix.imag = matrix_imag;
    if (!GATEGPU::unitarydouble(mvQuantumStat, qn_0, qn_1, gpu_matrix, isConjugate))
    {
        return undefineError;
    }

    return qErrorNone;
}

QError GPUImplQPU::controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    Qnum& qnum,
    QStat& matrix,
    bool isConjugate,
    GateType type)
{
    gpu_qstate_t matrix_real[16];
    gpu_qstate_t matrix_imag[16];
    for (int i = 0; i < 16; i++)
    {
        matrix_real[i] = matrix[i].real();
        matrix_imag[i] = matrix[i].imag();
    }
    GATEGPU::QState gpu_matrix;
    gpu_matrix.real = matrix_real;
    gpu_matrix.imag = matrix_imag;
    if (!GATEGPU::controlunitarydouble(mvQuantumStat, qnum, gpu_matrix, isConjugate))
    {
        return undefineError;
    }

    return qErrorNone;
}

QError GPUImplQPU::DiagonalGate(Qnum & vQubit, QStat & matrix, bool isConjugate, double error_rate)
{
   /* size_t sdimension = matrix.size();
    gpu_qstate_t matrix_real;
    gpu_qstate_t matrix_imag;
    if (!GATEGPU::DiagonalGate(mvQuantumStat, vQubit, matrix, isConjugate, error_rate))
    {
        return undefineError;
    }*/
    return qErrorNone;
}

QError GPUImplQPU::controlDiagonalGate(Qnum& vQubit, QStat & matrix, Qnum& vControlBit,
                                       bool isConjugate, double error_rate)
{
    /*if (!GATEGPU::controlDiagonalGate(mvQuantumStat, vQubit, matrix, vControlBit, isConjugate, error_rate))
    {
        return undefineError;
    }*/
    return qErrorNone;
}

QError GPUImplQPU::Reset(size_t qn)
{
    if (!GATEGPU::qbReset(mvQuantumStat, qn, 0))
    {
        return undefineError;
    }
    return qErrorNone;
}

bool GPUImplQPU::qubitMeasure(size_t qn)
{
    return GATEGPU::qubitmeasure(mvQuantumStat, 1ull << qn, m_resultgpu, m_probgpu);
}

QError GPUImplQPU::pMeasure(Qnum& qnum, prob_tuple &mResult,int select_max)
{
    if (!GATEGPU::pMeasurenew(mvQuantumStat, mResult, qnum, select_max))
    {
        return undefineError;
    }
    return qErrorNone;
}

QError GPUImplQPU::pMeasure(Qnum& qnum, prob_vec &mResult)
{
    if (!GATEGPU::pMeasure_no_index(mvQuantumStat, mResult, qnum))
    {
        return undefineError;
    }
    return qErrorNone;
}



QError GPUImplQPU::P0(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::P0(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::P1(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError GPUImplQPU::P1(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::I(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::I(size_t qn, Qnum & vControlBit, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError GPUImplQPU::initState(size_t qubit_num, const QStat &state)
{
    return undefineError;
}


#endif
