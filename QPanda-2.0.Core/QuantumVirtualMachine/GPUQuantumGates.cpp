/******************************************************************************
Copyright (c) 2017-2018 Origin Quantum Computing Co., Ltd.. All Rights Reserved.



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
#include "GPUQuantumGates.h"
#ifdef USE_CUDA
#include "GPUGatesDecl.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>
using std::stringstream;

GPUQuantumGates::~GPUQuantumGates()
{
	if (miQbitNum > 0)
		GATEGPU::destroyState(mvCPUQuantumStat, mvQuantumStat, miQbitNum);
}

size_t GPUQuantumGates::getQStateSize()
{
	if (!mbIsInitQState)
		return 0;
	else
	{
		return 1u << miQbitNum;
	}
}

/*****************************************************************************************************************
Name:        initState
Description: initialize the quantum state
Argin:       stNumber  Quantum number
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError GPUQuantumGates::initState(QuantumGateParam * pQuantumProParam)
{
    miQbitNum     = pQuantumProParam->mQuantumBitNumber;
    
    if (!initstate(mvCPUQuantumStat, mvQuantumStat, pQuantumProParam->mQuantumBitNumber))
    {
        return undefineError;
    }
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
bool GPUQuantumGates::getQState(string & sState, QuantumGateParam *pQuantumProParam)
{
	if (miQbitNum <= 0)
		return false;

	getState(mvCPUQuantumStat, mvQuantumStat, pQuantumProParam->mQuantumBitNumber);
	size_t uiDim = 1u << (pQuantumProParam->mQuantumBitNumber);
	stringstream ssTemp;
	for (size_t i = 0; i < uiDim; i++)
	{
		ssTemp << "state[" << i << "].real = "
			<< mvCPUQuantumStat.real[i]
			<< " " << "state[" << i << "].imag = "
			<< mvCPUQuantumStat.imag[i] << "\n";
	}
	sState.append(ssTemp.str());
	return true;
}

/*****************************************************************************************************************
Name:        endGate
Description: end gate
Argin:       pQuantumProParam       quantum program param pointer
pQGate                 quantum gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError GPUQuantumGates::endGate(QuantumGateParam *pQuantumProParam, QuantumGates * pQGate)
{
	if (!clearState(mvCPUQuantumStat, mvQuantumStat, pQuantumProParam->mQuantumBitNumber))
	{
		return undefineError;
	}

	return qErrorNone;
}



/*****************************************************************************************************************
Name:        Hadamard
Description: Hadamard gate,the matrix is:[1/sqrt(2),1/sqrt(2);1/sqrt(2),-1/sqrt(2)]
Argin:       qn          qubit number that the Hadamard gate operates on.
error_rate   the errorrate of the gate
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError GPUQuantumGates::Hadamard(size_t qn, bool isConjugate, double error_rate)
{
	if (!GATEGPU::Hadamard(mvQuantumStat, qn, isConjugate, error_rate))
	{
		return undefineError;
	}
	return qErrorNone;
}

QError GPUQuantumGates::Hadamard(
	size_t qn,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlHadamard(mvQuantumStat, vControlBit, isConjugate, error_rate))
	{	
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::X(size_t qn, bool isConjugate, double error_rate)
{
	if (!GATEGPU::X(mvQuantumStat, qn, isConjugate, error_rate))
	{
		return undefineError;
	}
	return qErrorNone;
}
QError GPUQuantumGates::X(
	size_t qn,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlX(mvQuantumStat, vControlBit, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::Y(size_t qn, bool isConjugate, double error_rate)
{
	if (!GATEGPU::Y(mvQuantumStat, qn, isConjugate, error_rate))
	{
		return undefineError;
	}
	return qErrorNone;
}
QError GPUQuantumGates::Y(
	size_t qn,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlY(mvQuantumStat, vControlBit, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::Z(size_t qn, bool isConjugate, double error_rate)
{
	if (!GATEGPU::Z(mvQuantumStat, qn, isConjugate, error_rate))
	{
		return undefineError;
	}
	return qErrorNone;
}
QError GPUQuantumGates::Z(
	size_t qn,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlZ(mvQuantumStat, vControlBit, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}
QError GPUQuantumGates::S(size_t qn, bool isConjugate, double error_rate)
{
	if (!GATEGPU::S(mvQuantumStat, qn, isConjugate, error_rate))
	{
		return undefineError;
	}
	return qErrorNone;
}
QError GPUQuantumGates::S(
	size_t qn,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlS(mvQuantumStat, vControlBit, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}
QError GPUQuantumGates::T(size_t qn, bool isConjugate, double error_rate)
{
	if (!GATEGPU::T(mvQuantumStat, qn, isConjugate, error_rate))
	{
		return undefineError;
	}
	return qErrorNone;
}

QError GPUQuantumGates::T(
	size_t qn,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlT(mvQuantumStat, vControlBit, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}


QError GPUQuantumGates::RX_GATE(size_t qn, double theta,
	bool isConjugate, double error_rate)
{
	if (!GATEGPU::RX(mvQuantumStat, qn, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::RX_GATE(
	size_t qn,
	double theta,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlRX(mvQuantumStat, vControlBit, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::RY_GATE(size_t qn, double theta,
	bool isConjugate, double error_rate)
{
	if (!GATEGPU::RY(mvQuantumStat, qn, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::RY_GATE(
	size_t qn,
	double theta,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlRY(mvQuantumStat, vControlBit, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}


QError GPUQuantumGates::RZ_GATE(size_t qn, double theta,
	bool isConjugate, double error_rate)
{
	if (!GATEGPU::RZ(mvQuantumStat, qn, theta, isConjugate, error_rate))
	{
		return undefineError;
	}
	return qErrorNone;
}

QError GPUQuantumGates::RZ_GATE(
	size_t qn,
	double theta,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlRZ(mvQuantumStat, vControlBit, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}



//double quantum gate
QError GPUQuantumGates::CNOT(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
	if (!GATEGPU::CNOT(mvQuantumStat, qn_0, qn_1, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::CNOT(
	size_t qn_0,
	size_t qn_1,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlCNOT(mvQuantumStat, qn_0, qn_1, vControlBit, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::CZ(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
	if (!GATEGPU::CZ(mvQuantumStat, qn_0, qn_1, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::CZ(
	size_t qn_0,
	size_t qn_1,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlCZ(mvQuantumStat, qn_0, qn_1, vControlBit, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::CR(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
	if (!GATEGPU::CR(mvQuantumStat, qn_0, qn_1, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::CR(
	size_t qn_0,
	size_t qn_1,
	Qnum& vControlBit,
	double theta,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controlCR(mvQuantumStat, qn_0, qn_1, vControlBit, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::iSWAP(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
	if (!GATEGPU::iSWAP(mvQuantumStat, qn_0, qn_1, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::iSWAP(
	size_t qn_0,
	size_t qn_1,
	Qnum& vControlBit,
	double theta,
	bool isConjugate,
	double error_rate)
{
	if (!GATEGPU::controliSWAP(mvQuantumStat, qn_0, qn_1, vControlBit, theta, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

//pi/2 iSWAP
QError GPUQuantumGates::iSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
	double half_pi = 3.1415926 / 2;

	if (!GATEGPU::iSWAP(mvQuantumStat, qn_0, qn_1, half_pi, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::iSWAP(
	size_t qn_0,
	size_t qn_1,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	double half_pi = 3.1415926 / 2;

	if (!GATEGPU::controliSWAP(mvQuantumStat, qn_0, qn_1, vControlBit, half_pi, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}


//pi/4 SqiSWAP
QError GPUQuantumGates::SqiSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{

	double quarter_pi = 3.1415926 / 4;

	if (!GATEGPU::iSWAP(mvQuantumStat, qn_0, qn_1, quarter_pi, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}
QError GPUQuantumGates::SqiSWAP(
	size_t qn_0,
	size_t qn_1,
	Qnum& vControlBit,
	bool isConjugate,
	double error_rate)
{
	double quarter_pi = 3.1415926 / 4;

	if (!GATEGPU::controliSWAP(mvQuantumStat, qn_0, qn_1, vControlBit, quarter_pi, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}


QError GPUQuantumGates::unitarySingleQubitGate(
	size_t qn,
	QStat & matrix,
	bool isConjugate,
	double error_rate)
{
	STATE_T matrix_real[4] = { matrix[0].real(),matrix[1].real(), matrix[2].real(), matrix[3].real()};
	STATE_T matrix_imag[4] = { matrix[0].imag(),matrix[1].imag(), matrix[2].imag(), matrix[3].imag() };
	GATEGPU::QState gpu_matrix;
	gpu_matrix.real = matrix_real;
	gpu_matrix.imag = matrix_imag;
	if (!GATEGPU::unitarysingle(mvQuantumStat, qn, gpu_matrix, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::controlunitarySingleQubitGate(
	size_t qn,
	Qnum& qnum,
	QStat& matrix,
	bool isConjugate,
	double error_rate)
{
	STATE_T matrix_real[4] = { matrix[0].real(),matrix[1].real(), matrix[2].real(), matrix[3].real() };
	STATE_T matrix_imag[4] = { matrix[0].imag(),matrix[1].imag(), matrix[2].imag(), matrix[3].imag() };
	GATEGPU::QState gpu_matrix;
	gpu_matrix.real = matrix_real;
	gpu_matrix.imag = matrix_imag;
	if (!GATEGPU::controlunitarysingle(mvQuantumStat, qnum, gpu_matrix, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}


QError GPUQuantumGates::unitaryDoubleQubitGate(
	size_t qn_0,
	size_t qn_1,
	QStat& matrix,
	bool isConjugate,
	double error_rate)
{
	STATE_T matrix_real[16];
	STATE_T matrix_imag[16];
	for (int i = 0; i < 16; i++)
	{
		matrix_real[i] = matrix[i].real();
		matrix_imag[i] = matrix[i].imag();
	}
	GATEGPU::QState gpu_matrix;
	gpu_matrix.real = matrix_real;
	gpu_matrix.imag = matrix_imag;
	if (!GATEGPU::unitarydouble(mvQuantumStat, qn_0, qn_1, gpu_matrix, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}

QError GPUQuantumGates::controlunitaryDoubleQubitGate(
	size_t qn_0,
	size_t qn_1,
	Qnum& qnum,
	QStat& matrix,
	bool isConjugate,
	double error_rate)
{
	STATE_T matrix_real[16];
	STATE_T matrix_imag[16];
	for (int i = 0; i < 16; i++)
	{
		matrix_real[i] = matrix[i].real();
		matrix_imag[i] = matrix[i].imag();
	}
	GATEGPU::QState gpu_matrix;
	gpu_matrix.real = matrix_real;
	gpu_matrix.imag = matrix_imag;
	if (!GATEGPU::controlunitarydouble(mvQuantumStat, qnum, gpu_matrix, isConjugate, error_rate))
	{
		return undefineError;
	}

	return qErrorNone;
}


/*****************************************************************************************************************
Name:        Reset
Description: reset bit gate
Argin:       qn          qubit number that the Hadamard gate operates on.
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError GPUQuantumGates::Reset(size_t qn)
{

	if (!GATEGPU::qbReset(mvQuantumStat, qn, 0))
	{
		return undefineError;
	}
	return qErrorNone;
}


bool GPUQuantumGates::qubitMeasure(size_t qn)
{
	return GATEGPU::qubitmeasure(mvQuantumStat, 1 << qn);
}
/*****************************************************************************************************************
Name:        pMeasure
Description: pMeasure gate
Argin:       qnum        qubit bit number vector
mResult     reuslt vector
Argout:      None
return:      quantum error
*****************************************************************************************************************/
QError GPUQuantumGates::pMeasure(Qnum& qnum, vector<pair<size_t, double>> &mResult,int select_max)
{
	if (!GATEGPU::pMeasurenew(mvQuantumStat, mResult, qnum))
	{
		return undefineError;
	}
	return qErrorNone;
}


QError GPUQuantumGates::pMeasure(Qnum& qnum, vector<double> &mResult)
{
	vector<pair<size_t, double>> pair_reuslt;
	if (!GATEGPU::pMeasurenew(mvQuantumStat, pair_reuslt, qnum))
	{
		return undefineError;
	}

	for (auto aiter : pair_reuslt)
	{
		mResult.push_back(aiter.second);
	}

	return qErrorNone;

}


#endif
