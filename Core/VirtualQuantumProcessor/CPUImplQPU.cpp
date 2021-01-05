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
#include "QPandaConfig.h"
#include "CPUImplQPU.h"
#include "QPandaNamespace.h"
#include "Core/Utilities/Tools/Utils.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "ThirdParty/Eigen/Eigen"

using namespace std;
using namespace Eigen;

using qmatrix2cf_t = Eigen::Matrix<qcomplex_t, 2, 2, Eigen::RowMajor>;
using qmatrix4cf_t = Eigen::Matrix<qcomplex_t, 4, 4, Eigen::RowMajor>;

REGISTER_GATE_MATRIX(X, 0, 1, 1, 0)
REGISTER_GATE_MATRIX(Hadamard, SQ2, SQ2, SQ2, -SQ2)
REGISTER_GATE_MATRIX(Y, 0, -iunit, iunit, 0)
REGISTER_GATE_MATRIX(Z, 1, 0, 0, -1)
REGISTER_GATE_MATRIX(P0, 1, 0, 0, 0)
REGISTER_GATE_MATRIX(P1, 0, 0, 0, 1)
REGISTER_GATE_MATRIX(T, 1, 0, 0, (qstate_type)SQ2 + iunit * (qstate_type)SQ2)
REGISTER_GATE_MATRIX(S, 1, 0, 0, iunit)
REGISTER_ANGLE_GATE_MATRIX(RX_GATE, 1, 0, 0)
REGISTER_ANGLE_GATE_MATRIX(RY_GATE, 0, 1, 0)
REGISTER_ANGLE_GATE_MATRIX(RZ_GATE, 0, 0, 1)

static uint64_t insert(int value, int n1, int n2)
{
    if (n1 > n2)
    {
        std::swap(n1, n2);
    }

    uint64_t mask1 = (1ll << n1) - 1;
    uint64_t mask2 = (1ll << (n2 - 1)) - 1;
    uint64_t z = value & mask1;
    uint64_t y = ~mask1 & value & mask2;
    uint64_t x = ~mask2 & value;

    return ((x << 2) | (y << 1) | z);
}

static uint64_t insert(int value, int n)
{
    uint64_t number = 1ll << n;
    if (value < number)
    {
        return value;
    }

    uint64_t mask = number - 1;
    uint64_t x = mask & value;
    uint64_t y = ~mask & value;
    return ((y << 1) | x);
}


CPUImplQPU::CPUImplQPU()
{
}

CPUImplQPU::~CPUImplQPU()
{
    qubit2stat.clear();
}
CPUImplQPU::CPUImplQPU(size_t qubitSumNumber) :qubit2stat(qubitSumNumber)
{
}

QGateParam& CPUImplQPU::findgroup(size_t qn)
{
    for (auto iter = qubit2stat.begin(); iter != qubit2stat.end(); ++iter)
    {
        if (iter->enable == false) continue;
        if (find(iter->qVec.begin(), iter->qVec.end(), qn) != iter->qVec.end()) return *iter;
    }
    QCERR("unknow error");
    throw runtime_error("unknow error");
}

static bool probcompare(pair<size_t, double> a, pair<size_t, double> b)
{
    return a.second > b.second;
}

QError CPUImplQPU::pMeasure(Qnum& qnum, prob_tuple &mResult, int select_max)
{
    mResult.resize(1ull << qnum.size());
    QGateParam& group0 = findgroup(qnum[0]);
    for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
    {
        TensorProduct(group0, findgroup(*iter));
    }

    for (size_t i = 0; i < 1ull << qnum.size(); i++)
    {
        mResult[i].first = i;
        mResult[i].second = 0;
    }
    Qnum qvtemp;
    for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
    {
        qvtemp.push_back(find(group0.qVec.begin(), group0.qVec.end(), *iter) - group0.qVec.begin());
    }

    for (size_t i = 0; i < group0.qstate.size(); i++)
    {
        size_t idx = 0;
        for (size_t j = 0; j < qvtemp.size(); j++)
        {
            idx += (((i >> (qvtemp[j])) % 2) << j);
        }
        mResult[idx].second += abs(group0.qstate[i])*abs(group0.qstate[i]);
    }

    if (select_max == -1)
    {
        sort(mResult.begin(), mResult.end(), probcompare);
        return qErrorNone;
    }
    else if (mResult.size() <= select_max)
    {
        sort(mResult.begin(), mResult.end(), probcompare);
        return qErrorNone;
    }
    else
    {
        sort(mResult.begin(), mResult.end(), probcompare);
        mResult.erase(mResult.begin() + select_max, mResult.end());
    }

    return qErrorNone;
}

/**
Note: Added by Agony5757, it is a simplified version of pMeasure and with
more efficiency.
**/
QError CPUImplQPU::pMeasure(Qnum& qnum, prob_vec &mResult)
{
    mResult.resize(1ull << qnum.size());
    QGateParam& group0 = findgroup(qnum[0]);
    for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
    {
        TensorProduct(group0, findgroup(*iter));
    }

    Qnum qvtemp;
    for (auto iter = qnum.begin(); iter != qnum.end(); iter++)
    {
        qvtemp.push_back(find(group0.qVec.begin(), group0.qVec.end(), *iter) - group0.qVec.begin());
    }

    for (size_t i = 0; i < group0.qstate.size(); i++)
    {
        size_t idx = 0;
        for (size_t j = 0; j < qvtemp.size(); j++)
        {
            idx += (((i >> (qvtemp[j])) % 2) << j);
        }
        mResult[idx] += group0.qstate[i].real()*group0.qstate[i].real()
            + group0.qstate[i].imag()*group0.qstate[i].imag();
    }

    return qErrorNone;
}

bool CPUImplQPU::qubitMeasure(size_t qn)
{
    QGateParam& qgroup = findgroup(qn);
    size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    double dprob(0);

    for (size_t i = 0; i< qgroup.qstate.size(); i += ststep * 2)
    {
        for (size_t j = i; j<i + ststep; j++)
        {
            dprob += abs(qgroup.qstate[j])*abs(qgroup.qstate[j]);
        }
    }
    int ioutcome(0);

    float fi = (float)QPanda::RandomNumberGenerator();

    if (fi> dprob)
    {
        ioutcome = 1;
    }

    /*
    *  POVM measurement
    */
    if (ioutcome == 0)
    {
        dprob = 1 / sqrt(dprob);

        size_t j;
        //#pragma omp parallel for private(j)
        for (size_t i = 0; i < qgroup.qstate.size(); i = i + 2 * ststep)
        {
            for (j = i; j < i + ststep; j++)
            {
                qgroup.qstate[j] *= dprob;
                qgroup.qstate[j + ststep] = 0;
            }
        }
    }
    else
    {
        dprob = 1 / sqrt(1 - dprob);

        size_t j;
        //#pragma omp parallel for private(j)
        for (size_t i = 0; i < qgroup.qstate.size(); i = i + 2 * ststep)
        {
            for (j = i; j<i + ststep; j++) {
                qgroup.qstate[j] = 0;
                qgroup.qstate[j + ststep] *= dprob;
            }
        }
    }
    return ioutcome;
}

QError CPUImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{
    if (init_qubit2stat.empty())
    {
        qubit2stat.erase(qubit2stat.begin(), qubit2stat.end());
        qubit2stat.resize(qubit_num);
        for (auto i = 0; i < qubit_num; i++)
        {
            qubit2stat[i].qVec.push_back(i);
            qubit2stat[i].qstate.push_back(1);
            qubit2stat[i].qstate.push_back(0);
            qubit2stat[i].qubitnumber = 1;
        }
    }
    else
    {
        qubit2stat.assign(init_qubit2stat.begin(), init_qubit2stat.end());
    }

    return qErrorNone;
}

QError CPUImplQPU::initState(size_t qubit_num,const QStat &state)
{
    init_qubit2stat.clear();

    if (!state.empty())
    {
        double probs = .0;
        for (auto amplitude : state)
        {
            probs += std::norm(amplitude);
        }

        if (qubit_num != (size_t)std::log2(state.size()) || std::abs(probs - 1.) > 1e-6)
        {
            QCERR("state error");
            throw std::runtime_error("state error");
        }

        init_qubit2stat.resize(qubit_num);
        for (auto i = 0; i < qubit_num; i++)
        {
            init_qubit2stat[0].qVec.push_back(i);
        }

        init_qubit2stat[0].qstate = state;
        init_qubit2stat[0].qubitnumber = 1;
        init_qubit2stat[0].enable = true;

        for (auto i = 1; i < qubit_num; i++)
        {
            init_qubit2stat[i].qVec.push_back(i);
            init_qubit2stat[i].qstate.push_back(1);
            init_qubit2stat[i].qstate.push_back(0);
            init_qubit2stat[i].qubitnumber = 1;
            init_qubit2stat[i].enable = false;
        }

    }
   
    return qErrorNone;
}

#ifdef _MSC_VER   
QError  CPUImplQPU::
unitarySingleQubitGate(size_t qn,
    QStat& matrix,
    bool isConjugate,
    GateType)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    QGateParam& qgroup = findgroup(qn);

    size_t n = find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    size_t ststep = 1ull << n;
    
    if (isConjugate)
    {
        qcomplex_t temp;
        temp = matrix[1];
        matrix[1] = matrix[2];
        matrix[2] = temp;  //convert
        for (size_t i = 0; i < 4; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }//dagger
    }

#pragma omp parallel for private(alpha, beta)
    for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
    {
        int64_t real00_idx = insert(i, n);
        int64_t real01_idx = real00_idx + ststep;

        alpha = qgroup.qstate[real00_idx];
        beta = qgroup.qstate[real01_idx];
        qgroup.qstate[real00_idx] = matrix[0] * alpha + matrix[1] * beta;
        qgroup.qstate[real01_idx] = matrix[2] * alpha + matrix[3] * beta;
    }

    return qErrorNone;
}

QError  CPUImplQPU::
controlunitarySingleQubitGate(size_t qn,
    Qnum& vControlBit,
    QStat & matrix,
    bool isConjugate,
    GateType)
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
    if (isConjugate)
    {
        qcomplex_t temp;
        temp = matrix[1];
        matrix[1] = matrix[2];
        matrix[2] = temp;  //转置
        for (size_t i = 0; i < 4; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }//共轭
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
#pragma omp parallel for private(j,index,x,qiter,alpha,beta)
    for (long long i = 0; i < (long long)M; i++)
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
        qgroup0.qstate[index] = alpha * matrix[0] + beta * matrix[1];
        qgroup0.qstate[index + ststep] = alpha * matrix[2] + beta * matrix[3];
    }
    return qErrorNone;
}

QError CPUImplQPU::
unitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool isConjugate,
    GateType)
{
    QGateParam& qgroup0 = findgroup(qn_0);
    QGateParam& qgroup1 = findgroup(qn_1);
    if (qgroup0.qVec[0] != qgroup1.qVec[0])
    {
        TensorProduct(qgroup0, qgroup1);
    }

    size_t n1 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0) - qgroup0.qVec.begin();
    size_t n2 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1) - qgroup0.qVec.begin();
    size_t ststep1 = 1ull << n1;
    size_t ststep2 = 1ull << n2;

    if (n1 < n2)
    {
        std::swap(n1, n2);
    }

    qcomplex_t phi00, phi01, phi10, phi11;
    auto stateSize = qgroup0.qstate.size();
    if (isConjugate)
    {
        qcomplex_t temp;
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = i + 1; j < 4; j++)
            {
                temp = matrix[4 * i + j];
                matrix[4 * i + j] = matrix[4 * j + i];
                matrix[4 * j + i] = temp;
            }
        }
        for (size_t i = 0; i < 16; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }//dagger
    }

#pragma omp parallel for private(phi00, phi01, phi10, phi11)
    for (int64_t i = 0; i < (stateSize >> 2); i++)
    {
        int64_t real00_idx = insert(i, n2, n1);
        phi00 = qgroup0.qstate[real00_idx];
        phi01 = qgroup0.qstate[real00_idx + ststep2];
        phi10 = qgroup0.qstate[real00_idx + ststep1];
        phi11 = qgroup0.qstate[real00_idx + ststep1 + ststep2];

        qgroup0.qstate[real00_idx] = matrix[0] * phi00 + matrix[1] * phi01
            + matrix[2] * phi10 + matrix[3] * phi11;
        qgroup0.qstate[real00_idx + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
            + matrix[6] * phi10 + matrix[7] * phi11;
        qgroup0.qstate[real00_idx + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
            + matrix[10] * phi10 + matrix[11] * phi11;
        qgroup0.qstate[real00_idx + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
            + matrix[14] * phi10 + matrix[15] * phi11;
    }

    return qErrorNone;

}

QError  CPUImplQPU::
controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    QStat& matrix,
    bool isConjugate,
    GateType)
{
    QGateParam& qgroup0 = findgroup(qn_0);
    QGateParam& qgroup1 = findgroup(qn_1);
    TensorProduct(qgroup0, qgroup1);
    for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
    {
        TensorProduct(qgroup0, findgroup(*iter));
    }
    qcomplex_t temp;
    if (isConjugate)
    {
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = i + 1; j < 4; j++)
            {
                temp = matrix[4 * i + j];
                matrix[4 * i + j] = matrix[4 * j + i];
                matrix[4 * j + i] = temp;
            }
        }
        for (size_t i = 0; i < 16; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }
    }//dagger

        //combine all qubits;
    size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());

    size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
        - qgroup0.qVec.begin());
    size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
        - qgroup0.qVec.begin());
    size_t block = 0;
    qcomplex_t phi00, phi01, phi10, phi11;
    Qnum qvtemp;
    for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
    {
        size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
            - qgroup0.qVec.begin());
        block += 1ull << stemp;
        qvtemp.push_back(stemp);
    }
    //block: all related qubits are 1,others are 0
    sort(qvtemp.begin(), qvtemp.end());
    Qnum::iterator qiter;
    size_t j;
    size_t index = 0;
    size_t x;
    size_t n = qgroup0.qVec.size();

#pragma omp parallel for private(j,index,x,qiter,phi00,phi01,phi10,phi11)
    for (long long i = 0; i < (long long)M; i++)
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
        index = index + block - ststep0 - ststep1;                             /*control qubits are 1,target qubit are 0 */
        phi00 = qgroup0.qstate[index];             //00
        phi01 = qgroup0.qstate[index + ststep1];   //01
        phi10 = qgroup0.qstate[index + ststep0];   //10
        phi11 = qgroup0.qstate[index + ststep0 + ststep1];  //11
        qgroup0.qstate[index] = matrix[0] * phi00 + matrix[1] * phi01
            + matrix[2] * phi10 + matrix[3] * phi11;
        qgroup0.qstate[index + ststep1] = matrix[4] * phi00 + matrix[5] * phi01
            + matrix[6] * phi10 + matrix[7] * phi11;
        qgroup0.qstate[index + ststep0] = matrix[8] * phi00 + matrix[9] * phi01
            + matrix[10] * phi10 + matrix[11] * phi11;
        qgroup0.qstate[index + ststep0 + ststep1] = matrix[12] * phi00 + matrix[13] * phi01
            + matrix[14] * phi10 + matrix[15] * phi11;
    }
    return qErrorNone;
}

#else
QError  CPUImplQPU::
unitarySingleQubitGate(size_t qn,
	QStat& matrix,
	bool isConjugate,
	GateType)
{
	qcomplex_t alpha;
	qcomplex_t beta;
	QGateParam& qgroup = findgroup(qn);

	size_t n = find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
	size_t ststep = 1ull << n;

	qmatrix2cf_t mat = qmatrix2cf_t::Map(&matrix[0]);
	if (isConjugate)
	{
		mat.adjointInPlace();
	}

#pragma omp parallel for private(alpha, beta)
	for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
	{
		int64_t real00_idx = insert(i, n);
		int64_t real01_idx = real00_idx + ststep;

		alpha = qgroup.qstate[real00_idx];
		beta = qgroup.qstate[real01_idx];

		Matrix<qcomplex_t, 2, 1> state_vec;
		state_vec << alpha, beta;
		state_vec = mat * state_vec;
		qgroup.qstate[real00_idx] = state_vec[0];
		qgroup.qstate[real01_idx] = state_vec[1];
	}

	return qErrorNone;
}

QError  CPUImplQPU::
controlunitarySingleQubitGate(size_t qn,
	Qnum& vControlBit,
	QStat & matrix,
	bool isConjugate,
	GateType)
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
	qmatrix2cf_t mat = qmatrix2cf_t::Map(&matrix[0]);
	if (isConjugate)
	{
		mat.adjointInPlace();
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
#pragma omp parallel for private(j,index,x,qiter,alpha,beta)
	for (long long i = 0; i < (long long)M; i++)
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
	
		Matrix<qcomplex_t, 2, 1> state_vec;
		state_vec << alpha, beta;
		state_vec = mat * state_vec;
		qgroup0.qstate[index] = state_vec[0];
		qgroup0.qstate[index + ststep] = state_vec[1];
	}
	return qErrorNone;
}

QError CPUImplQPU::
unitaryDoubleQubitGate(size_t qn_0,
	size_t qn_1,
	QStat& matrix,
	bool isConjugate,
	GateType)
{
	QGateParam& qgroup0 = findgroup(qn_0);
	QGateParam& qgroup1 = findgroup(qn_1);
	if (qgroup0.qVec[0] != qgroup1.qVec[0])
	{
		TensorProduct(qgroup0, qgroup1);
	}

	size_t n1 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0) - qgroup0.qVec.begin();
	size_t n2 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1) - qgroup0.qVec.begin();
	size_t ststep1 = 1ull << n1;
	size_t ststep2 = 1ull << n2;

	if (n1 < n2)
	{
		std::swap(n1, n2);
	}

	qcomplex_t phi00, phi01, phi10, phi11;
	auto stateSize = qgroup0.qstate.size();

	qmatrix4cf_t mat = qmatrix4cf_t::Map(&matrix[0]);
	if (isConjugate)
	{
		mat.adjointInPlace();
	}

#pragma omp parallel for private(phi00, phi01, phi10, phi11)
	for (int64_t i = 0; i < (stateSize >> 2); i++)
	{
		int64_t real00_idx = insert(i, n2, n1);
		phi00 = qgroup0.qstate[real00_idx];
		phi01 = qgroup0.qstate[real00_idx + ststep2];
		phi10 = qgroup0.qstate[real00_idx + ststep1];
		phi11 = qgroup0.qstate[real00_idx + ststep1 + ststep2];

		Matrix<qcomplex_t, 4, 1>  state_vec;
		state_vec << phi00, phi01, phi10, phi11;
		state_vec = mat * state_vec;

		qgroup0.qstate[real00_idx] = state_vec[0];
		qgroup0.qstate[real00_idx + ststep2] = state_vec[1];
		qgroup0.qstate[real00_idx + ststep1] = state_vec[2];
		qgroup0.qstate[real00_idx + ststep1 + ststep2] = state_vec[3];
	}

	return qErrorNone;
}

QError  CPUImplQPU::
controlunitaryDoubleQubitGate(size_t qn_0,
	size_t qn_1,
	Qnum& vControlBit,
	QStat& matrix,
	bool isConjugate,
	GateType)
{
	QGateParam& qgroup0 = findgroup(qn_0);
	QGateParam& qgroup1 = findgroup(qn_1);
	TensorProduct(qgroup0, qgroup1);
	for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
	{
		TensorProduct(qgroup0, findgroup(*iter));
	}

	qmatrix4cf_t mat = qmatrix4cf_t::Map(&matrix[0]);
	if (isConjugate)
	{
		mat.adjointInPlace();
	}//dagger

		//combine all qubits;
	size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());

	size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
		- qgroup0.qVec.begin());
	size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
		- qgroup0.qVec.begin());
	size_t block = 0;
	qcomplex_t phi00, phi01, phi10, phi11;
	Qnum qvtemp;
	for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
	{
		size_t stemp = (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), *iter)
			- qgroup0.qVec.begin());
		block += 1ull << stemp;
		qvtemp.push_back(stemp);
	}
	//block: all related qubits are 1,others are 0
	sort(qvtemp.begin(), qvtemp.end());
	Qnum::iterator qiter;
	size_t j;
	size_t index = 0;
	size_t x;
	size_t n = qgroup0.qVec.size();

#pragma omp parallel for private(j,index,x,qiter,phi00,phi01,phi10,phi11)
	for (long long i = 0; i < (long long)M; i++)
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
		index = index + block - ststep0 - ststep1;                             /*control qubits are 1,target qubit are 0 */
		phi00 = qgroup0.qstate[index];             //00
		phi01 = qgroup0.qstate[index + ststep1];   //01
		phi10 = qgroup0.qstate[index + ststep0];   //10
		phi11 = qgroup0.qstate[index + ststep0 + ststep1];  //11

		Matrix<qcomplex_t, 4, 1> state_vec;
		state_vec << phi00, phi01, phi10, phi11;
		state_vec = mat * state_vec;

		qgroup0.qstate[index] = state_vec[0];
		qgroup0.qstate[index + ststep1] = state_vec[1];
		qgroup0.qstate[index + ststep0] = state_vec[2];
		qgroup0.qstate[index + ststep0 + ststep1] = state_vec[3];
	}
	return qErrorNone;
}
#endif


QError CPUImplQPU::iSWAP(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    if (QPanda::RandomNumberGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);

        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());

        /*
        * iSWAP(qn_1,qn_2) is agree with
        * iSWAP(qn_2,qn_1)
        */
        if (ststep0 < ststep1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(0);
            compll.imag(1.0);
        }
        else
        {
            compll.real(0);
            compll.imag(-1.0);
        }

        qcomplex_t alpha, beta;

        /*
        *  traverse all the states
        */
        size_t j, k;
        //#pragma omp parallel for private(j,k,alpha,beta)
        for (size_t i = 0; i < qgroup0.qstate.size(); i = i + 2 * ststep0)
        {
            for (j = i + ststep1; j < i + ststep0; j = j + 2 * ststep1)
            {
                for (k = j; k < j + ststep1; k++)
                {
                    alpha = qgroup0.qstate[k];             //01
                    beta = qgroup0.qstate[k + sttemp];     //10

                    qgroup0.qstate[k] = (qstate_type)cos(theta)*alpha + compll * beta*(qstate_type)sin(theta);           /* k:|01>                               */
                    qgroup0.qstate[k + sttemp] = compll * (qstate_type)sin(theta)* alpha + (qstate_type)cos(theta)*beta;          /* k+sttemp:|10>                        */
                }
            }
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::iSWAP(size_t qn_0, size_t qn_1, Qnum & vControlBit, double theta, bool isConjugate, double error_rate)
{
    if (QPanda::RandomNumberGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            TensorProduct(qgroup0, findgroup(*iter));
        }
        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());
        if (ststep0 < ststep1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(0);
            compll.imag(1.0);
        }
        else
        {
            compll.real(0);
            compll.imag(-1.0);
        }

        size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());
        size_t block = 0;
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
        size_t index = 0;
        size_t x;
        size_t n = qgroup0.qVec.size();
        //#pragma omp parallel for private(j,alpha,beta,index,x,qiter)
        qcomplex_t alpha, beta;
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
                index += ((x & 1) << j);
                x >>= 1;
            }
            index = index + block - ststep0 - ststep1;                  /*control qubits are 1,target qubit are 0 */
            alpha = qgroup0.qstate[index + ststep1];             //01
            beta = qgroup0.qstate[index + ststep0];     //10
            qgroup0.qstate[index + ststep1] = (qstate_type)cos(theta)*alpha + compll * beta*(qstate_type)sin(theta);
            qgroup0.qstate[index + ststep0] = compll * (qstate_type)sin(theta)* alpha + (qstate_type)cos(theta)*beta;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::CR(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    if (QPanda::RandomNumberGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);

        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());

        /*
        * iSWAP(qn_1,qn_2) is agree with
        * iSWAP(qn_2,qn_1)
        */
        if (ststep0 < ststep1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(cos(theta));
            compll.imag(sin(theta));
        }
        else
        {
            compll.real(cos(theta));
            compll.imag(-sin(theta));
        }
        /*
        *  traverse all the states
        */
        size_t j, k;
        //#pragma omp parallel for private(j,k,alpha,beta)
        for (size_t i = ststep0; i < qgroup0.qstate.size(); i = i + 2 * ststep0)
        {
            for (j = i + ststep1; j < i + ststep0; j = j + 2 * ststep1)
            {
                for (k = j; k < j + ststep1; k++)
                {
                    qgroup0.qstate[k] = compll * qgroup0.qstate[k];
                }
            }
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::CR(size_t qn_0, size_t qn_1, Qnum & vControlBit, double theta, bool isConjugate, double error_rate)
{
    if (QPanda::RandomNumberGenerator() > error_rate)
    {

        QGateParam& qgroup0 = findgroup(qn_0);
        QGateParam& qgroup1 = findgroup(qn_1);
        TensorProduct(qgroup0, qgroup1);
        for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
        {
            TensorProduct(qgroup0, findgroup(*iter));
        }
        size_t sttemp = 0;
        size_t ststep0 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
            - qgroup0.qVec.begin());
        size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
            - qgroup0.qVec.begin());
        if (ststep0 < ststep1)
        {
            sttemp = ststep0;
            ststep0 = ststep1;
            ststep1 = sttemp;
        }
        sttemp = ststep0 - ststep1;
        qcomplex_t compll;
        if (!isConjugate)
        {
            compll.real(cos(theta));
            compll.imag(sin(theta));
        }
        else
        {
            compll.real(cos(theta));
            compll.imag(-sin(theta));
        }

        size_t M = 1ull << (qgroup0.qVec.size() - vControlBit.size());
        size_t block = 0;
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
        size_t index = 0;
        size_t x;
        size_t n = qgroup0.qVec.size();
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
                index += ((x & 1) << j);
                x >>= 1;
            }
            qgroup0.qstate[index + block] *= compll;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::Reset(size_t qn)
{
    QGateParam& qgroup = findgroup(qn);
    size_t j;
    size_t ststep = 1ull << (find(qgroup.qVec.begin(), qgroup.qVec.end(), qn)
        - qgroup.qVec.begin());
    //#pragma omp parallel for private(j,alpha,beta)
	double dsum = 0;
    for (size_t i = 0; i < qgroup.qstate.size(); i += ststep * 2)
    {
        for (j = i; j<i + ststep; j++)
        {
            qgroup.qstate[j + ststep] = 0;                              /* in j+ststep,the goal qubit is in |1> */
			dsum += (abs(qgroup.qstate[j])*abs(qgroup.qstate[j]) + abs(qgroup.qstate[j + ststep])*abs(qgroup.qstate[j + ststep]));
        }
    }

	dsum = sqrt(dsum);
	for (size_t i = 0; i < qgroup.qstate.size(); i++)
	{
		qgroup.qstate[i] /= dsum;
	}

    return qErrorNone;
}

QStat CPUImplQPU::getQState()
{
    if (0 == qubit2stat.size())
    {
        return QStat();
    }
    size_t sEnable = 0;
    while (!qubit2stat[sEnable].enable)
    {
        sEnable++;
    }
    for (auto i = sEnable; i < qubit2stat.size(); i++)
    {
        if (qubit2stat[i].enable)
        {
            TensorProduct(qubit2stat[sEnable], qubit2stat[i]);
        }
    }
    QStat state(qubit2stat[sEnable].qstate.size(), 0);
    size_t slabel = 0;
    for (auto i = 0; i < qubit2stat[sEnable].qstate.size(); i++)
    {
        slabel = 0;
        for (auto j = 0; j < qubit2stat[sEnable].qVec.size(); j++)
        {
            slabel += (((i >> j) % 2) << qubit2stat[sEnable].qVec[j]);
        }
        state[slabel] = qubit2stat[sEnable].qstate[i];
    }
    return state;
}

QError CPUImplQPU::DiagonalGate(Qnum & vQubit, QStat & matrix, bool isConjugate, double error_rate)
{

    QGateParam& qgroup0 = findgroup(vQubit[0]);
    for (auto iter = vQubit.begin() + 1; iter != vQubit.end(); iter++)
    {
        TensorProduct(qgroup0, findgroup(*iter));
    }
    size_t index = 0;
    if (isConjugate)
    {
        for (size_t i = 0; i < matrix.size(); i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }//共轭
    }
    size_t  j, k;
#pragma omp parallel for private(j,k,index)
    for (long long i = 0; i < qgroup0.qstate.size(); i++)
    {
        index = 0;
        for (j = 0; j < qgroup0.qVec.size(); j++)
        {
            for (k = 0; k < vQubit.size(); k++)
            {
                if (qgroup0.qVec[j] == vQubit[k])
                {
                    index += (i >> j) % 2 * (1 << k);
                }
            }
        }
        qgroup0.qstate[i] *= matrix[index];
    }
    return QError();
}

QError CPUImplQPU::controlDiagonalGate(Qnum & vQubit, QStat & matrix, Qnum & vControlBit, bool isConjugate, double error_rate)
{

    QGateParam& qgroup0 = findgroup(vQubit[0]);
    for (auto iter = vQubit.begin() + 1; iter != vQubit.end(); iter++)
    {
        TensorProduct(qgroup0, findgroup(*iter));
    }
    for (auto iter = vControlBit.begin(); iter != vControlBit.end(); iter++)
    {
        TensorProduct(qgroup0, findgroup(*iter));
    }
    if (isConjugate)
    {
        for (size_t i = 0; i < matrix.size(); i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }//共轭
    }
    size_t index = 0;
    size_t block = 0;
    size_t j, k;
#pragma omp parallel for private(j,k,index,block)
    for (long long i = 0; i < qgroup0.qstate.size(); i++)
    {
        index = 0;    // corresponding matrix number of i
        block = 0;    // The number of control bits is 1.
        for (j = 0; j < qgroup0.qVec.size(); j++)
        {
            for ( k = 0; k < vQubit.size(); k++)
            {
                if (qgroup0.qVec[j] == vQubit[k])
                {
                    index += (i >> j) % 2 * (1 << k);
                }
            }
            for ( k = 0; k < vControlBit.size(); k++)
            {
                if (qgroup0.qVec[j] == vControlBit[k] && (i << j) % 2 == 1)
                {
                    block++;
                }
            }
        }
        if (block == vControlBit.size())
        {
            qgroup0.qstate[i] *= matrix[index];
        }
    }
    return QError();
}

QError CPUImplQPUWithOracle::controlOracularGate(std::vector<size_t> bits, std::vector<size_t> controlbits, bool is_dagger, std::string name)
{
	if (name == "oracle_test") {

	}
	else {
		throw runtime_error("Not Implemented.");
	}
}
