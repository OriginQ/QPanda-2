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
#include "CPUImplQPUSingleThread.h"
#include "QPandaNamespace.h"
#include "Core/Utilities/Tools/Utils.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>

using namespace std;

CPUImplQPUSingleThread::CPUImplQPUSingleThread()
{
}

CPUImplQPUSingleThread::~CPUImplQPUSingleThread()
{
    qubit2stat.clear();
}

CPUImplQPUSingleThread::CPUImplQPUSingleThread(size_t qubitSumNumber) 
    :qubit2stat(qubitSumNumber)
{
}

QGateParam& CPUImplQPUSingleThread::findgroup(size_t qn)
{
    for (auto iter = qubit2stat.begin(); iter != qubit2stat.end(); ++iter)
    {
        if (iter->enable == false) continue;
        if (find(iter->qVec.begin(), iter->qVec.end(), qn) != iter->qVec.end()) return *iter;
    }
    QCERR("unknow error");
    throw runtime_error("unknow error");
}

bool CPUImplQPUSingleThread::TensorProduct(QGateParam& qgroup0, QGateParam& qgroup1)
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

static bool probcompare(pair<size_t, double> a, pair<size_t, double> b)
{
    return a.second > b.second;
}

QError CPUImplQPUSingleThread::pMeasure
(Qnum& qnum, prob_tuple &mResult, int select_max)
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
    else if(mResult.size() <= select_max)
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
QError CPUImplQPUSingleThread::pMeasure(Qnum& qnum, prob_vec &mResult)
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

bool CPUImplQPUSingleThread::qubitMeasure(size_t qn)
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
	float fi = get_random_double();

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

QError CPUImplQPUSingleThread::initState(size_t qubit_num, const QStat &state)
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


QError CPUImplQPUSingleThread::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
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

QError  CPUImplQPUSingleThread::unitarySingleQubitGate
(size_t qn, QStat& matrix, bool isConjugate, GateType type)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    QGateParam& qgroup = findgroup(qn);
    size_t j;
    size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
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

    for (long long i = 0; i < (long long)qgroup.qstate.size(); i += ststep * 2)
    {
        for (j = i; j<i + ststep; j++)
        {
            alpha = qgroup.qstate[j];
            beta = qgroup.qstate[j + ststep];
            qgroup.qstate[j] = matrix[0] * alpha + matrix[1] * beta;         /* in j,the goal qubit is in |0>        */
            qgroup.qstate[j + ststep] = matrix[2] * alpha + matrix[3] * beta;         /* in j+ststep,the goal qubit is in |1> */
        }
    }
    return qErrorNone;
}

QError CPUImplQPUSingleThread::
controlunitarySingleQubitGate(size_t qn,
    Qnum& vControlBit,
    QStat & matrix,
    bool isConjugate,
    GateType type)
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

QError CPUImplQPUSingleThread::
unitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool isConjugate,
    GateType type)
{
    QGateParam& qgroup0 = findgroup(qn_0);
    QGateParam& qgroup1 = findgroup(qn_1);
    if (qgroup0.qVec[0] != qgroup1.qVec[0])
    {
        TensorProduct(qgroup0, qgroup1);
    }
    size_t ststep1 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_0)
        - qgroup0.qVec.begin());
    size_t ststep2 = 1ull << (find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn_1)
        - qgroup0.qVec.begin());
    size_t stemp1 = (ststep1>ststep2) ? ststep1 : ststep2;
    size_t stemp2 = (ststep1>ststep2) ? ststep2 : ststep1;

    bool bmark = true;
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
    long long j, k;

    for (long long i = 0; i<(long long)stateSize; i = i + 2 * stemp1)
    {
        for (j = i; j <(long long)(i + stemp1); j = j + 2 * stemp2)
        {
            for (k = j; k < (long long)(j + stemp2); k++)
            {
                phi00 = qgroup0.qstate[k];        //00
                phi01 = qgroup0.qstate[k + ststep2];  //01
                phi10 = qgroup0.qstate[k + ststep1];  //10
                phi11 = qgroup0.qstate[k + ststep1 + ststep2]; //11

                qgroup0.qstate[k] = matrix[0] * phi00 + matrix[1] * phi01
                    + matrix[2] * phi10 + matrix[3] * phi11;
                qgroup0.qstate[k + ststep2] = matrix[4] * phi00 + matrix[5] * phi01
                    + matrix[6] * phi10 + matrix[7] * phi11;
                qgroup0.qstate[k + ststep1] = matrix[8] * phi00 + matrix[9] * phi01
                    + matrix[10] * phi10 + matrix[11] * phi11;
                qgroup0.qstate[k + ststep1 + ststep2] = matrix[12] * phi00 + matrix[13] * phi01
                    + matrix[14] * phi10 + matrix[15] * phi11;
            }
        }
    }
    return qErrorNone;
}

QError CPUImplQPUSingleThread::
controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    QStat& matrix,
    bool isConjugate,
    GateType type)
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

QError CPUImplQPUSingleThread::Hadamard(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::Hadamard(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::X(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::X(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::Y(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::Y(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::P0(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::P0(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::P1(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::P1(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}


QError CPUImplQPUSingleThread::Z(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::Z(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::S(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::S(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::T(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::T(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::U1_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::RX_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::RX_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::RY_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::RY_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::RZ_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::RZ_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

//double quantum gate
QError CPUImplQPUSingleThread::CNOT(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::CNOT(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::CZ(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::CZ(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::CR(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::CR(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    double theta,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::iSWAP(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::iSWAP(
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
QError CPUImplQPUSingleThread::iSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::iSWAP(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}


//pi/4 SqiSWAP
QError CPUImplQPUSingleThread::SqiSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError CPUImplQPUSingleThread::SqiSWAP(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError CPUImplQPUSingleThread::DiagonalGate(Qnum & vQubit,QStat & matrix, bool isConjugate, double error_rate)
{

    QGateParam& qgroup0 = findgroup(vQubit[0]);
    for (auto iter = vQubit.begin()+1; iter != vQubit.end(); iter++)
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
    for (size_t i = 0; i < qgroup0.qstate.size(); i++)
    {
        index = 0;
        for (size_t j = 0; j < qgroup0.qVec.size(); j++)
        {
            for (size_t k = 0; k < vQubit.size(); k++)
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

QError CPUImplQPUSingleThread::controlDiagonalGate(Qnum & vQubit, QStat & matrix, Qnum & vControlBit, bool isConjugate, double error_rate)
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
    for (long long i = 0; i < qgroup0.qstate.size(); i++)
    {
        index = 0;    // corresponding matrix number of i
        block = 0;    // The number of control bits is 1.
        for (long long j = 0; j < qgroup0.qVec.size(); j++)
        {
            for (size_t k = 0; k < vQubit.size(); k++)
            {
                if (qgroup0.qVec[j] == vQubit[k])
                {
                    index += (i >> j) % 2 * (1 << k);
                }
            }
            for (size_t k = 0; k < vControlBit.size(); k++)
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

QStat CPUImplQPUSingleThread::getQState()
{
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

QError CPUImplQPUSingleThread::Reset(size_t qn)
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

size_t extract_bit(size_t rawnumber, vector<size_t> bits) {
	size_t extracted_number = 0;
	for (int i = 0; i < bits.size(); ++i) {
		int bit = bits[i];
		int digit = (rawnumber >> bit) % 2;
		extracted_number += (1ull << i)*digit;
	}
	return extracted_number;
}

size_t reconstruct_number(size_t extracted_number, vector<size_t> bits) {
	size_t raw_number = 0;
	for (int i = 0; i < bits.size(); ++i) {
		int digit = (extracted_number >> i) % 2;
		raw_number += (1ull << bits[i])*digit;
	}
	return raw_number;
}

QError CPUImplQPUSingleThreadWithOracle::controlOracularGate(
	vector<size_t> bits,
	vector<size_t> controlbits,
	bool is_dagger,
	string name) {

	vector<size_t> name_qubits;
	string oracle_name;
	QPanda::parse_oracle_name(name, oracle_name, name_qubits);

	QGateParam& qgroup0 = findgroup(bits[0]);
	for (auto iter = bits.begin() + 1; iter != bits.end(); iter++)
	{
		TensorProduct(qgroup0, findgroup(*iter));
	}
	for (auto iter = controlbits.begin(); iter != controlbits.end(); iter++)
	{
		TensorProduct(qgroup0, findgroup(*iter));
	}

	size_t controller_mask = 0;
	for (size_t i = 0; i < controlbits.size(); ++i) {
		controller_mask += (1ull << controlbits[i]);
	}
	size_t remain_mask = controller_mask;
	for (size_t i = 0; i < bits.size(); ++i) {
		remain_mask += (1ull << bits[i]);
	}
	remain_mask = ~remain_mask;

	if (oracle_name == "add") {
		assert(name_qubits.size() == 2);
		assert(name_qubits[0] == name_qubits[1]);
		assert(name_qubits[0] + name_qubits[1] == bits.size());

		size_t qubitnumber = qgroup0.qubitnumber;
		Qnum qVec = qgroup0.qVec;
		QGateParam newgroup(qubitnumber, qVec);
		newgroup.qstate[0] = 0;
		for (size_t i = 0; i < (1ull << qubitnumber); ++i) {
			if (i & controller_mask == controller_mask) {
				continue;
			}
			size_t remain_i = i & remain_mask;
			size_t x = 0;
			size_t y = 0;
			Qnum qvecx = { bits.begin(), bits.begin() + name_qubits[0] };
			Qnum qvecy = { bits.begin() + name_qubits[0], bits.end() };
			x = extract_bit(i, qvecx);
			y = extract_bit(i, qvecy);
			size_t x_plus_y = (x + y) % (1ull << name_qubits[1]);
			size_t new_i = 0;
			new_i += remain_i;
			new_i += reconstruct_number(x, qvecx);
			new_i += reconstruct_number(x_plus_y, qvecy);

			newgroup.qstate[new_i] += qgroup0.qstate[i];
		}
		qgroup0.qstate = newgroup.qstate;
	}
	else {
		throw runtime_error("Not Implemented.");
	}
}


