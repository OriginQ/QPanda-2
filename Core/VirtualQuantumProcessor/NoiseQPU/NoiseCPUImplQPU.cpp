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
#include "QPandaConfig.h"
#include "NoiseCPUImplQPU.h"
#include "Core/Utilities/Tools/Utils.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>
#include "QPandaNamespace.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
using namespace std;
USING_QPANDA

NoisyCPUImplQPU::NoisyCPUImplQPU()
{
}

NoisyCPUImplQPU::~NoisyCPUImplQPU()
{
    qubit2stat.clear();
}

NoisyCPUImplQPU::NoisyCPUImplQPU(rapidjson::Document & doc)
{
    m_doc.CopyFrom(doc["noisemodel"], m_doc.GetAllocator());
}

QGateParam& NoisyCPUImplQPU::findgroup(size_t qn)
{
    for (auto iter = qubit2stat.begin(); iter != qubit2stat.end(); ++iter)
    {
        if (iter->enable == false) continue;
        if (find(iter->qVec.begin(), iter->qVec.end(), qn) != iter->qVec.end()) return *iter;
    }
    QCERR("unknown error");
    throw runtime_error("unknown error");
}

bool NoisyCPUImplQPU::TensorProduct(QGateParam& qgroup0, QGateParam& qgroup1)
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

QError NoisyCPUImplQPU::pMeasure
(Qnum& qnum, prob_tuple &mResult, int select_max)
{
    return undefineError;
}

QError NoisyCPUImplQPU::pMeasure(Qnum& qnum, prob_vec &mResult)
{
    return undefineError;
}

bool NoisyCPUImplQPU::qubitMeasure(size_t qn)
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

    double fi = get_random_double();

    if (fi> dprob)
    {
        ioutcome = 1;
    }

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

QError NoisyCPUImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{
    qubit2stat.erase(qubit2stat.begin(),qubit2stat.end());
    qubit2stat.resize(qubit_num);
    for (auto i = 0; i<qubit_num; i++)
    {
        qubit2stat[i].qVec.push_back(i);
        qubit2stat[i].qstate.push_back(1);
        qubit2stat[i].qstate.push_back(0);
        qubit2stat[i].qubitnumber = 1;
    }

    for (auto iter = qubit2stat.begin(); iter != qubit2stat.end(); iter++)
    {
        for (auto iter1 = (*iter).qstate.begin(); iter1 != (*iter).qstate.end(); iter1++)
        {
            *iter1 = 0;
        }
        (*iter).qstate[0] = 1;
    }
    return qErrorNone;
}

QError NoisyCPUImplQPU::_get_probabilities(prob_vec& probabilities, size_t qn, NoiseOp & noise)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    probabilities.resize(noise.size());
    for (auto iter : probabilities)
    {
        iter = 0;
    }
    QGateParam& qgroup = findgroup(qn);
    QStat qstat;
    size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    for (size_t k=0;k<noise.size();k++)
    {
        if (k > 0)
        {
            probabilities[k] = probabilities[k - 1];
        }
        qstat.assign(qgroup.qstate.begin(), qgroup.qstate.end());
        for (size_t i = 0; i < (size_t)qstat.size(); i += ststep * 2)
        {
            for (size_t j = i; j < i + ststep; j++)
            {
                alpha = qstat[j];
                beta = qstat[j + ststep];
                qstat[j] = noise[k][0] * alpha + noise[k][1] * beta;
                qstat[j + ststep] = noise[k][2] * alpha + noise[k][3] * beta;
                probabilities[k] += (abs(qstat[j])*abs(qstat[j]) + abs(qstat[j + ststep])*abs(qstat[j + ststep]));
            }
        }
    }
    return qErrorNone;
}

QError NoisyCPUImplQPU::_get_probabilities(prob_vec& probabilities, size_t qn_0,size_t qn_1, NoiseOp & noise)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    probabilities.resize(noise.size());
    for (auto iter : probabilities)
    {
        iter = 0;
    }
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
    size_t stemp1 = (ststep1 > ststep2) ? ststep1 : ststep2;
    size_t stemp2 = (ststep1 > ststep2) ? ststep2 : ststep1;

    qcomplex_t phi00, phi01, phi10, phi11;
    auto stateSize = qgroup0.qstate.size();
    QStat qstat;

    for (size_t k = 0; k < noise.size(); k++)
    {
        if (k > 0)
        {
            probabilities[k] = probabilities[k - 1];
        }
        qstat.assign(qgroup0.qstate.begin(), qgroup0.qstate.end());
        size_t j, l;
        for (size_t i = 0; i < (size_t)stateSize; i = i + 2 * stemp1)
        {
            for (j = i; j < (size_t)(i + stemp1); j = j + 2 * stemp2)
            {
                for (l = j; l < (size_t)(j + stemp2); l++)
                {
                    phi00 = qstat[l];        //00
                    phi01 = qstat[l + ststep2];  //01
                    phi10 = qstat[l + ststep1];  //10
                    phi11 = qstat[l + ststep1 + ststep2]; //11
                    qstat[l] = noise[k][0] * phi00 + noise[k][1] * phi01
                        + noise[k][2] * phi10 + noise[k][3] * phi11;
                    qstat[l + ststep2] = noise[k][4] * phi00 + noise[k][5] * phi01
                        + noise[k][6] * phi10 + noise[k][7] * phi11;
                    qstat[l + ststep1] = noise[k][8] * phi00 + noise[k][9] * phi01
                        + noise[k][10] * phi10 + noise[k][11] * phi11;
                    qstat[l + ststep1 + ststep2] = noise[k][12] * phi00 + noise[k][13] * phi01
                        + noise[k][14] * phi10 + noise[k][15] * phi11;

                    probabilities[k] += (abs(qstat[l])*abs(qstat[l])
                        + abs(qstat[l + ststep1])*abs(qstat[l + ststep1])
                        + abs(qstat[l + ststep2])*abs(qstat[l + ststep2])
                        + abs(qstat[l + ststep1 + ststep2])*abs(qstat[l + ststep1 + ststep2]));
                }
            }
        }
    }
    return qErrorNone;
}

size_t choose_operator(prob_vec & probabilities, qstate_type drand)
{
    size_t number = 0;
    for (size_t i = 0; i < probabilities.size()-1; i++)
    {
        if (probabilities[i] < drand && probabilities[i + 1]>drand)
        {
            number = i+1;
        }
    }
    return number;
}

QStat matrix_multiply(const QStat &matrix_left, const QStat &matrix_right)
{
    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);
    int dimension = (int)sqrt(size);

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            qcomplex_t temp = 0;
            for (int k = 0; k < dimension; k++)
            {
                temp += matrix_left[i*dimension + k] * matrix_right[k*dimension + j];
            }
            matrix_result[i*dimension + j] = temp;
        }
    }
    return matrix_result;
}

QError  NoisyCPUImplQPU::singleQubitGateNoise
(size_t qn, NoiseOp & noise)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    QGateParam& qgroup = findgroup(qn);
    size_t j;
    size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    prob_vec probabilities;
    _get_probabilities(probabilities, qn, noise);
    double dtemp = get_random_double();
    size_t op_number = choose_operator(probabilities, dtemp);
    double dsum = 0;
    for (size_t i = 0; i < (size_t)qgroup.qstate.size(); i += ststep * 2)
    {
        for (j = i; j < i + ststep; j++)
        {
            alpha = qgroup.qstate[j];
            beta = qgroup.qstate[j + ststep];
            qgroup.qstate[j] = noise[op_number][0] * alpha + noise[op_number][1] * beta;         /* in j,the goal qubit is in |0>        */
            qgroup.qstate[j + ststep] = noise[op_number][2] * alpha + noise[op_number][3] * beta;         /* in j+ststep,the goal qubit is in |1> */
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

QError NoisyCPUImplQPU::doubleQubitGateNoise
(size_t qn_0,size_t qn_1, NoiseOp & noise)
{
    prob_vec probabilities;
    _get_probabilities(probabilities, qn_0, qn_1, noise);
    double dtemp = get_random_double();
    size_t op_number = choose_operator(probabilities, dtemp);
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
    size_t stemp1 = (ststep1 > ststep2) ? ststep1 : ststep2;
    size_t stemp2 = (ststep1 > ststep2) ? ststep2 : ststep1;

    qcomplex_t phi00, phi01, phi10, phi11;
    auto stateSize = qgroup0.qstate.size();
    double dsum = 0;
    size_t j, k;
    for (size_t i = 0; i < (size_t)stateSize; i = i + 2 * stemp1)
    {
        for (j = i; j < (size_t)(i + stemp1); j = j + 2 * stemp2)
        {
            for (k = j; k < (size_t)(j + stemp2); k++)
            {
                phi00 = qgroup0.qstate[k];        //00
                phi01 = qgroup0.qstate[k + ststep2];  //01
                phi10 = qgroup0.qstate[k + ststep1];  //10
                phi11 = qgroup0.qstate[k + ststep1 + ststep2]; //11
                qgroup0.qstate[k] = noise[op_number][0] * phi00 + noise[op_number][1] * phi01
                    + noise[op_number][2] * phi10 + noise[op_number][3] * phi11;
                qgroup0.qstate[k + ststep2] = noise[op_number][4] * phi00 + noise[op_number][5] * phi01
                    + noise[op_number][6] * phi10 + noise[op_number][7] * phi11;
                qgroup0.qstate[k + ststep1] = noise[op_number][8] * phi00 + noise[op_number][9] * phi01
                    + noise[op_number][10] * phi10 + noise[op_number][11] * phi11;
                qgroup0.qstate[k + ststep1 + ststep2] = noise[op_number][12] * phi00 + noise[op_number][13] * phi01
                    + noise[op_number][14] * phi10 + noise[op_number][15] * phi11;
                dsum += (abs(qgroup0.qstate[k])*abs(qgroup0.qstate[k])
                    + abs(qgroup0.qstate[j + ststep1])*abs(qgroup0.qstate[j + ststep1])
                    + abs(qgroup0.qstate[j + ststep2])*abs(qgroup0.qstate[j + ststep2])
                    + abs(qgroup0.qstate[j + ststep1 + ststep2])*abs(qgroup0.qstate[j + ststep1 + ststep2]));
            }
        }
    }

    dsum = sqrt(dsum);
    for (size_t i = 0; i < qgroup0.qstate.size(); i++)
    {
        qgroup0.qstate[i] /= dsum;
    }
    return qErrorNone;
}

QError NoisyCPUImplQPU::noisyUnitarySingleQubitGate
(size_t qn, QStat& matrix, bool isConjugate, NoiseOp & noise)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    QGateParam& qgroup = findgroup(qn);
    size_t j;
    size_t ststep = 1ull << find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    prob_vec probabilities;
    _get_probabilities(probabilities, qn,noise);
    double dtemp = get_random_double();
    size_t op_number = choose_operator(probabilities, dtemp);

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
    const QStat m1(noise[op_number]);
    const QStat m2(matrix);
    QStat matrix_new = matrix_multiply(matrix, noise[op_number]);
    double dsum = 0;
    for (size_t i = 0; i < (size_t)qgroup.qstate.size(); i += ststep * 2)
    {
        for (j = i; j<i + ststep; j++)
        {
            alpha = qgroup.qstate[j];
            beta = qgroup.qstate[j + ststep];
            qgroup.qstate[j] = matrix_new[0] * alpha + matrix_new[1] * beta;         /* in j,the goal qubit is in |0>        */
            qgroup.qstate[j + ststep] = matrix_new[2] * alpha + matrix_new[3] * beta;         /* in j+ststep,the goal qubit is in |1> */
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

QError NoisyCPUImplQPU::unitarySingleQubitGate
(size_t qn, QStat& matrix, bool isConjugate, GateType type)
{   
    auto gate_name = TransformQGateType::getInstance()[type];
    if (gate_name.size() == 0)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    if (m_doc.HasMember(gate_name.c_str()))
    {
        auto &value = m_doc[gate_name.c_str()];
        NoiseOp noise;
        auto status = SingleGateNoiseModeMap::getInstance()[(NOISE_MODEL)value[0].
            GetInt()](value, noise);
        if (!status)
        {
            QCERR("noise model function fail");
            throw invalid_argument("noise model function fail");
        }
        return noisyUnitarySingleQubitGate(qn, matrix, isConjugate, noise);
    }
    else
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

        for (size_t i = 0; i < (size_t)qgroup.qstate.size(); i += ststep * 2)
        {
            for (j = i; j < i + ststep; j++)
            {
                alpha = qgroup.qstate[j];
                beta = qgroup.qstate[j + ststep];
                qgroup.qstate[j] = matrix[0] * alpha + matrix[1] * beta;         /* in j,the goal qubit is in |0>        */
                qgroup.qstate[j + ststep] = matrix[2] * alpha + matrix[3] * beta;         /* in j+ststep,the goal qubit is in |1> */
            }
        }
    }
    
    return qErrorNone;
}

QError NoisyCPUImplQPU::
controlunitarySingleQubitGate(size_t qn,
    Qnum& vControlBit,
    QStat & matrix,
    bool isConjugate,
    GateType type)
{
    return qErrorNone;
}

QError NoisyCPUImplQPU::
unitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool isConjugate,
    GateType type)
{
    auto gate_name = TransformQGateType::getInstance()[type];
    if (gate_name.size() == 0)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    if (m_doc.HasMember(gate_name.c_str()))
    {
        auto &value = m_doc[gate_name.c_str()];
        NoiseOp noise;
        auto status = DoubleGateNoiseModeMap::getInstance()[(NOISE_MODEL)value[0].
            GetInt()](value, noise);
        if (!status)
        {
            QCERR("noise model function fail");
            throw invalid_argument("noise model function fail");
        }
        return noisyUnitaryDoubleQubitGate(qn_0,qn_1, matrix, isConjugate, noise);
    }
    else
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
        size_t j, k;
        for (size_t i = 0; i<(size_t)stateSize; i = i + 2 * stemp1)
        {
            for (j = i; j <(size_t)(i + stemp1); j = j + 2 * stemp2)
            {
                for (k = j; k < (size_t)(j + stemp2); k++)
                {
                    phi00 = qgroup0.qstate[k];            //00
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
    }
    return qErrorNone;
}

QError NoisyCPUImplQPU::
noisyUnitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool isConjugate,
    NoiseOp & noise)
{
  
    prob_vec probabilities;
    _get_probabilities(probabilities, qn_0,qn_1,noise);
    double dtemp = get_random_double();
    size_t op_number = choose_operator(probabilities, dtemp);
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
    size_t stemp1 = (ststep1 > ststep2) ? ststep1 : ststep2;
    size_t stemp2 = (ststep1 > ststep2) ? ststep2 : ststep1;

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
    double dsum = 0;
    QStat matrix_new = matrix_multiply(matrix,noise[op_number] );
    size_t j, k;
    for (size_t i = 0; i < (size_t)stateSize; i = i + 2 * stemp1)
    {
        for (j = i; j < (size_t)(i + stemp1); j = j + 2 * stemp2)
        {
            for (k = j; k < (size_t)(j + stemp2); k++)
            {
                phi00 = qgroup0.qstate[k];        //00
                phi01 = qgroup0.qstate[k + ststep2];  //01
                phi10 = qgroup0.qstate[k + ststep1];  //10
                phi11 = qgroup0.qstate[k + ststep1 + ststep2]; //11
                qgroup0.qstate[k] = matrix_new[0] * phi00 + matrix_new[1] * phi01
                    + matrix_new[2] * phi10 + matrix_new[3] * phi11;
                qgroup0.qstate[k + ststep2] = matrix_new[4] * phi00 + matrix_new[5] * phi01
                    + matrix_new[6] * phi10 + matrix_new[7] * phi11;
                qgroup0.qstate[k + ststep1] = matrix_new[8] * phi00 + matrix_new[9] * phi01
                    + matrix_new[10] * phi10 + matrix_new[11] * phi11;
                qgroup0.qstate[k + ststep1 + ststep2] = matrix_new[12] * phi00 + matrix_new[13] * phi01
                    + matrix_new[14] * phi10 + matrix_new[15] * phi11;
                dsum += (abs(qgroup0.qstate[k])*abs(qgroup0.qstate[k]) 
                    + abs(qgroup0.qstate[k + ststep1])*abs(qgroup0.qstate[k + ststep1])
                    + abs(qgroup0.qstate[k + ststep2])*abs(qgroup0.qstate[k + ststep2])
                    + abs(qgroup0.qstate[k + ststep1 + ststep2])*abs(qgroup0.qstate[k + ststep1 + ststep2]));
            }
        }
    }

    dsum = sqrt(dsum);
    for (size_t i = 0; i < qgroup0.qstate.size(); i++)
    {
        qgroup0.qstate[i] /= dsum;
    }
    return qErrorNone;
}

QError NoisyCPUImplQPU::
controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    QStat& matrix,
    bool isConjugate,
    GateType type)
{
    return qErrorNone;
}

QError NoisyCPUImplQPU::Hadamard(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::Hadamard(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::X(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::X(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::Y(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::Y(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::Z(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::Z(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::S(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::S(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::T(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::T(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::U1_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::RX_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::RX_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::RY_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::RY_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::RZ_GATE(size_t qn, double theta,
    bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::RZ_GATE(
    size_t qn,
    double theta,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

//double quantum gate
QError NoisyCPUImplQPU::CNOT(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::CNOT(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::CZ(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::CZ(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::CR(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::CR(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    double theta,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::iSWAP(size_t qn_0, size_t qn_1, double theta, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::iSWAP(
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
QError NoisyCPUImplQPU::iSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::iSWAP(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

//pi/4 SqiSWAP
QError NoisyCPUImplQPU::SqiSWAP(size_t qn_0, size_t qn_1, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::SqiSWAP(
    size_t qn_0,
    size_t qn_1,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::DiagonalGate(Qnum& vQubit,
    QStat & matrix,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::controlDiagonalGate(Qnum& vQubit,
    QStat & matrix,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QStat NoisyCPUImplQPU::getQState()
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

QError NoisyCPUImplQPU::Reset(size_t qn)
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

QError NoisyCPUImplQPU::P0(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::P0(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}

QError NoisyCPUImplQPU::P1(size_t qn, bool isConjugate, double error_rate)
{
    return undefineError;
}
QError NoisyCPUImplQPU::P1(
    size_t qn,
    Qnum& vControlBit,
    bool isConjugate,
    double error_rate)
{
    return undefineError;
}