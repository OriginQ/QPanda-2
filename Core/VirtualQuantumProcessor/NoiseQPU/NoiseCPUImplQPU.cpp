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
#include "NoiseCPUImplQPU.h"
#include "Core/Utilities/Tools/Utils.h"
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>
#include "QPandaNamespace.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"


using namespace std;
USING_QPANDA

#ifdef USE_OPENMP
#include <omp.h>
#endif

const QStat kMatrix_I = {1, 0, 0, 1};

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

NoisyCPUImplQPU::NoisyCPUImplQPU(NoisyQuantum &quantum_noise)
    :m_quantum_noise(quantum_noise)
{

}

void NoisyCPUImplQPU::set_quantum_noise(const NoisyQuantum &quantum_noise)
{
    m_quantum_noise = quantum_noise;
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
    unitary_noise_qubit_gate({qn}, QStat(), false, GATE_TYPE_MEASURE);
    bool measure = measure_standard(qn);
    return readout(measure, qn);
}

QError NoisyCPUImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
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

QError NoisyCPUImplQPU::initState(size_t qubit_num, const QStat &state)
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

QError NoisyCPUImplQPU::unitary_noise_qubit_gate(const Qnum &qns, const QStat &matrix,
                                                 bool is_conjugate, GateType type)
{
    NOISE_MODEL noise_model;
    NoiseOp ops;
    Qnum effect_qubits;
    auto is_noise = m_quantum_noise.sample_noisy_op(type, qns, noise_model, ops, effect_qubits, m_rng);

    if (!is_noise)
    {
        return QError::qErrorNone;
    }

    switch (noise_model)
    {
    case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR: // X
    case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR: // Y
    case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR: // Z
    case NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR:
    case NOISE_MODEL::PHASE_DAMPING_OPRATOR:
        QPANDA_ASSERT(ops.size() != effect_qubits.size(), "Error: noise kruas");
        for (size_t i = 0; i < ops.size(); i++)
        {
            unitary_qubit_gate_standard(effect_qubits[i], ops[i], false);
        }
        break;
    case NOISE_MODEL::DAMPING_KRAUS_OPERATOR:
    case NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR:
        {
            QStat standard_matrix;
            unitary_noise_qubit_kraus(effect_qubits, ops, standard_matrix);
            if (1 == qns.size())
            {
                unitary_qubit_gate_standard(effect_qubits[0], standard_matrix, false);
            }
            else
            {
                unitary_qubit_gate_standard(effect_qubits[0], effect_qubits[1], standard_matrix, false);
            }
        }
        break;
    case NOISE_MODEL::MIXED_UNITARY_OPRATOR:
    {
        if (1 == qns.size())
        {
            unitary_qubit_gate_standard(effect_qubits[0], ops[0], false);
        }
        else
        {
            unitary_qubit_gate_standard(effect_qubits[0], effect_qubits[1], ops[0], false);
        }
    }
        break;
    default:
        throw std::runtime_error("Error: noise model");
    }

    return QError::qErrorNone;
}

QError NoisyCPUImplQPU::unitary_noise_qubit_kraus(const Qnum & qns, const NoiseOp & ops, QStat & standard_matrix)
{
    auto random_p = m_rng.random_double();
    double sum_p = 0;
    double p = 0;
    bool is_get_noise_ops = false;

    for (size_t i = 0; i < ops.size() - 1; i++)
    {
        p = unitary_kraus(qns, ops[i]);
        sum_p += p;
        if (sum_p > random_p)
        {
            is_get_noise_ops = true;
            standard_matrix = ops[i];
            QPANDA_ASSERT(abs(p) < FLT_EPSILON, "Error: normlize prob");
            normlize(standard_matrix, 1 / sqrt(p));
            break;
        }
    }

    if (!is_get_noise_ops)
    {
        auto res_p = 1 - sum_p;
        QPANDA_ASSERT(abs(res_p) < FLT_EPSILON, "Error: normlize prob");
        standard_matrix = ops.back();
        normlize(standard_matrix, 1 / sqrt(res_p));
    }

    return QError::qErrorNone;
}


QError NoisyCPUImplQPU::unitary_qubit_gate_standard(size_t qn, QStat &matrix,
                                                    bool is_conjugate)
{
    QPANDA_RETURN(!mat_compare(matrix, kMatrix_I), QError::qErrorNone);
    qcomplex_t alpha;
    qcomplex_t beta;
    QGateParam& qgroup = findgroup(qn);

    size_t n = find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    size_t ststep = 1ull << n;

    if (is_conjugate)
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

    return QError::qErrorNone;
}

QError NoisyCPUImplQPU::unitary_qubit_gate_standard(size_t qn0, size_t qn1,
                                                    QStat &matrix, bool is_conjugate)
{
    QGateParam& qgroup0 = findgroup(qn0);
    QGateParam& qgroup1 = findgroup(qn1);
    if (qgroup0.qVec[0] != qgroup1.qVec[0])
    {
        TensorProduct(qgroup0, qgroup1);
    }

    size_t n1 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn0) - qgroup0.qVec.begin();
    size_t n2 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qn1) - qgroup0.qVec.begin();
    size_t ststep1 = 1ull << n1;
    size_t ststep2 = 1ull << n2;

    if (n1 < n2)
    {
        std::swap(n1, n2);
    }

    qcomplex_t phi00, phi01, phi10, phi11;
    auto stateSize = qgroup0.qstate.size();
    if (is_conjugate)
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

    return QError::qErrorNone;
}

QError NoisyCPUImplQPU::reset_standard(size_t qn)
{
    QGateParam& qgroup = findgroup(qn);
    size_t j;
    size_t ststep = 1ull << (find(qgroup.qVec.begin(), qgroup.qVec.end(), qn)
        - qgroup.qVec.begin());
    //#pragma omp parallel for private(j,alpha,beta)
    double dsum = 0;
    for (size_t i = 0; i < qgroup.qstate.size(); i += ststep * 2)
    {
        for (j = i; j < i + ststep; j++)
        {
            qgroup.qstate[j + ststep] = 0;                              /* in j+ststep,the goal qubit is in |1> */
            dsum += (abs(qgroup.qstate[j])*abs(qgroup.qstate[j]) + abs(qgroup.qstate[j + ststep])*abs(qgroup.qstate[j + ststep]));
        }
    }

    dsum = sqrt(dsum);
    QPANDA_RETURN(std::abs(dsum) < FLT_EPSILON, QError::qErrorNone);

    for (size_t i = 0; i < qgroup.qstate.size(); i++)
    {
        qgroup.qstate[i] /= dsum;
    }
    return QError::qErrorNone;
}

bool NoisyCPUImplQPU::measure_standard(size_t qn)
{
    QGateParam& qgroup = findgroup(qn);
    size_t n = find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    size_t ststep = 1ull << n;
    double dprob(0);

#pragma omp parallel for reduction(+:dprob)
    for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
    {
        int64_t real00_idx = insert(i, n);
        dprob += std::norm(qgroup.qstate[real00_idx]);
    }
    int ioutcome(0);

    double fi = get_random_double();

    if (fi > dprob)
    {
        ioutcome = 1;
    }

    if (ioutcome == 0)
    {
        dprob = 1 / sqrt(dprob);
#pragma omp parallel for
        for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
        {
            int64_t real00_idx = insert(i, n);
            qgroup.qstate[real00_idx] *= dprob;
            qgroup.qstate[real00_idx + ststep] = 0;
        }
    }
    else
    {
        dprob = 1 / sqrt(1 - dprob);
#pragma omp parallel for
        for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
        {
            int64_t real00_idx = insert(i, n);
            qgroup.qstate[real00_idx] = 0;
            qgroup.qstate[real00_idx + ststep] *= dprob;
        }
    }
    return ioutcome;
}

bool NoisyCPUImplQPU::readout(bool measure, size_t qn)
{
    vector<vector<double>> readout_error;
    auto is_noise = m_quantum_noise.sample_noisy_op(qn, readout_error, m_rng);

    if (!is_noise)
    {
        return measure;
    }

    if (measure)
    {
        measure = m_rng.random_discrete(readout_error[1]);
    }
    else
    {
        measure = m_rng.random_discrete(readout_error[0]);
    }

    return measure;
}

void NoisyCPUImplQPU::normlize(QStat & matrix, double p)
{
    for (auto &val : matrix)
    {
        val *= p;
    }

    return;
}

double NoisyCPUImplQPU::unitary_kraus(const Qnum & qns, const QStat & op)
{
    double p = 0;
    qcomplex_t alpha, beta, gamma, phi;

    if (1 == qns.size())
    {
        QGateParam& qgroup = findgroup(qns[0]);
        //QStat qstat;
        size_t n = find(qgroup.qVec.begin(), qgroup.qVec.end(), qns[0]) - qgroup.qVec.begin();
        size_t ststep = 1ull << n;

#pragma omp parallel for private(alpha, beta) reduction(+:p)
        for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
        {
            int64_t real00_idx = insert(i, n);
            int64_t real01_idx = real00_idx + ststep;

            alpha = op[0] * qgroup.qstate[real00_idx] + op[1] * qgroup.qstate[real01_idx];
            beta = op[2] * qgroup.qstate[real00_idx] + op[3] * qgroup.qstate[real01_idx];
            p += std::norm(alpha) + std::norm(beta);
        }
    }
    else if (2 == qns.size())
    {
        QGateParam& qgroup0 = findgroup(qns[0]);
        QGateParam& qgroup1 = findgroup(qns[1]);

        if (qgroup0.qVec[0] != qgroup1.qVec[0])
        {
            TensorProduct(qgroup0, qgroup1);
        }

        size_t n1 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qns[0]) - qgroup0.qVec.begin();
        size_t n2 = find(qgroup0.qVec.begin(), qgroup0.qVec.end(), qns[1]) - qgroup0.qVec.begin();
        size_t ststep1 = 1ull << n1;
        size_t ststep2 = 1ull << n2;

        if (n1 < n2)
        {
            std::swap(n1, n2);
        }
        auto stateSize = qgroup0.qstate.size();

#pragma omp parallel for private(alpha, beta, gamma, phi) reduction(+:p)
        for (int64_t i = 0; i < (stateSize >> 2); i++)
        {
            int64_t real00_idx = insert(i, n2, n1);

            alpha = op[0] * qgroup0.qstate[real00_idx] + op[1] * qgroup0.qstate[real00_idx + ststep2]
                + op[2] * qgroup0.qstate[real00_idx + ststep1] + op[3] * qgroup0.qstate[real00_idx + ststep2 + ststep1];
            beta = op[4] * qgroup0.qstate[real00_idx] + op[5] * qgroup0.qstate[real00_idx + ststep2]
                + op[6] * qgroup0.qstate[real00_idx + ststep1] + op[7] * qgroup0.qstate[real00_idx + ststep2 + ststep1];

            gamma = op[8] * qgroup0.qstate[real00_idx] + op[9] * qgroup0.qstate[real00_idx + ststep2]
                + op[10] * qgroup0.qstate[real00_idx + ststep1] + op[11] * qgroup0.qstate[real00_idx + ststep2 + ststep1];
            phi = op[12] * qgroup0.qstate[real00_idx] + op[13] * qgroup0.qstate[real00_idx + ststep2]
                + op[14] * qgroup0.qstate[real00_idx + ststep1] + op[15] * qgroup0.qstate[real00_idx + ststep2 + ststep1];

            p += std::norm(alpha) + std::norm(beta) + std::norm(gamma) + std::norm(phi);
        }
    }
    else
    {
        throw std::runtime_error("Error: noise ops qubit");
    }

    return p;
}

QError NoisyCPUImplQPU::_get_probabilities(prob_vec& probabilities, size_t qn, NoiseOp & noise)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    probabilities.assign(noise.size(), 0);
    QGateParam& qgroup = findgroup(qn);
    //QStat qstat;
    size_t n = find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    size_t ststep = 1ull << n;

    for (int64_t k = 0; k < noise.size(); k++)
    {
        if (k > 0)
        {
            probabilities[k] = probabilities[k - 1];
        }
        //qstat.assign(qgroup.qstate.begin(), qgroup.qstate.end());

        double p = 0;
#pragma omp parallel for private(alpha, beta) reduction(+:p)
        for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
        {
            int64_t real00_idx = insert(i, n);
            int64_t real01_idx = real00_idx + ststep;

            alpha = noise[k][0] * qgroup.qstate[real00_idx] + noise[k][1] * qgroup.qstate[real01_idx];
            beta = noise[k][2] * qgroup.qstate[real00_idx] + noise[k][3] * qgroup.qstate[real01_idx];
            p += std::norm(alpha) + std::norm(beta);
        }

        probabilities[k] += p;
    }
    return qErrorNone;
}

QError NoisyCPUImplQPU::_get_probabilities(prob_vec& probabilities, size_t qn_0, size_t qn_1, NoiseOp & noise)
{
    qcomplex_t alpha;
    qcomplex_t beta;
    probabilities.assign(noise.size(), 0);

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
    //QStat qstat;

    for (size_t k = 0; k < noise.size(); k++)
    {
        if (k > 0)
        {
            probabilities[k] = probabilities[k - 1];
        }
        //qstat.assign(qgroup0.qstate.begin(), qgroup0.qstate.end());

        int64_t j, l;
        double p = 0;
#pragma omp parallel for private(phi00, phi01, phi10, phi11) reduction(+:p)
        for (int64_t i = 0; i < (stateSize >> 2); i++)
        {
            int64_t real00_idx = insert(i, n2, n1);

            phi00 = noise[k][0] * qgroup0.qstate[real00_idx] + noise[k][1] * qgroup0.qstate[real00_idx + ststep2]
                + noise[k][2] * qgroup0.qstate[real00_idx + ststep1] + noise[k][3] * qgroup0.qstate[real00_idx + ststep2 + ststep1];
            phi01 = noise[k][4] * qgroup0.qstate[real00_idx] + noise[k][5] * qgroup0.qstate[real00_idx + ststep2]
                + noise[k][6] * qgroup0.qstate[real00_idx + ststep1] + noise[k][7] * qgroup0.qstate[real00_idx + ststep2 + ststep1];

            phi10 = noise[k][8] * qgroup0.qstate[real00_idx] + noise[k][9] * qgroup0.qstate[real00_idx + ststep2]
                + noise[k][10] * qgroup0.qstate[real00_idx + ststep1] + noise[k][11] * qgroup0.qstate[real00_idx + ststep2 + ststep1];
            phi11 = noise[k][12] * qgroup0.qstate[real00_idx] + noise[k][13] * qgroup0.qstate[real00_idx + ststep2]
                + noise[k][14] * qgroup0.qstate[real00_idx + ststep1] + noise[k][15] * qgroup0.qstate[real00_idx + ststep2 + ststep1];

            p += std::norm(phi00) + std::norm(phi01) + std::norm(phi10) + std::norm(phi11);
        }

        probabilities[k] += p;
    }
    return qErrorNone;
}

size_t choose_operator(prob_vec & probabilities, qstate_type drand)
{
    size_t number = 0;
    for (size_t i = 0; i < probabilities.size() - 1; i++)
    {
        if (probabilities[i] < drand && probabilities[i + 1]>drand)
        {
            number = i + 1;
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
(size_t qn_0, size_t qn_1, NoiseOp & noise)
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

    size_t n = find(qgroup.qVec.begin(), qgroup.qVec.end(), qn) - qgroup.qVec.begin();
    size_t ststep = 1ull << n;

    prob_vec probabilities;
    _get_probabilities(probabilities, qn, noise);
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

    #pragma omp parallel for private(alpha, beta) reduction(+:dsum)
        for (int64_t i = 0; i < (qgroup.qstate.size() >> 1); i++)
        {
            int64_t real00_idx = insert(i, n);
            int64_t real01_idx = real00_idx + ststep;

            alpha = qgroup.qstate[real00_idx];
            beta = qgroup.qstate[real01_idx];
            qgroup.qstate[real00_idx] = matrix_new[0] * alpha + matrix_new[1] * beta;
            qgroup.qstate[real01_idx] = matrix_new[2] * alpha + matrix_new[3] * beta;
            dsum += std::norm(qgroup.qstate[real00_idx]) + std::norm(qgroup.qstate[real01_idx]);
        }

        dsum = sqrt(dsum);

#pragma omp parallel for
    for (int64_t i = 0; i < qgroup.qstate.size(); i++)
    {
        qgroup.qstate[i] /= dsum;
    }

    return qErrorNone;
}

QError NoisyCPUImplQPU::unitarySingleQubitGate
(size_t qn, QStat& matrix, bool isConjugate, GateType type)
{
    unitary_qubit_gate_standard(qn, matrix, isConjugate);
    return unitary_noise_qubit_gate({qn}, matrix, isConjugate, type);
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
    unitary_qubit_gate_standard(qn_0, qn_1, matrix, isConjugate);
    return unitary_noise_qubit_gate({qn_0, qn_1}, matrix, isConjugate, type);
}

QError NoisyCPUImplQPU::
noisyUnitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool isConjugate,
    NoiseOp & noise)
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

    double dsum = 0;
    QStat matrix_new = matrix_multiply(matrix, noise[op_number]);
    int64_t j, k;

#pragma omp parallel for private(phi00, phi01, phi10, phi11) reduction(+:dsum)
    for (int64_t i = 0; i < (stateSize >> 2); i++)
    {
        int64_t real00_idx = insert(i, n2, n1);
        phi00 = qgroup0.qstate[real00_idx];
        phi01 = qgroup0.qstate[real00_idx + ststep2];
        phi10 = qgroup0.qstate[real00_idx + ststep1];
        phi11 = qgroup0.qstate[real00_idx + ststep1 + ststep2];

        qgroup0.qstate[real00_idx] = matrix_new[0] * phi00 + matrix_new[1] * phi01
            + matrix_new[2] * phi10 + matrix_new[3] * phi11;
        qgroup0.qstate[real00_idx + ststep2] = matrix_new[4] * phi00 + matrix_new[5] * phi01
            + matrix_new[6] * phi10 + matrix_new[7] * phi11;
        qgroup0.qstate[real00_idx + ststep1] = matrix_new[8] * phi00 + matrix_new[9] * phi01
            + matrix_new[10] * phi10 + matrix_new[11] * phi11;
        qgroup0.qstate[real00_idx + ststep1 + ststep2] = matrix_new[12] * phi00 + matrix_new[13] * phi01
            + matrix_new[14] * phi10 + matrix_new[15] * phi11;

        dsum += std::norm(qgroup0.qstate[real00_idx]) + std::norm(qgroup0.qstate[real00_idx + ststep1])
            + std::norm(qgroup0.qstate[real00_idx + ststep2]) + std::norm(qgroup0.qstate[real00_idx + ststep1 + ststep2]);
    }

    dsum = sqrt(dsum);
#pragma omp parallel for
    for (int64_t i = 0; i < qgroup0.qstate.size(); i++)
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
    NoiseOp ops;
    Qnum effect_qubits;
    auto is_noise = m_quantum_noise.sample_noisy_op(GATE_TYPE_RESET, {qn}, ops, effect_qubits, m_rng);

    if (!is_noise)
    {
        return reset_standard(qn);
    }

    if (2 == ops.size())
    {
        reset_standard(qn);
        unitary_qubit_gate_standard(effect_qubits[0], ops[1], false);
    }
    else
    {
        QStat I = {1, 0, 0, 1};
        if (!equal(ops[0], I))
        {
            reset_standard(effect_qubits[0]);
        }
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
