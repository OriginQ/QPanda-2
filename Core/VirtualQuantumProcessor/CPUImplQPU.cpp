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




CPUImplQPU::CPUImplQPU()
{
}

CPUImplQPU::~CPUImplQPU()
{
}
CPUImplQPU::CPUImplQPU(size_t qubit_num)
{
}

static bool probcompare(pair<size_t, double> a, pair<size_t, double> b)
{
    return a.second > b.second;
}

QError CPUImplQPU::pMeasure(Qnum& qnum, prob_tuple &probs, int select_max)
{
    pMeasure(qnum, probs);
    if (select_max == -1 || probs.size() <= select_max)
    {
        stable_sort(probs.begin(), probs.end(), probcompare);
        return qErrorNone;
    }
    else
    {
        stable_sort(probs.begin(), probs.end(), probcompare);
        probs.erase(probs.begin() + select_max, probs.end());
    }
    return qErrorNone;
}


QError CPUImplQPU::pMeasure(Qnum& qnum, prob_vec &probs)
{
    probs.resize(1ll << qnum.size());
    size_t size = 1ll << m_qubit_num;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t idx = 0;
            for (int64_t j = 0; j < qnum.size(); j++)
            {
                idx += (((i >> (qnum[j])) % 2) << j);
            }
            probs[idx] += std::norm(m_state[i]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t idx = 0;
            for (int64_t j = 0; j < qnum.size(); j++)
            {
                idx += (((i >> (qnum[j])) % 2) << j);
            }
            probs[idx] += std::norm(m_state[i]);
        }
    }

    return qErrorNone;
}

bool CPUImplQPU::qubitMeasure(size_t qn)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    double dprob = 0;

    if (size > m_threshold)
    {
#pragma omp parallel for reduction (+:dprob)
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            dprob += std::norm(m_state[real00_idx]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            dprob += std::norm(m_state[real00_idx]);
        }
    }


    bool measure_out = false;
    double fi = random_generator19937();
    if (fi > dprob)
    {
        measure_out = true;
    }

    if (!measure_out)
    {
        dprob = 1 / sqrt(dprob);
        if (size > m_threshold)
        {
#pragma omp parallel for
            for (int64_t i = 0; i < size; i++)
            {
                int64_t real00_idx = _insert(i, qn);
                m_state[real00_idx] *= dprob;
                m_state[real00_idx | offset] = 0;
            }
        }
        else
        {
            for (int64_t i = 0; i < size; i++)
            {
                int64_t real00_idx = _insert(i, qn);
                m_state[real00_idx] *= dprob;
                m_state[real00_idx | offset] = 0;
            }
        }
    }
    else
    {
        dprob = 1 / sqrt(1 - dprob);
        if (size > m_threshold)
        {
#pragma omp parallel for
            for (int64_t i = 0; i < size; i++)
            {
                int64_t real00_idx = _insert(i, qn);
                m_state[real00_idx] = 0;
                m_state[real00_idx | offset] *= dprob;
            }
        }
        else
        {
            for (int64_t i = 0; i < size; i++)
            {
                int64_t real00_idx = _insert(i, qn);
                m_state[real00_idx] = 0;
                m_state[real00_idx | offset] *= dprob;
            }
        }
    }
    return measure_out;
}

QError CPUImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{
    return initState(qubit_num);
}

QError CPUImplQPU::initState(size_t qubit_num, const QStat &state)
{
    if (0 == state.size())
    {
        m_qubit_num = qubit_num;
        m_state.assign(1ull << m_qubit_num, 0);
        m_state[0] = { 1, 0 };
    }
    else
    {
        m_qubit_num = qubit_num;
        m_state.assign(1ull << m_qubit_num, 0);
        QPANDA_ASSERT(1ll << m_qubit_num != state.size(), "Error: initState size.");
        if (state.size() > m_threshold)
        {
            double prob = 0;
#pragma omp parallel for reduction(+:prob)
            for (int64_t i = 0; i < state.size(); i++)
            {
                prob += std::norm(state[i]);
            }

            QPANDA_ASSERT(std::abs(1 - prob) > DBL_EPSILON, "Error: initState size.");
#pragma omp parallel for
            for (int64_t i = 0; i < state.size(); i++)
            {
                m_state[i] = state[i];
            }
        }
        else
        {
            double prob = 0;
            for (int64_t i = 0; i < state.size(); i++)
            {
                prob += std::norm(state[i]);
            }

            QPANDA_ASSERT(std::abs(1 - prob) > DBL_EPSILON, "Error: initState size.");
            for (int64_t i = 0; i < state.size(); i++)
            {
                m_state[i] = state[i];
            }
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_X(size_t qn)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            std::swap(m_state[real00_idx], m_state[real00_idx | offset]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            std::swap(m_state[real00_idx], m_state[real00_idx | offset]);
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_Y(size_t qn)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = { beta.imag(), -beta.real() };
            m_state[real01_idx] = { -alpha.imag(), alpha.real() };
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = { beta.imag(), -beta.real() };
            m_state[real01_idx] = { -alpha.imag(), alpha.real() };
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_Z(size_t qn)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            m_state[real00_idx | offset] = -m_state[real00_idx | offset];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            m_state[real00_idx | offset] = -m_state[real00_idx | offset];
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_S(size_t qn, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            if (is_dagger)
            {
                m_state[real01_idx] = { m_state[real01_idx].imag(), -m_state[real01_idx].real() };
            }
            else
            {
                m_state[real01_idx] = { -m_state[real01_idx].imag(), m_state[real01_idx].real() };
            }
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            if (is_dagger)
            {
                m_state[real01_idx] = { m_state[real01_idx].imag(), -m_state[real01_idx].real() };
            }
            else
            {
                m_state[real01_idx] = { -m_state[real01_idx].imag(), m_state[real01_idx].real() };
            }
        }
    }

    return qErrorNone;
}


QError CPUImplQPU::_RZ(size_t qn, QStat &matrix, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    if (is_dagger)
    {
        matrix[0] = qcomplex_t(matrix[0].real(), -matrix[0].imag());
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            m_state[real00_idx] *= matrix[0];
            m_state[real01_idx] *= matrix[3];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            m_state[real00_idx] *= matrix[0];
            m_state[real01_idx] *= matrix[3];
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_H(size_t qn, QStat &matrix)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;

            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = (alpha + beta) * SQ2;
            m_state[real01_idx] = (alpha - beta) * SQ2;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;

            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = (alpha + beta) * SQ2;
            m_state[real01_idx] = (alpha - beta) * SQ2;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CNOT(size_t qn_0, size_t qn_1)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            std::swap(m_state[real00_idx | offset0], m_state[real00_idx | offset0 | offset1]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            std::swap(m_state[real00_idx | offset0], m_state[real00_idx | offset0 | offset1]);
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CZ(size_t qn_0, size_t qn_1, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            m_state[real00_idx | offset0 | offset1] = -m_state[real00_idx | offset0 | offset1];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            m_state[real00_idx | offset0 | offset1] = -m_state[real00_idx | offset0 | offset1];
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CR(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    if (is_dagger)
    {
        matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            m_state[real00_idx | offset0 | offset1] *= matrix[15];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            m_state[real00_idx | offset0 | offset1] *= matrix[15];
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_SWAP(size_t qn_0, size_t qn_1)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            std::swap(m_state[real00_idx | offset1], m_state[real00_idx | offset0]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            std::swap(m_state[real00_idx | offset1], m_state[real00_idx | offset0]);
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_iSWAP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (is_dagger)
    {
        matrix[6] = { 0, 1 };
        matrix[9] = { 0, 1 };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_iSWAP_theta(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (is_dagger)
    {
        matrix[6] = { matrix[6].real(), -matrix[6].real() };
        matrix[9] = { matrix[9].real(), -matrix[9].real() };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[5] * phi01 + matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01 + matrix[10] * phi10;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[5] * phi01 + matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01 + matrix[10] * phi10;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CU(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (is_dagger)
    {
        matrix[10] = { matrix[10].real(), -matrix[10].real() };
        matrix[11] = { matrix[14].real(), -matrix[14].real() };
        matrix[14] = { matrix[11].real(), -matrix[11].real() };
        matrix[15] = { matrix[15].real(), -matrix[15].real() };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 | offset1];
            m_state[real00_idx | offset0] = matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 | offset1];
            m_state[real00_idx | offset0] = matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_U1(size_t qn, QStat &matrix, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    if (is_dagger)
    {
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            m_state[real01_idx] *= matrix[3];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;
            m_state[real01_idx] *= matrix[3];
        }
    }
    return qErrorNone;
}


QError CPUImplQPU::_X(size_t qn, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            std::swap(m_state[real00_idx], m_state[real01_idx]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            std::swap(m_state[real00_idx], m_state[real01_idx]);
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_Y(size_t qn, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = { beta.imag(), -beta.real() };
            m_state[real01_idx] = { -alpha.imag(), alpha.real() };
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = { beta.imag(), -beta.real() };
            m_state[real01_idx] = { -alpha.imag(), alpha.real() };
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_Z(size_t qn, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            m_state[real01_idx] = -m_state[real01_idx];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            m_state[real01_idx] = -m_state[real01_idx];
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_S(size_t qn, bool is_dagger, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            if (is_dagger)
            {
                m_state[real01_idx] = { m_state[real01_idx].imag(), -m_state[real01_idx].real() };
            }
            else
            {
                m_state[real01_idx] = { -m_state[real01_idx].imag(), m_state[real01_idx].real() };
            }
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            if (is_dagger)
            {
                m_state[real01_idx] = { m_state[real01_idx].imag(), -m_state[real01_idx].real() };
            }
            else
            {
                m_state[real01_idx] = { -m_state[real01_idx].imag(), m_state[real01_idx].real() };
            }
        }
    }

    return qErrorNone;
}


QError CPUImplQPU::_RZ(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (is_dagger)
    {
        matrix[0] = qcomplex_t(matrix[0].real(), -matrix[0].imag());
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            m_state[real00_idx] *= matrix[0];
            m_state[real01_idx] *= matrix[3];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            m_state[real00_idx] *= matrix[0];
            m_state[real01_idx] *= matrix[3];
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_H(size_t qn, QStat &matrix, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;

            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = (alpha + beta) * SQ2;
            m_state[real01_idx] = (alpha - beta) * SQ2;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;

            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = (alpha + beta) * SQ2;
            m_state[real01_idx] = (alpha - beta) * SQ2;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CNOT(size_t qn_0, size_t qn_1, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            std::swap(m_state[real00_idx | offset0], m_state[real00_idx | offset0 | offset1]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            std::swap(m_state[real00_idx | offset0], m_state[real00_idx | offset0 | offset1]);
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CZ(size_t qn_0, size_t qn_1)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            m_state[real00_idx | offset0 | offset1] = -m_state[real00_idx | offset0 | offset1];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            m_state[real00_idx | offset0 | offset1] = -m_state[real00_idx | offset0 | offset1];
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CR(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (is_dagger)
    {
        matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            m_state[real00_idx | offset0 | offset1] *= matrix[15];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            m_state[real00_idx | offset0 | offset1] *= matrix[15];
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_SWAP(size_t qn_0, size_t qn_1, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            std::swap(m_state[real00_idx | offset1], m_state[real00_idx | offset0]);
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            std::swap(m_state[real00_idx | offset1], m_state[real00_idx | offset0]);
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_iSWAP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (is_dagger)
    {
        matrix[6] = { 0, 1 };
        matrix[9] = { 0, 1 };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_iSWAP_theta(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (is_dagger)
    {
        matrix[6] = { matrix[6].real(), -matrix[6].real() };
        matrix[9] = { matrix[9].real(), -matrix[9].real() };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[5] * phi01 + matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01 + matrix[10] * phi10;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            m_state[real00_idx | offset1] = matrix[5] * phi01 + matrix[6] * phi10;
            m_state[real00_idx | offset0] = matrix[9] * phi01 + matrix[10] * phi10;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_CU(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (is_dagger)
    {
        matrix[10] = { matrix[10].real(), -matrix[10].real() };
        matrix[11] = { matrix[14].real(), -matrix[14].real() };
        matrix[14] = { matrix[11].real(), -matrix[11].real() };
        matrix[15] = { matrix[15].real(), -matrix[15].real() };
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 | offset1];
            m_state[real00_idx | offset0] = matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 | offset1];
            m_state[real00_idx | offset0] = matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_U1(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls)
{
    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (is_dagger)
    {
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            m_state[real01_idx] *= matrix[3];
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            int64_t real01_idx = real00_idx | offset;
            m_state[real01_idx] *= matrix[3];
        }
    }
    return qErrorNone;
}


QError CPUImplQPU::_single_qubit_normal_unitary(size_t qn, QStat &matrix, bool is_dagger)
{
    if (is_dagger)
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

    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;

            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = matrix[0] * alpha + matrix[1] * beta;
            m_state[real01_idx] = matrix[2] * alpha + matrix[3] * beta;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            int64_t real01_idx = real00_idx | offset;

            auto alpha = m_state[real00_idx];
            auto beta = m_state[real01_idx];
            m_state[real00_idx] = matrix[0] * alpha + matrix[1] * beta;
            m_state[real01_idx] = matrix[2] * alpha + matrix[3] * beta;
        }
    }
    return qErrorNone;
}

QError CPUImplQPU::_single_qubit_normal_unitary(size_t qn, Qnum &controls,
    QStat &matrix, bool is_dagger)
{
    if (is_dagger)
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

    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 1, [&](size_t &q) {
        mask |= 1ll << q;
    });

    int64_t size = 1ll << (m_qubit_num - 1);
    int64_t offset = 1ll << qn;

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            auto alpha = m_state[real00_idx];
            auto beta = m_state[real00_idx | offset];
            m_state[real00_idx] = matrix[0] * alpha + matrix[1] * beta;
            m_state[real00_idx | offset] = matrix[2] * alpha + matrix[3] * beta;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn);
            if (mask != (mask & real00_idx))
                continue;
            auto alpha = m_state[real00_idx];
            auto beta = m_state[real00_idx | offset];
            m_state[real00_idx] = matrix[0] * alpha + matrix[1] * beta;
            m_state[real00_idx | offset] = matrix[2] * alpha + matrix[3] * beta;
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_double_qubit_normal_unitary(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger)
{
    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;

    if (is_dagger)
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

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            auto phi00 = m_state[real00_idx];
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 | offset1];

            m_state[real00_idx] = matrix[0] * phi00 + matrix[1] * phi01
                + matrix[2] * phi10 + matrix[3] * phi11;
            m_state[real00_idx | offset1] = matrix[4] * phi00 + matrix[5] * phi01
                + matrix[6] * phi10 + matrix[7] * phi11;
            m_state[real00_idx | offset0] = matrix[8] * phi00 + matrix[9] * phi01
                + matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[12] * phi00 + matrix[13] * phi01
                + matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_0, qn_1);
            auto phi00 = m_state[real00_idx];
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 | offset1];

            m_state[real00_idx] = matrix[0] * phi00 + matrix[1] * phi01
                + matrix[2] * phi10 + matrix[3] * phi11;
            m_state[real00_idx | offset1] = matrix[4] * phi00 + matrix[5] * phi01
                + matrix[6] * phi10 + matrix[7] * phi11;
            m_state[real00_idx | offset0] = matrix[8] * phi00 + matrix[9] * phi01
                + matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[12] * phi00 + matrix[13] * phi01
                + matrix[14] * phi10 + matrix[15] * phi11;
        }
    }

    return qErrorNone;
}

QError CPUImplQPU::_double_qubit_normal_unitary(size_t qn_0, size_t qn_1, Qnum &controls,
    QStat &matrix, bool is_dagger)
{
    if (is_dagger)
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

    int64_t size = 1ll << (m_qubit_num - 2);
    int64_t offset0 = 1ll << qn_0;
    int64_t offset1 = 1ll << qn_1;
    int64_t mask = 0;
    for_each(controls.begin(), controls.end() - 2, [&](size_t &q) {
        mask |= 1ll << q;
    });

    if (size > m_threshold)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_1, qn_0);
            if (mask != (mask & real00_idx))
                continue;

            auto phi00 = m_state[real00_idx];
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 + offset1];

            m_state[real00_idx] = matrix[0] * phi00 + matrix[1] * phi01
                + matrix[2] * phi10 + matrix[3] * phi11;
            m_state[real00_idx | offset1] = matrix[4] * phi00 + matrix[5] * phi01
                + matrix[6] * phi10 + matrix[7] * phi11;
            m_state[real00_idx | offset0] = matrix[8] * phi00 + matrix[9] * phi01
                + matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[12] * phi00 + matrix[13] * phi01
                + matrix[14] * phi10 + matrix[15] * phi11;
        }
    }
    else
    {
        for (int64_t i = 0; i < size; i++)
        {
            int64_t real00_idx = _insert(i, qn_0, qn_1);
            if (mask != (mask & real00_idx))
                continue;

            auto phi00 = m_state[real00_idx];
            auto phi01 = m_state[real00_idx | offset1];
            auto phi10 = m_state[real00_idx | offset0];
            auto phi11 = m_state[real00_idx | offset0 + offset1];

            m_state[real00_idx] = matrix[0] * phi00 + matrix[1] * phi01
                + matrix[2] * phi10 + matrix[3] * phi11;
            m_state[real00_idx | offset1] = matrix[4] * phi00 + matrix[5] * phi01
                + matrix[6] * phi10 + matrix[7] * phi11;
            m_state[real00_idx | offset0] = matrix[8] * phi00 + matrix[9] * phi01
                + matrix[10] * phi10 + matrix[11] * phi11;
            m_state[real00_idx | offset0 | offset1] = matrix[12] * phi00 + matrix[13] * phi01
                + matrix[14] * phi10 + matrix[15] * phi11;
        }
    }

    return qErrorNone;
}

QError  CPUImplQPU::
unitarySingleQubitGate(size_t qn,
    QStat& matrix,
    bool is_dagger,
    GateType type)
{
    switch (type)
    {
    case GateType::I_GATE:
    case GateType::BARRIER_GATE:
    case GateType::ECHO_GATE:
        break;
    case GateType::PAULI_X_GATE:
        _X(qn);
        break;
    case GateType::PAULI_Y_GATE:
        _Y(qn);
        break;
    case GateType::PAULI_Z_GATE:
        _Z(qn);
        break;
    case GateType::S_GATE:
        _S(qn, is_dagger);
        break;
    case GateType::T_GATE:
    case GateType::U1_GATE:
        _U1(qn, matrix, is_dagger);
        break;
    case GateType::RZ_GATE:
    case GateType::Z_HALF_PI:
        _RZ(qn, matrix, is_dagger);
        break;
    case GateType::HADAMARD_GATE:
        _H(qn, matrix);
        break;
    case GateType::X_HALF_PI:
    case GateType::Y_HALF_PI:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::U2_GATE:
    case GateType::U3_GATE:
    case GateType::U4_GATE:
    case GateType::RPHI_GATE:
        _single_qubit_normal_unitary(qn, matrix, is_dagger);
        break;
    default:
        throw std::runtime_error("Error: gate type: " + std::to_string(type));
    }

    return qErrorNone;
}

QError  CPUImplQPU::
controlunitarySingleQubitGate(size_t qn,
    Qnum& controls,
    QStat & matrix,
    bool is_dagger,
    GateType type)
{
    switch (type)
    {
    case GateType::I_GATE:
    case GateType::BARRIER_GATE:
    case GateType::ECHO_GATE:
        break;
    case GateType::PAULI_X_GATE:
        _X(qn, controls);
        break;
    case GateType::PAULI_Y_GATE:
        _Y(qn, controls);
        break;
    case GateType::PAULI_Z_GATE:
        _Z(qn, controls);
        break;
    case GateType::S_GATE:
        _S(qn, is_dagger, controls);
        break;
    case GateType::T_GATE:
    case GateType::U1_GATE:
        _U1(qn, matrix, is_dagger, controls);
        break;
    case GateType::RZ_GATE:
    case GateType::Z_HALF_PI:
        _RZ(qn, matrix, is_dagger, controls);
        break;
    case GateType::HADAMARD_GATE:
        _H(qn, matrix, controls);
        break;
    case GateType::X_HALF_PI:
    case GateType::Y_HALF_PI:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::U2_GATE:
    case GateType::U3_GATE:
    case GateType::U4_GATE:
    case GateType::RPHI_GATE:
        _single_qubit_normal_unitary(qn, controls, matrix, is_dagger);
        break;
    default:
        throw std::runtime_error("Error: gate type: " + std::to_string(type));
    }

    return qErrorNone;
}

QError CPUImplQPU::
unitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool is_dagger,
    GateType type)
{
    switch (type)
    {
    case GateType::CNOT_GATE:
        _CNOT(qn_0, qn_1);
        break;
    case GateType::CZ_GATE:
        _CZ(qn_0, qn_1);
        break;
    case GateType::CPHASE_GATE:
        _CR(qn_0, qn_1, matrix, is_dagger);
        break;
    case GateType::SWAP_GATE:
        _SWAP(qn_0, qn_1);
        break;
    case GateType::ISWAP_GATE:
        _iSWAP(qn_0, qn_1, matrix, is_dagger);
        break;
    case GateType::ISWAP_THETA_GATE:
    case GateType::SQISWAP_GATE:
        _iSWAP_theta(qn_0, qn_1, matrix, is_dagger);
        break;
    case GateType::CU_GATE:
        _CU(qn_0, qn_1, matrix, is_dagger);
        break;
    case GateType::TWO_QUBIT_GATE:
        _double_qubit_normal_unitary(qn_0, qn_1, matrix, is_dagger);
        break;
    default:
        throw std::runtime_error("Error: gate type: " + std::to_string(type));
    }
    return qErrorNone;
}

QError  CPUImplQPU::
controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    Qnum& controls,
    QStat& matrix,
    bool is_dagger,
    GateType type)
{
    switch (type)
    {
    case GateType::CNOT_GATE:
        _CNOT(qn_0, qn_1, controls);
        break;
    case GateType::CZ_GATE:
        _CZ(qn_0, qn_1, controls);
        break;
    case GateType::CPHASE_GATE:
        _CR(qn_0, qn_1, matrix, is_dagger, controls);
        break;
    case GateType::SWAP_GATE:
        _SWAP(qn_0, qn_1, controls);
        break;
    case GateType::ISWAP_GATE:
        _iSWAP(qn_0, qn_1, matrix, is_dagger, controls);
        break;
    case GateType::ISWAP_THETA_GATE:
    case GateType::SQISWAP_GATE:
        _iSWAP_theta(qn_0, qn_1, matrix, is_dagger, controls);
        break;
    case GateType::CU_GATE:
        _CU(qn_0, qn_1, matrix, is_dagger, controls);
        break;
    case GateType::TWO_QUBIT_GATE:
        _double_qubit_normal_unitary(qn_0, qn_1, controls, matrix, is_dagger);
        break;
    default:
        throw std::runtime_error("Error: gate type: " + std::to_string(type));
    }
    return qErrorNone;
}


QError CPUImplQPU::Reset(size_t qn)
{
    bool measure_out = qubitMeasure(qn);
    if (measure_out)
    {
        QStat matrix_x = { 0, 1, 1, 0 };
        _single_qubit_normal_unitary(qn, matrix_x, false);
    }
    return qErrorNone;
}

QStat CPUImplQPU::getQState()
{
    return m_state;
}

QError CPUImplQPU::DiagonalGate(Qnum & vQubit, QStat & matrix, bool isConjugate, double error_rate)
{
    return qErrorNone;
}

QError CPUImplQPU::controlDiagonalGate(Qnum & vQubit, QStat & matrix, Qnum & vControlBit, bool isConjugate, double error_rate)
{
    return qErrorNone;
}

QError CPUImplQPUWithOracle::controlOracularGate(std::vector<size_t> bits, std::vector<size_t> controlbits, bool is_dagger, std::string name)
{
    if (name == "oracle_test") {

    }
    else {
        throw runtime_error("Not Implemented.");
    }
    return qErrorNone;
}
