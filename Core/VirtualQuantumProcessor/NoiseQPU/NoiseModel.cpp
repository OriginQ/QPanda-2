
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

#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include <complex>
#include "Core/Utilities/Tools/QStatMatrix.h"
#include <ostream>
#include <iterator>
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseCPUImplQPU.h"
#include <algorithm>
#include <numeric>


USING_QPANDA
using namespace rapidjson;
using namespace std;

static string qubits_to_string(const Qnum &qns)
{
    stringstream ss;
    copy(qns.begin(), qns.end(), ostream_iterator<Qnum::value_type>(ss, "|"));
    return ss.str();
}


QStat matrix_tensor(const QStat &matrix_left, const QStat &matrix_right)
{
    int size = (int)matrix_left.size();
    QStat matrix_result(size*size, 0);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix_result[((i >> 1) << 3) + ((j >> 1) << 2) + ((i % 2) << 1) + (j % 2)] = matrix_left[i] * matrix_right[j];
        }
    }
    return matrix_result;
}

bool equal(const QStat &lhs, const QStat &rhs)
{
    QPANDA_RETURN(lhs.size() != rhs.size(), false);

    for (size_t i = 0; i < lhs.size(); i++)
    {
        QPANDA_RETURN(abs(lhs[i].real() - rhs[i].real()) > FLT_EPSILON ||
                      abs(lhs[i].imag() - rhs[i].imag()) > FLT_EPSILON, false);
    }

    return true;
}


bool damping_kraus_operator(Value &value, NoiseOp & noise)
{
    if ((!value.IsArray()) || (value.Size()!=2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    if (NOISE_MODEL::DAMPING_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    double probability = value[1].GetDouble();

    noise.resize(2);

    noise[0] = { 1,0,0,(qstate_type)sqrt(1 - probability) };
    noise[1] = { 0,(qstate_type)sqrt(probability),0,0 };
    return 1;
}
bool dephasing_kraus_operator(Value & value, NoiseOp & noise)
{
    if ((!value.IsArray()) || (value.Size()!=2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    if (NOISE_MODEL::DEPHASING_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    double probability = value[1].GetDouble();
    noise.resize(2);
    noise[0] = { (qstate_type)sqrt(1 - probability),0,0,(qstate_type)sqrt(1 - probability) };
    noise[1] = { (qstate_type)sqrt(probability),0,0,-(qstate_type)sqrt(probability) };
    return 1;
}
//1 - np.exp(-float(gate_time) / float(T1))
//gamma_phi -= float(gate_time) / float(2 * T1)
//p=.5 * (1 - np.exp(-2 * gamma_phi))
//bool decoherence_kraus_operator(double t1, double t2, double gate_time, NoiseOp & noise)
bool decoherence_kraus_operator(Value & value, NoiseOp & noise)
{
    if ((!value.IsArray())||(value.Size()!=4))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    for (SizeType i = 1; i < value.Size(); i++)
    {
        if (!value[i].IsDouble())
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }
    }

    double t1 = value[1].GetDouble();
    double t2 = value[2].GetDouble();
    double gate_time = value[3].GetDouble();

    Document document;
    document.SetObject();
    auto & alloc =document.GetAllocator();
    NoiseOp damping, dephasing;

    Value damping_value(kArrayType);
    damping_value.PushBack(NOISE_MODEL::DAMPING_KRAUS_OPERATOR,alloc);
    damping_value.PushBack(1 - exp(-gate_time / t1), alloc);
    damping_kraus_operator(damping_value, damping);
    double gamma_phi = gate_time / t2;
    gamma_phi -= gate_time / (2 * t1);

    Value dephasing_value(kArrayType);
    dephasing_value.PushBack(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, alloc);
    dephasing_value.PushBack(0.5 * (1 - exp(-2 * gamma_phi)), alloc);
    dephasing_kraus_operator(dephasing_value, dephasing);

    for (auto iter : damping)
    {
        for (auto iter1 : dephasing)
        {
            noise.push_back(matrix_multiply(iter, iter1));
        }
    }
    return 1;
}

bool double_damping_kraus_operator(Value & value, NoiseOp & noise)
{
    NoiseOp ntemp;
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    if (NOISE_MODEL::DAMPING_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    double probability = value[1].GetDouble();

    ntemp.resize(2);
    ntemp[0] = { 1,0,0,(qstate_type)sqrt(1 - probability) };
    ntemp[1] = { 0,(qstate_type)sqrt(probability),0,0 };
    for (auto i = 0; i < ntemp.size(); i++)
    {
        for (auto j = 0; j < ntemp.size(); j++)
        {
            noise.push_back(matrix_tensor(ntemp[i], ntemp[j]));
        }
    }
    return 1;
}

bool double_dephasing_kraus_operator(Value & value, NoiseOp & noise)
{
    NoiseOp ntemp;
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    if (NOISE_MODEL::DEPHASING_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    double probability = value[1].GetDouble();
    ntemp.resize(2);
    ntemp[0] = { (qstate_type)sqrt(1 - probability),0,0,(qstate_type)sqrt(1 - probability) };
    ntemp[1] = { (qstate_type)sqrt(probability),0,0,-(qstate_type)sqrt(probability) };
    for (auto i = 0; i < ntemp.size(); i++)
    {
        for (auto j = 0; j < ntemp.size(); j++)
        {
            noise.push_back(matrix_tensor(ntemp[i], ntemp[j]));
        }
    }
    return 1;
}

bool double_decoherence_kraus_operator(Value & value, NoiseOp & noise)
{
    NoiseOp ntemp;
    if ((!value.IsArray()) || (value.Size() != 4))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    for (SizeType i = 1; i < value.Size(); i++)
    {
        if (!value[i].IsDouble())
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }
    }

    double t1 = value[1].GetDouble();
    double t2 = value[2].GetDouble();
    double gate_time = value[3].GetDouble();

    Document document;
    document.SetObject();
    auto & alloc = document.GetAllocator();
    NoiseOp damping, dephasing;

    Value damping_value(kArrayType);
    damping_value.PushBack(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, alloc);
    damping_value.PushBack(1 - exp(-gate_time / t1), alloc);
    damping_kraus_operator(damping_value, damping);
    double gamma_phi = gate_time / t2;
    gamma_phi -= gate_time / (2 * t1);

    Value dephasing_value(kArrayType);
    dephasing_value.PushBack(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, alloc);
    dephasing_value.PushBack(0.5 * (1 - exp(-2 * gamma_phi)), alloc);
    dephasing_kraus_operator(dephasing_value, dephasing);

    for (auto iter : damping)
    {
        for (auto iter1 : dephasing)
        {
            ntemp.push_back(matrix_multiply(iter, iter1));
        }
    }
    for (auto i = 0; i < ntemp.size(); i++)
    {
        for (auto j = 0; j < ntemp.size(); j++)
        {
            noise.push_back(matrix_tensor(ntemp[i], ntemp[j]));
        }
    }
    return 1;
}

bool pauli_kraus_map(Value & value, NoiseOp & noise)
{
    if (!value.IsArray())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if ((value.Size() != 5) && (value.Size() != 17))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::PAULI_KRAUS_MAP != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    vector<double> probability;
    for (SizeType i = 1; i < value.Size(); i++)
    {
        if (!value[i].IsDouble())
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }
        else
        {
            probability.push_back(value[i].GetDouble());
        }
    }
    QStat PAULI_I = { 1,0,0,1 };
    QStat PAULI_X = { 0,1,1,0 };
    QStat PAULI_Y = {0,qcomplex_t(0,-1),qcomplex_t(0,1),0 };
    QStat PAULI_Z = { 1,0,0,-1 };
    if (probability.size() == 4)
    {
        noise.resize(4);
        noise[0] = { (qstate_type)sqrt(probability[0]),0,0,(qstate_type)sqrt(probability[0]) };
        noise[1] = { 0,(qstate_type)sqrt(probability[1]),(qstate_type)sqrt(probability[1]),0 };
        noise[2] = { 0,(qstate_type)sqrt(probability[2])*qcomplex_t(0,-1),(qstate_type)sqrt(probability[2])*qcomplex_t(0,1),0 };
        noise[3] = { (qstate_type)sqrt(probability[3]),0,0,-(qstate_type)sqrt(probability[3]) };
    }
    else if (probability.size() == 16)
    {
        noise.resize(16);
        noise[0] = matrix_tensor({ (qstate_type)sqrt(probability[0]),0,0,(qstate_type)sqrt(probability[0]) }, { 1,0,0,1 });
        noise[1] = matrix_tensor({ (qstate_type)sqrt(probability[1]),0,0,(qstate_type)sqrt(probability[1]) }, { 0,1,1,0 });
        noise[2] = matrix_tensor({ (qstate_type)sqrt(probability[2]),0,0,(qstate_type)sqrt(probability[2]) }, { 0,qcomplex_t(0,-1),qcomplex_t(0,1),0 });
        noise[3] = matrix_tensor({ (qstate_type)sqrt(probability[3]),0,0,(qstate_type)sqrt(probability[3]) }, { 1,0,0,-1 });
        noise[4] = matrix_tensor({ 0, (qstate_type)sqrt(probability[4]),(qstate_type)sqrt(probability[4]),0 }, { 1,0,0,1 });
        noise[5] = matrix_tensor({ 0, (qstate_type)sqrt(probability[5]),(qstate_type)sqrt(probability[5]),0 }, { 0,1,1,0 });
        noise[6] = matrix_tensor({ 0, (qstate_type)sqrt(probability[6]),(qstate_type)sqrt(probability[6]),0 }, { 0,qcomplex_t(0,-1),qcomplex_t(0,1),0 });
        noise[7] = matrix_tensor({ 0, (qstate_type)sqrt(probability[7]),(qstate_type)sqrt(probability[7]),0 }, { 1,0,0,-1 });
        noise[8] = matrix_tensor({ 0,(qstate_type)sqrt(probability[8])*qcomplex_t(0,-1),(qstate_type)sqrt(probability[8])*qcomplex_t(0,1),0 }, { 1,0,0,1 });
        noise[9] = matrix_tensor({ 0,(qstate_type)sqrt(probability[9])*qcomplex_t(0,-1),(qstate_type)sqrt(probability[9])*qcomplex_t(0,1),0 }, { 0,1,1,0 });
        noise[10] = matrix_tensor({ 0,(qstate_type)sqrt(probability[10])*qcomplex_t(0,-1),(qstate_type)sqrt(probability[10])*qcomplex_t(0,1),0 },
            { 0,qcomplex_t(0,-1),qcomplex_t(0,1),0 });
        noise[11] = matrix_tensor({ 0,(qstate_type)sqrt(probability[11])*qcomplex_t(0,-1),(qstate_type)sqrt(probability[11])*qcomplex_t(0,1),0 }, { 1,0,0,-1 });
        noise[12] = matrix_tensor({ (qstate_type)sqrt(probability[12]),0,0,-(qstate_type)sqrt(probability[12]) }, { 1,0,0,1 });
        noise[13] = matrix_tensor({ (qstate_type)sqrt(probability[13]),0,0,-(qstate_type)sqrt(probability[13]) }, { 0,1,1,0 });
        noise[14] = matrix_tensor({ (qstate_type)sqrt(probability[14]),0,0,-(qstate_type)sqrt(probability[14]) }, { 0,qcomplex_t(0,-1),qcomplex_t(0,1),0 });
        noise[15] = matrix_tensor({ (qstate_type)sqrt(probability[15]),0,0,-(qstate_type)sqrt(probability[15]) }, { 1,0,0,-1 });
    }
    return 1;
}

SingleGateNoiseModeMap & SingleGateNoiseModeMap::getInstance()
{
    static SingleGateNoiseModeMap map;
    return map;
}

noise_mode_function SingleGateNoiseModeMap::operator[](NOISE_MODEL type)
{
    auto iter = m_function_map.find(type);
    if (iter == m_function_map.end())
    {
        QCERR("noise model type error");
        throw invalid_argument("noise model type error");
    }

    return iter->second;
}

SingleGateNoiseModeMap::SingleGateNoiseModeMap()
{
    m_function_map.insert(make_pair(DAMPING_KRAUS_OPERATOR, damping_kraus_operator));
    m_function_map.insert(make_pair(DEPHASING_KRAUS_OPERATOR, dephasing_kraus_operator));
    m_function_map.insert(make_pair(DECOHERENCE_KRAUS_OPERATOR, decoherence_kraus_operator));
    m_function_map.insert(make_pair(PAULI_KRAUS_MAP, pauli_kraus_map));

    m_function_map.insert(make_pair(KRAUS_MATRIX_OPRATOR, kraus_matrix_oprator));

    m_function_map.insert({DECOHERENCE_KRAUS_OPERATOR_P1_P2, decoherence_kraus_operator_p1_p2});
    m_function_map.insert({BITFLIP_KRAUS_OPERATOR, bitflip_kraus_operator});
    m_function_map.insert({DEPOLARIZING_KRAUS_OPERATOR, depolarizing_kraus_operator});
    m_function_map.insert({BIT_PHASE_FLIP_OPRATOR, bit_phase_flip_operator});
    m_function_map.insert({PHASE_DAMPING_OPRATOR, phase_damping_oprator});
}

DoubleGateNoiseModeMap & DoubleGateNoiseModeMap::getInstance()
{
    static DoubleGateNoiseModeMap map;
    return map;
}

noise_mode_function DoubleGateNoiseModeMap::operator[](NOISE_MODEL type)
{
    auto iter = m_function_map.find(type);
    if (iter == m_function_map.end())
    {
        QCERR("noise model type error");
        throw invalid_argument("noise model type error");
    }

    return iter->second;
}


DoubleGateNoiseModeMap::DoubleGateNoiseModeMap()
{
    m_function_map.insert(make_pair(DAMPING_KRAUS_OPERATOR, double_damping_kraus_operator));
    m_function_map.insert(make_pair(DEPHASING_KRAUS_OPERATOR, double_dephasing_kraus_operator));
    m_function_map.insert(make_pair(DECOHERENCE_KRAUS_OPERATOR, double_decoherence_kraus_operator));
    m_function_map.insert(make_pair(PAULI_KRAUS_MAP, pauli_kraus_map));

    m_function_map.insert(make_pair(KRAUS_MATRIX_OPRATOR, kraus_matrix_oprator));

    m_function_map.insert({DECOHERENCE_KRAUS_OPERATOR_P1_P2, double_decoherence_kraus_operator_p1_p2});
    m_function_map.insert({BITFLIP_KRAUS_OPERATOR, double_bitflip_kraus_operator});
    m_function_map.insert({DEPOLARIZING_KRAUS_OPERATOR, double_depolarizing_kraus_operator});
    m_function_map.insert({BIT_PHASE_FLIP_OPRATOR, double_bit_phase_flip_operator});
    m_function_map.insert({PHASE_DAMPING_OPRATOR, double_phase_damping_oprator});

}
bool decoherence_kraus_operator_p1_p2(Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() != 3))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR_P1_P2 != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    for (SizeType i = 1; i < value.Size(); i++)
    {
        if (!value[i].IsDouble())
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }
    }

    double p1 = value[1].GetDouble();
    double p2 = value[2].GetDouble();
    Document document;
    document.SetObject();
    auto & alloc = document.GetAllocator();
    NoiseOp damping, dephasing;
    Value damping_value(kArrayType);
    damping_value.PushBack(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, alloc);
    damping_value.PushBack(p1, alloc);
    damping_kraus_operator(damping_value, damping);

    Value dephasing_value(kArrayType);
    dephasing_value.PushBack(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, alloc);
    dephasing_value.PushBack(p2, alloc);
    dephasing_kraus_operator(dephasing_value, dephasing);

    for (auto iter : damping)
    {
        for (auto iter1 : dephasing)
        {
            noise.push_back(matrix_multiply(iter, iter1));
        }
    }

    return true;
}

bool bitflip_kraus_operator(Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::BITFLIP_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    double probability = value[1].GetDouble();
    noise.resize(2);
    noise[0] = {static_cast<qstate_type>(sqrt(1 - probability)), 0, 0, static_cast<qstate_type>(sqrt(1 - probability))};
    noise[1] = {0, static_cast<qstate_type>(sqrt(probability)), static_cast<qstate_type>(sqrt(probability)), 0};

    return true;
}

bool depolarizing_kraus_operator(Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    QStat matrix_i = {1, 0, 0, 1};
    QStat matrix_x = {0, 1, 1, 0};
    QStat matrix_y = {0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0};
    QStat matrix_z = {1, 0, 0, -1};

    double probability = value[1].GetDouble();
    noise.resize(4);
    noise[0] = static_cast<qstate_type>(sqrt(1 - probability * 0.75)) * matrix_i;
    noise[1] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_x;
    noise[2] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_y;
    noise[3] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_z;

    return true;
}

bool bit_phase_flip_operator(Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    double probability = value[1].GetDouble();
    noise.resize(2);
    noise[0] = {static_cast<qstate_type>(sqrt(1 - probability)), 0, 0, static_cast<qstate_type>(sqrt(1 - probability))};
    noise[1] = {0, qcomplex_t(0, -sqrt(probability)), qcomplex_t(0, sqrt(probability)), 0};

    return true;
}

bool phase_damping_oprator(Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::PHASE_DAMPING_OPRATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    double probability = value[1].GetDouble();
    noise.resize(2);
    noise[0] = {1, 0, 0, static_cast<qstate_type>(sqrt(1 - probability))};
    noise[1] = {0, 0, 0, static_cast<qstate_type>(sqrt(probability))};

    return true;
}

bool double_decoherence_kraus_operator_p1_p2(Value &value, NoiseOp &noise)
{
    NoiseOp ntemp;
    if ((!value.IsArray()) || (value.Size() != 3))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR_P1_P2 != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    for (SizeType i = 1; i < value.Size(); i++)
    {
        if (!value[i].IsDouble())
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }
    }

    double p1 = value[1].GetDouble();
    double p2 = value[2].GetDouble();
    Document document;
    document.SetObject();
    auto & alloc = document.GetAllocator();
    NoiseOp damping, dephasing;
    Value damping_value(kArrayType);
    damping_value.PushBack(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, alloc);
    damping_value.PushBack(p1, alloc);
    damping_kraus_operator(damping_value, damping);

    Value dephasing_value(kArrayType);
    dephasing_value.PushBack(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, alloc);
    dephasing_value.PushBack(p2, alloc);
    dephasing_kraus_operator(dephasing_value, dephasing);

    for (auto iter : damping)
    {
        for (auto iter1 : dephasing)
        {
            ntemp.push_back(matrix_multiply(iter, iter1));
        }
    }

    for (auto i = 0; i < ntemp.size(); i++)
    {
        for (auto j = 0; j < ntemp.size(); j++)
        {
            noise.push_back(matrix_tensor(ntemp[i], ntemp[j]));
        }
    }

    return true;
}

bool double_bitflip_kraus_operator(Value &value, NoiseOp &noise)
{
    NoiseOp ntemp;
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::BITFLIP_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    double probability = value[1].GetDouble();

    ntemp.resize(2);
    ntemp[0] = {static_cast<qstate_type>(sqrt(1 - probability)), 0, 0, static_cast<qstate_type>(sqrt(1 - probability))};
    ntemp[1] = {0, static_cast<qstate_type>(sqrt(probability)), static_cast<qstate_type>(sqrt(probability)), 0};
    for (auto i = 0; i < ntemp.size(); i++)
    {
        for (auto j = 0; j < ntemp.size(); j++)
        {
            noise.push_back(matrix_tensor(ntemp[i], ntemp[j]));
        }
    }

    return true;
}

bool double_depolarizing_kraus_operator(Value &value, NoiseOp &noise)
{
    NoiseOp ntemp;
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    QStat matrix_i = {1, 0, 0, 1};
    QStat matrix_x = {0, 1, 1, 0};
    QStat matrix_y = {0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0};
    QStat matrix_z = {1, 0, 0, -1};

    double probability = value[1].GetDouble();
    ntemp.resize(4);
    ntemp[0] = static_cast<qstate_type>(sqrt(1 - probability * 0.75)) * matrix_i;
    ntemp[1] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_x;
    ntemp[2] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_y;
    ntemp[3] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_z;

    for (auto i = 0; i < ntemp.size(); i++)
    {
        for (auto j = 0; j < ntemp.size(); j++)
        {
            noise.push_back(matrix_tensor(ntemp[i], ntemp[j]));
        }
    }

    return true;
}

bool double_bit_phase_flip_operator(Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    NoiseOp temp(2);
    double probability = value[1].GetDouble();
    temp[0] = {static_cast<qstate_type>(sqrt(1 - probability)), 0, 0, static_cast<qstate_type>(sqrt(1 - probability))};
    temp[1] = {0, qcomplex_t(0, -sqrt(probability)), qcomplex_t(0, sqrt(probability)), 0};

    for (auto i = 0; i < temp.size(); i++)
    {
        for (auto j = 0; j < temp.size(); j++)
        {
            noise.push_back(matrix_tensor(temp[i], temp[j]));
        }
    }

    return true;
}

bool double_phase_damping_oprator(Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() != 2))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::PHASE_DAMPING_OPRATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (!value[1].IsDouble())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    NoiseOp temp(2);
    double probability = value[1].GetDouble();
    temp[0] = {1, 0, 0, static_cast<qstate_type>(sqrt(1 - probability))};
    temp[1] = {0, 0, 0, static_cast<qstate_type>(sqrt(probability))};

    for (auto i = 0; i < temp.size(); i++)
    {
        for (auto j = 0; j < temp.size(); j++)
        {
            noise.push_back(matrix_tensor(temp[i], temp[j]));
        }
    }

    return true;
}

bool kraus_matrix_oprator(rapidjson::Value &value, NoiseOp &noise)
{
    if ((!value.IsArray()) || (value.Size() == 1))
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (NOISE_MODEL::KRAUS_MATRIX_OPRATOR != (NOISE_MODEL)value[0].GetUint())
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    for (int i = 1; i < value.Size(); i++)
    {
        if (!value[i].IsArray() || 8 != value[i].Size())
        {
            QCERR("param error");
            throw std::invalid_argument("param error");
        }

        std::vector<qcomplex_t> temp;
        for (int j = 0; j < value[i].Size(); j += 2)
        {
            qcomplex_t val(value[i][j].GetDouble(), value[i][j + 1].GetDouble());
            temp.push_back(val);
        }

        noise.push_back(temp);
    }

    return true;
}


NoisyQuantum::NoisyQuantum()
{
}


bool NoisyQuantum::sample_noisy_op(GateType type, const Qnum &qns,
                                   NOISE_MODEL &model, NoiseOp &ops,
                                   Qnum &effect_qubits, RandomEngine19937 &rng)
{
    auto gate_type_noise_iter_map = m_noisy.find(type);
    QPANDA_RETURN(m_noisy.end() == gate_type_noise_iter_map, false);

    auto gate_qubit_noise_iter = gate_type_noise_iter_map->second.find("");
    if (gate_type_noise_iter_map->second.end() == gate_qubit_noise_iter)
    {
        auto qubits_str = qubits_to_string(qns);
        gate_qubit_noise_iter = gate_type_noise_iter_map->second.find(qubits_str);
    }
    QPANDA_RETURN(gate_type_noise_iter_map->second.end() == gate_qubit_noise_iter, false);

    auto quantum_idx = gate_qubit_noise_iter->second;
    Qnum noise_qubits;

    QuantumError quantum_error = m_quamtum_error.at(quantum_idx);
    model = quantum_error.get_noise_model();
    if (NOISE_MODEL::MIXED_UNITARY_OPRATOR == quantum_error.get_noise_model())
    {
        m_quamtum_error.at(quantum_idx).sample_noise(ops, noise_qubits, rng);
    }
    else
    {
        m_quamtum_error.at(quantum_idx).sample_noise(model, ops, noise_qubits, rng);
    }

    effect_qubits.reserve(noise_qubits.size());

    for (auto & idx : noise_qubits)
    {
        effect_qubits.push_back(qns[idx]);
    }

    return true;
}

bool NoisyQuantum::sample_noisy_op(GateType type, const Qnum &qns, NoiseOp &ops, Qnum &effect_qubits, RandomEngine19937 &rng)
{
    auto gate_type_noise_iter_map = m_noisy.find(type);
    QPANDA_RETURN(m_noisy.end() == gate_type_noise_iter_map, false);

    auto gate_qubit_noise_iter = gate_type_noise_iter_map->second.find("");
    if (gate_type_noise_iter_map->second.end() == gate_qubit_noise_iter)
    {
        auto qubits_str = qubits_to_string(qns);
        gate_qubit_noise_iter = gate_type_noise_iter_map->second.find(qubits_str);
    }
    QPANDA_RETURN(gate_type_noise_iter_map->second.end() == gate_qubit_noise_iter, false);

    auto quantum_idx = gate_qubit_noise_iter->second;
    Qnum noise_qubits;
    m_quamtum_error.at(quantum_idx).sample_noise(ops, noise_qubits, rng);
    effect_qubits.reserve(noise_qubits.size());

    for (auto & idx : noise_qubits)
    {
        effect_qubits.push_back(qns[idx]);
    }

    return true;
}

bool NoisyQuantum::sample_noisy_op(size_t qn, std::vector<std::vector<double> > &readout, RandomEngine19937 &rng)
{
    auto gate_type_noise_iter_map = m_noisy.find(GATE_TYPE_READOUT);
    QPANDA_RETURN(m_noisy.end() == gate_type_noise_iter_map, false);

    auto gate_qubit_noise_iter = gate_type_noise_iter_map->second.find("");
    if (gate_type_noise_iter_map->second.end() == gate_qubit_noise_iter)
    {
        auto qubits_str = qubits_to_string({qn});
        gate_qubit_noise_iter = gate_type_noise_iter_map->second.find(qubits_str);
    }
    QPANDA_RETURN(gate_type_noise_iter_map->second.end() == gate_qubit_noise_iter, false);

    auto quantum_idx = gate_qubit_noise_iter->second;
    Qnum noise_qubits;
    m_quamtum_error.at(quantum_idx).sample_readout(readout);
    return true;
}


void NoisyQuantum::add_quamtum_error(GateType type, const QuantumError &quantum_error,
                                     const QuantumError::noise_qubits_t &noise_qubits)
{
    auto push_error = [&](GateType type, const QuantumError &quantum_error, const Qnum &effect_qubits)->void
    {
        m_quamtum_error.push_back(quantum_error);
        auto qubits_str = qubits_to_string(effect_qubits);
        qubit_quantum_error_map_t qubit_quantum_error_map;

        auto noise_type_iter = m_noisy.find(type);
        if (noise_type_iter == m_noisy.end())
        {
            qubit_quantum_error_map_t new_qubit_quantum_error;
            new_qubit_quantum_error.insert({qubits_str, m_quamtum_error.size() - 1});
            m_noisy.insert({type, new_qubit_quantum_error});
        }
        else
        {
            noise_type_iter->second.insert({qubits_str, m_quamtum_error.size() - 1});
        }
    };

    if (0 == noise_qubits.size())
    {
        push_error(type, quantum_error, {});
        return ;
    }

    size_t type_qubit_num = quantum_error.get_qubit_num();
    for (auto & effect_qubits : noise_qubits)
    {
        if (type_qubit_num != effect_qubits.size())
        {
            throw std::runtime_error("Error: noise qubit");
        }
        push_error(type, quantum_error, effect_qubits);
    }

    return ;
}


QuantumError::QuantumError()
{
}

QuantumError::QuantumError(const NOISE_MODEL &model, double prob, size_t qubit_num)
{
    set_noise(model, prob, qubit_num);
}

void QuantumError::set_noise(const NOISE_MODEL &model, double prob, size_t qubit_num)
{
    QPANDA_ASSERT(prob < 0 || prob > 1, "Error: noise prob range");
    m_model = model;
    m_qubit_num = qubit_num;

    switch (m_model)
    {
    case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR: // X
    case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR: // Y
    case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR: // Z
    case NOISE_MODEL::PHASE_DAMPING_OPRATOR:
        _set_pauli_noise(model, prob);
        break;
    case NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR:
        _set_depolarizing_noise(prob);
        break;
    case NOISE_MODEL::DAMPING_KRAUS_OPERATOR:
        _set_dampling_noise(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, prob);
        break;
    default:
        throw std::runtime_error("Error: NOISE_MODEL");
    }

    return ;
}

void QuantumError::set_noise(const NOISE_MODEL &model, double T1, double T2, double t_gate, size_t qubit_num)
{
    QPANDA_ASSERT(T1 < 0, "Error: param T1.");
    QPANDA_ASSERT(T2  < 0, "Error: param T2.");
    QPANDA_ASSERT(t_gate < 0, "Error: param t_gate");

    m_model = model;
    m_qubit_num = qubit_num;
    switch (m_model)
    {
    case NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR:
        _set_decoherence_noise(model, T1, T2, t_gate);
        break;
    default:
        throw std::runtime_error("Error: NOISE_MODEL");
    }
}

void QuantumError::set_noise(const NOISE_MODEL &model, const std::vector<QStat> &unitary_matrices,
                             const std::vector<double> &probs, size_t qubit_num)
{
    for (auto &prob : probs)
    {
        QPANDA_ASSERT(prob < 0 || prob > 1, "Error: noise prob range");
    }

    m_qubit_num = qubit_num;
    m_model = model;
    QPANDA_ASSERT(probs.size() != unitary_matrices.size(), "Error: mixed_unitary_noise paramters.");
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    QPANDA_ASSERT(std::abs(sum - 1) > FLT_EPSILON, "Error: mixed_unitary_noise paramters.");

    m_probs = probs;
    m_ops.reserve(unitary_matrices.size());
    for (auto &matrix : unitary_matrices)
    {
        QPANDA_ASSERT((1 != qubit_num && 2 != qubit_num) |
                      (1 == qubit_num && 4 != matrix.size()) |
                      (2 == qubit_num && 16 != matrix.size()),
                      "Error: mixed_unitary_noise paramters.");
        QPANDA_ASSERT(!is_unitary_matrix(matrix, FLT_EPSILON), "Error: mixed_unitary_noise paramters.");
        m_ops.push_back({matrix});
    }

    if (1 == m_qubit_num)
    {
        m_noise_qubits.assign(unitary_matrices.size(), {0});
    }
    else if (2 == m_qubit_num)
    {
        m_noise_qubits.assign(unitary_matrices.size(), {0, 1});
    }
    else
    {
        throw std::runtime_error("Error: noise qubit num");
    }

    return ;
}

void QuantumError::set_noise(const NOISE_MODEL &model, const std::vector<QStat> &unitary_matrices, size_t qubit_num)
{
    m_model = model;
    m_probs = {1};
    m_qubit_num = qubit_num;
    return ;
}


void QuantumError::set_reset_error(double p0, double p1)
{
    QPANDA_ASSERT(p0 < 0 || p0 > 1 || p1 < 0 || p1 > 1, "Error: noise prob range");
    m_qubit_num = 1;
    m_probs = {1 - p0 - p1, p0, p1};

    const NoiseOp ops = {
        {1., 0, 0, 1.},
        {1., 0, 0, 0}, // p0 to reset
        {0, 1., 1., 0},
    };

    m_ops = {{ops[0]}, {ops[1]}, {{ops[1], ops[2]}}};
    m_noise_qubits = {{0}, {0}, {0}};
    m_probs = {1- p0 - p1, p0, p1};

    return ;
}

void QuantumError::set_readout_error(const std::vector<std::vector<double> > &probs_list, size_t qubit_num)
{
    auto check_probs = [&](const vector<double> &probs)->bool
    {
        QPANDA_RETURN(probs.size() != 2, false);

        double sum = 0;
        for (auto &prob : probs)
        {
            QPANDA_ASSERT(prob < 0 || prob > 1, "Error: noise prob range");
            sum += prob;
        }
 
        QPANDA_RETURN(std::abs(sum - 1) > FLT_EPSILON, false);

        return true;
    };

    for (auto probs : probs_list)
    {
        QPANDA_ASSERT(!check_probs(probs), "Error: readout paramters.");
    }

    m_qubit_num = qubit_num;
    m_readout_probs_list = probs_list;
    return ;
}


bool QuantumError::sample_noise(NOISE_MODEL &model, NoiseOp &noise_ops,
                                Qnum &noise_qubits, RandomEngine19937 &rng)
{
    std::function<bool(NoiseOp &, Qnum &)> lambda;
    model = m_model;

    switch (m_model)
    {
    case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR: // X
    case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR: // Y
    case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR: // Z
    case NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR:
    case NOISE_MODEL::PHASE_DAMPING_OPRATOR:
    case NOISE_MODEL::DAMPING_KRAUS_OPERATOR:
    case NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR:
        lambda = [&](NoiseOp &noise_ops, Qnum &noise_qubits)->bool
        {
            size_t random_idx = rng.random_discrete(m_probs);
            noise_ops = m_ops[random_idx];
            noise_qubits = m_noise_qubits[random_idx];

            return true;
        };
        break;
    default:
        throw std::runtime_error("Error: NOISE_MODEL");
    }

    return lambda(noise_ops, noise_qubits);
}

bool QuantumError::sample_noise(NoiseOp &noise_ops, Qnum &noise_qubits, RandomEngine19937 &rng)
{
    size_t random_idx = rng.random_discrete(m_probs);
    noise_ops = m_ops[random_idx];
    noise_qubits = m_noise_qubits[random_idx];

    return true;
}

bool QuantumError::sample_readout(vector<vector<double> > &readout)
{
    readout = m_readout_probs_list;
    return true;
}

void QuantumError::set_qubit_num(int num)
{
    m_qubit_num = num;
    return ;
}

int QuantumError::get_qubit_num() const
{
    return m_qubit_num;
}

void QuantumError::_set_pauli_noise(NOISE_MODEL model, double prob)
{
    auto noise_lamda = [&](double prob, const NoiseOp &ops)->void
    {
        if (1 == m_qubit_num)
        {
            m_probs = {prob, 1 - prob};
            m_ops = {{ops[0]}, {ops[1]}};
            m_noise_qubits = {{0}, {0}};
        }
        else if (2 == m_qubit_num)
        {
            m_probs = {prob * prob, prob * (1 - prob),
                       prob* (1 - prob), (1 - prob) * (1 - prob)};
            m_ops = {{ops[0], ops[0]}, {ops[0]}, {ops[0]}, {ops[1]}};
            m_noise_qubits = {{0, 1}, {1}, {0}, {0}};
        }
        else
        {
            throw std::runtime_error("Error: noise qubit num");
        }

        return;
    };

    NoiseOp ops(2);
    ops[1] = {1, 0, 0, 1};
    switch (model)
    {
    case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR: // X
        ops[0] = {0, 1., 1., 0};
        break;
    case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR: // Y
        ops[0] = {0, -1i, 1i, 0};
        break;
    case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR: // Z
        ops[0] = {1., 0, 0, -1.};
        break;
    case NOISE_MODEL::PHASE_DAMPING_OPRATOR: // Z
        ops[0] = { 1., 0, 0, -1. };
        prob = (1. - sqrt(1. - prob)) / 2;
        break;
    default:
        throw std::runtime_error("Error: noise model");
    }

    noise_lamda(prob, ops);
    return ;
}

void QuantumError::_set_depolarizing_noise(double prob)
{
    const NoiseOp ops = {
        {0, 1., 1., 0},
        {0, -1i, 1i, 0},
        {1., 0, 0, -1.},
        {1, 0, 0, 1}
    };

    if (1 == m_qubit_num)
    {
        auto prob_i = 1 - prob / 4 * 3;
        m_probs = {prob / 4, prob / 4, prob / 4, prob_i};
        m_ops = {{ops[0]}, {ops[1]}, {ops[2]}, {ops[3]}};
        m_noise_qubits = {{0}, {0}, {0}, {0}};
    }
    else if (2 == m_qubit_num)
    {
        m_probs.reserve(16);
        auto prob_i = 1 - prob / 16 * 15;
        m_probs.insert(m_probs.begin(), 15, prob / 16);
        m_probs.push_back(prob_i);

        m_ops = { {ops[0]},{ops[1]},{ops[2]}, {ops[0]},
                  {ops[0], ops[0]},{ops[1], ops[0]},{ops[2], ops[0]},{ops[1]},
                  {ops[0], ops[1]},{ops[1], ops[1]},{ops[2], ops[1]},{ops[2]},
                  {ops[0], ops[2]},{ops[1], ops[2]},{ops[2], ops[2]},{ops[3]}
                };
        m_noise_qubits = { {0}, {0}, {0}, {1},
                           {0,1}, {0,1}, {0,1},{1},
                           {0,1}, {0,1}, {0,1},{1},
                           {0,1}, {0,1}, {0,1},{0}
                         };
    }
    else
    {
        throw std::runtime_error("Error: noise qubit num");
    }

    return ;
}

void QuantumError::_set_dampling_noise(NOISE_MODEL model, double prob)
{
    auto noise_lamda = [&](double prob, const NoiseOp &ops)->void
    {
        if (1 == m_qubit_num)
        {
            m_probs = { prob };
            m_ops = { {ops[0], ops[1]} };
            m_noise_qubits = { {0} };
        }
        else if (2 == m_qubit_num)
        {
            m_probs = { prob };
            auto tensor_ops = _noise_ops_tensor(ops);
            _optimize_ops(tensor_ops);
            m_ops = { tensor_ops };
            m_noise_qubits = { {0, 1} };
        }
        else
        {
            throw std::runtime_error("Error: noise qubit num");
        }

        return;
    };

    NoiseOp ops = {
        {1., 0, 0, sqrt(1. - prob)},
        {0, sqrt(prob), 0, 0}
    };

    noise_lamda(1, ops);
    return;
}

void QuantumError::_set_decoherence_noise(NOISE_MODEL model, double T1, double T2, double t_gate)
{
    auto noise_lamda = [&](double prob, const NoiseOp &ops)->void
    {
        if (1 == m_qubit_num)
        {
            m_probs = { prob };
            m_ops = { ops };
            m_noise_qubits = { {0} };
        }
        else if (2 == m_qubit_num)
        {
            m_probs = { prob };
            auto tensor_ops = _noise_ops_tensor(ops);
            _optimize_ops(tensor_ops);
            m_ops = { tensor_ops };
            m_noise_qubits = { {0, 1} };
        }
        else
        {
            throw std::runtime_error("Error: noise qubit num");
        }

        return;
    };

    double p_damping = 1. - std::exp(-(t_gate / T1));
    double p_dephasing = 0.5 * (1. - std::exp(-(t_gate / T2 - t_gate / (2 * T1))));

    const NoiseOp ops = {
        { std::sqrt(1 - p_dephasing), 0, 0, std::sqrt((1 - p_damping)*(1 - p_dephasing)) },
        { 0, std::sqrt(p_damping*(1 - p_dephasing)), 0, 0 },
        { 0, std::sqrt(p_damping*(1 - p_dephasing)), 0, 0 },
        { 0, -std::sqrt(p_damping*p_dephasing), 0, 0 }
    };

    noise_lamda(1, ops);
    return ;
}

NoiseOp QuantumError::_noise_ops_tensor(const NoiseOp & ops)
{
    NoiseOp tensor_ops;
    tensor_ops.reserve(1ull << ops.size());
    for (size_t i = 0; i < ops.size(); i++)
    {
        for (size_t j = 0; j < ops.size(); j++)
        {
            tensor_ops.push_back(matrix_tensor(ops[i], ops[j]));
        }
    }

    return tensor_ops;
}

NoiseOp QuantumError::_noise_ops_tensor(const NoiseOp &lhs, const NoiseOp &rhs)
{
    QPANDA_ASSERT(lhs.size() != rhs.size(), "Error: NoiseOp tensor");
    auto size = lhs.size();
    NoiseOp res;
    res.reserve(1ull << size);

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            res.push_back(matrix_tensor(lhs[i], rhs[j]));
        }
    }

    return res;
}

NoiseOp QuantumError::_combine(const NoiseOp &lhs, const NoiseOp &rhs)
{
    QPANDA_ASSERT(lhs.size() != rhs.size(), "Error: NoiseOp combine");
    auto size = lhs.size();
    NoiseOp res;
    res.reserve(1ull << size);

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            res.push_back(lhs[i] * rhs[j]);
        }
    }

    _optimize_ops(res);

    return res;
}

NoiseOp QuantumError::_tensor(const NoiseOp &lhs, const NoiseOp &rhs)
{
    QPANDA_ASSERT(lhs.size() != rhs.size(), "Error: NoiseOp combine");
    auto size = lhs.size();
    NoiseOp res;
    res.resize(size * size);

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            //res.push_back(matrix_tensor(lhs[i], rhs[j]));
            res[j + size * i] = matrix_tensor(lhs[i], rhs[j]);
        }
    }

    _optimize_ops(res);
    return res;
}

bool QuantumError::_optimize_ops(NoiseOp &ops)
{
    auto is_zero_matrix = [](const QStat &matrix)->bool
    {
        for (auto &ele : matrix)
        {
            QPANDA_RETURN(abs(ele.real()) > FLT_EPSILON || abs(ele.imag()) > FLT_EPSILON, false);
        }
        return true;
    };

    for (size_t idx = 0; idx < ops.size(); idx++)
    {
        if (is_zero_matrix(ops[idx]))
        {
            ops.erase(ops.begin() + idx);
            idx--;
        }
    }

    vector<double> factors;
    factors.reserve(ops.size());
    for(auto &op : ops)
    {
        auto max_iter = std::max_element(op.begin(), op.end(), [](const qcomplex_t &lhs, const qcomplex_t &rhs)
        {
            return std::norm(lhs) < std::norm(rhs);
        });

        auto factor = sqrt(std::norm(*max_iter));
        factors.push_back(factor);
        for_each(op.begin(), op.end(), [&](qcomplex_t &value)->void{
            value /= factor;
        });
    }

    for (size_t i = 0; i < ops.size() - 1; i++)
    {
        for (size_t j = i + 1; j < ops.size(); j++)
        {
            if (equal(ops[i], ops[j]))
            {
                factors[i] = sqrt(std::norm(factors[i]) + std::norm(factors[j]));
                factors.erase(factors.begin() + j);
                ops.erase(ops.begin() + j);
                j--;
            }
        }
    }

    for (size_t i = 0; i < ops.size(); i++)
    {
        for (size_t j = 0; j < ops[i].size(); j++)
        {
            ops[i][j] *= factors[i];
        }
    }

    return true;
}

NOISE_MODEL QuantumError::get_noise_model()
{
    return m_model;
}






















