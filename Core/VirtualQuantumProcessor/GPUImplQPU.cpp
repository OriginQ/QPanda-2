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
#include <algorithm>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QPandaException.h"


using std::stringstream;


GPUImplQPU::GPUImplQPU()
{
    m_device_qpu = make_unique<DeviceQPU>();
}

GPUImplQPU::~GPUImplQPU()
{

}

size_t GPUImplQPU::getQStateSize()
{
    if (!m_is_init_state)
        return 0;
    else
    {
        return 1ull << m_qubit_num;
    }
}


QError GPUImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{   
    m_qubit_num = qubit_num;
    if (m_is_init_state)
    {
        m_device_qpu->init_state(m_qubit_num, m_init_state);
    }
    else
    {
        m_device_qpu->init_state(m_qubit_num);
    }

    return qErrorNone;
}

QError GPUImplQPU::initState(size_t qubit_num, const QStat &state)
{
    if (0 == state.size())
    {
        m_device_qpu->init_state(m_qubit_num);
        m_is_init_state = false;
    }
    else
    {
        m_qubit_num = qubit_num;
        m_init_state.resize(1ull << m_qubit_num, 0);
        QPANDA_ASSERT(1ll << m_qubit_num != state.size(), "Error: initState size.");
        m_is_init_state = true;

        for (int64_t i = 0; i < state.size(); i++)
        {
            m_init_state[i] = state[i];
        }
    }
    return undefineError;
}

QStat GPUImplQPU::getQState()
{
    m_device_qpu->get_qstate(m_init_state);
    return m_init_state;
}


QError GPUImplQPU::unitarySingleQubitGate(size_t qn,
    QStat & matrix,
    bool is_dagger,
    GateType type)
{
    m_device_qpu->exec_gate(type, matrix, {qn}, 1, is_dagger);
    return QError::qErrorNone;
}

QError GPUImplQPU::controlunitarySingleQubitGate(size_t qn,
    Qnum& qnum,
    QStat& matrix,
    bool is_dagger,
    GateType type)
{
    m_device_qpu->exec_gate(type, matrix, qnum, 1, is_dagger);
    return qErrorNone;
}

QError GPUImplQPU::unitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    QStat& matrix,
    bool is_dagger,
    GateType type)
{
    m_device_qpu->exec_gate(type, matrix, {qn_0, qn_1}, 2, is_dagger);
    return qErrorNone;
}

QError GPUImplQPU::controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1,
    Qnum& qnum,
    QStat& matrix,
    bool is_dagger,
    GateType type)
{
    m_device_qpu->exec_gate(type, matrix, qnum, 2, is_dagger);
    return qErrorNone;
}

QError GPUImplQPU::DiagonalGate(Qnum & qnum, QStat & matrix, bool is_dagger, double error_rate)
{

    return qErrorNone;
}

QError GPUImplQPU::controlDiagonalGate(Qnum& qnum, QStat & matrix, Qnum& controls,
                                       bool is_dagger, double error_rate)
{

    return qErrorNone;
}

QError GPUImplQPU::Reset(size_t qn)
{
    m_device_qpu->reset(qn);
    return qErrorNone;
}


bool GPUImplQPU::qubitMeasure(size_t qn)
{
    m_device_qpu->qubit_measure(qn);
    return true;
}

QError GPUImplQPU::pMeasure(Qnum& qnum, prob_tuple &mResult,int select_max)
{
    m_device_qpu->probs_measure(qnum, mResult, select_max);
    return qErrorNone;
}

QError GPUImplQPU::pMeasure(Qnum& qnum, prob_vec &mResult)
{
    m_device_qpu->probs_measure(qnum, mResult);
    return qErrorNone;
}

QError GPUImplQPU::OracleGate(Qnum& qubits, QStat &matrix,
                  bool is_dagger)
{
    throw std::invalid_argument("Error: not support QOracle");
}

QError GPUImplQPU::controlOracleGate(Qnum& qubits, const Qnum &controls,
                         QStat &matrix, bool is_dagger)
{
    throw std::invalid_argument("Error: not support QOracle");
}


#endif
