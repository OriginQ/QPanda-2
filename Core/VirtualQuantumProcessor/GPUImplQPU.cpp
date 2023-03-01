/******************************************************************************
Copyright (c) 2017-2023 Origin Quantum Computing Co., Ltd.. All Rights Reserved.

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

#include "Core/Utilities/Tools/Macro.h"
#include "Core/VirtualQuantumProcessor/GPUImplQPU.h"
#ifdef USE_CUDA
#include <map>
#include <thread>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QPandaException.h"

using namespace std;
using std::stringstream;

GPUImplQPU::GPUImplQPU()
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu = make_unique<DeviceQPU>();
}

GPUImplQPU::~GPUImplQPU(){PRINT_DEBUG_MESSAGE}

size_t GPUImplQPU::getQStateSize()
{
    PRINT_DEBUG_MESSAGE
    if (!m_is_init_state)
    {
        return 0;
    }
    else
    {
        return 1ull << m_qubit_num;
    }
}

QError GPUImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{
    PRINT_DEBUG_MESSAGE
    m_qubit_num = qubit_num;
    if (m_is_init_state)
    {
        m_device_qpu->init_state(m_qubit_num,m_init_state);
    }
    else
    {
        m_is_init_state = false;
        m_device_qpu->init_state(m_qubit_num);
    }
    return qErrorNone;
}

QError GPUImplQPU::initState(size_t qubit_num, const QStat &state)
{
    PRINT_DEBUG_MESSAGE
    m_qubit_num = qubit_num;
    if (0 == state.size())
    {
        m_is_init_state = false;
        m_device_qpu->init_state(m_qubit_num);
    }
    else
    {
        m_init_state.resize(state.size()), m_init_state = state;
        m_device_qpu->init_state(m_qubit_num, state);
        m_is_init_state = true;
    }
    return undefineError;
}

QStat GPUImplQPU::getQState()
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->get_qstate(m_init_state);
    return m_init_state;
}

QError GPUImplQPU::unitarySingleQubitGate(size_t qn, QStat &matrix, bool is_dagger, GateType type)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->exec_gate(type, matrix, {qn}, 1, is_dagger);
    return QError::qErrorNone;
}

QError GPUImplQPU::controlunitarySingleQubitGate(size_t qn, Qnum &qnum, QStat &matrix, bool is_dagger, GateType type)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->exec_gate(type, matrix, qnum, 1, is_dagger);
    return qErrorNone;
}

QError GPUImplQPU::unitaryDoubleQubitGate(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, GateType type)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->exec_gate(type, matrix, {qn_0, qn_1}, 2, is_dagger);
    return qErrorNone;
}

QError GPUImplQPU::controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum &qnum, QStat &matrix, bool is_dagger, GateType type)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->exec_gate(type, matrix, qnum, 2, is_dagger);
    return qErrorNone;
}

QError GPUImplQPU::process_noise(Qnum &qnum, QStat &matrix)
{
    PRINT_DEBUG_MESSAGE
    switch (qnum.size())
    {
    case 1:
        m_device_qpu->exec_gate(GateType::NoiseSingle_GATE, matrix, qnum, 1, false);
        break;
    case 2:
        m_device_qpu->exec_gate(GateType::NoiseDouble_GATE, matrix, qnum, 2, false);
        break;
    default:
        throw std::runtime_error("parms error with process_noise");
    }
    return qErrorNone;
}

QError GPUImplQPU::debug(std::shared_ptr<QPanda::AbstractQDebugNode> debugger)
{
    PRINT_DEBUG_MESSAGE
    getQState();
    debugger->save_qstate_ref(m_init_state);
    return qErrorNone;
}

QError GPUImplQPU::DiagonalGate(Qnum &qnum, QStat &matrix, bool is_dagger, double error_rate)
{
    PRINT_DEBUG_MESSAGE
    return qErrorNone;
}

QError GPUImplQPU::controlDiagonalGate(Qnum &qnum, QStat &matrix, Qnum &controls, bool is_dagger, double error_rate)
{
    PRINT_DEBUG_MESSAGE
    return qErrorNone;
}

QError GPUImplQPU::Reset(size_t qn)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->reset(qn);
    return qErrorNone;
}

bool GPUImplQPU::qubitMeasure(size_t qn)
{
    PRINT_DEBUG_MESSAGE
    return m_device_qpu->qubit_measure(qn);
}

QError GPUImplQPU::pMeasure(Qnum &qnum, prob_tuple &mResult, int select_max)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->probs_measure(qnum, mResult, select_max);
    return qErrorNone;
}

QError GPUImplQPU::pMeasure(Qnum &qnum, prob_vec &mResult)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->probs_measure(qnum, mResult);
    return qErrorNone;
}

QError GPUImplQPU::OracleGate(Qnum &qubits, QStat &matrix, bool is_dagger)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->exec_gate(GateType::ORACLE_GATE, matrix, qubits, qubits.size(), is_dagger);
    return qErrorNone;
}

QError GPUImplQPU::controlOracleGate(Qnum &qubits, const Qnum &controls, QStat &matrix, bool is_dagger)
{
    PRINT_DEBUG_MESSAGE
    m_device_qpu->exec_gate(GateType::CORACLE_GATE, matrix, qubits, controls, is_dagger);
    return qErrorNone;
}

#endif