#ifndef _GPU_QUANTUM_GATE_H
#define _GPU_QUANTUM_GATE_H

#include "QPandaConfig.h"
#ifdef USE_CUDA

#include "Core/Utilities/Tools/Traversal.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.cuh"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUGatesWrapper.cuh"

class GPUImplQPU : public QPUImpl
{
    QStat m_init_state;
    size_t m_qubit_num{0};
    bool m_is_init_state{false};
    std::unique_ptr<DeviceQPU> m_device_qpu;

public:
    GPUImplQPU();
    ~GPUImplQPU();

    QStat getQState();
    size_t getQStateSize();
    QError Reset(size_t qn);

    bool qubitMeasure(size_t qn);
    QError pMeasure(Qnum &qnum, prob_vec &mResult);
    QError initState(size_t qubit_num, const QStat &stat = {});
    QError pMeasure(Qnum &qnum, prob_tuple &mResult, int select_max = -1);
    QError initState(size_t head_rank, size_t rank_size, size_t qubit_num);

    QError OracleGate(Qnum &qubits, QStat &matrix, bool is_dagger);
    QError controlOracleGate(Qnum &qubits, const Qnum &controls, QStat &matrix, bool is_dagger);

    QError DiagonalGate(Qnum &qnum, QStat &matrix, bool is_dagger, double error_rate);
    QError controlDiagonalGate(Qnum &qnum, QStat &matrix, Qnum &controls, bool is_dagger, double error_rate);

    QError unitarySingleQubitGate(size_t qn, QStat &matrix, bool is_dagger, GateType type);
    QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, GateType type);
    QError controlunitarySingleQubitGate(size_t qn, Qnum &qnum, QStat &matrix, bool is_dagger, GateType type);
    QError controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum &qnum, QStat &matrix, bool is_dagger, GateType type);

    virtual void set_parallel_threads_size(size_t size){};

    virtual QError process_noise(Qnum &qnum, QStat &matrix);
    virtual QError debug(std::shared_ptr<QPanda::AbstractQDebugNode> debugger);
};

#endif
#endif