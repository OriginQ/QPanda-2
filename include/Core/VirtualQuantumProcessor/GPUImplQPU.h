#ifndef _GPU_QUANTUM_GATE_H
#define _GPU_QUANTUM_GATE_H
#include "QPandaConfig.h"

#ifdef USE_CUDA

#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.cuh"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUGatesWrapper.cuh"



class GPUImplQPU : public QPUImpl
{
    QStat m_state;
    QStat m_init_state;
    size_t m_qubit_num{ 0 };
    bool m_is_init_state{ false };
   std::unique_ptr<DeviceQPU> m_device_qpu;
public:

    GPUImplQPU();
    ~GPUImplQPU();
    size_t getQStateSize();
    QStat getQState();

    bool qubitMeasure(size_t qn);
    QError pMeasure(Qnum& qnum, prob_tuple &mResult, int select_max = -1);
    QError pMeasure(Qnum& qnum, prob_vec &mResult);
    QError initState(size_t head_rank, size_t rank_size, size_t qubit_num);
    QError initState(size_t qubit_num, const QStat &stat = {});

    QError unitarySingleQubitGate(size_t qn, QStat& matrix,
        bool is_dagger, GateType type);
    QError controlunitarySingleQubitGate(size_t qn, Qnum& qnum, QStat& matrix,
        bool is_dagger, GateType type);
    QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1, QStat& matrix,
        bool is_dagger, GateType type);
    QError controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum& qnum, QStat& matrix, 
        bool is_dagger, GateType type);

    QError DiagonalGate(Qnum& qnum, QStat & matrix,
        bool is_dagger, double error_rate);
    QError controlDiagonalGate(Qnum& qnum, QStat & matrix, Qnum& controls,
        bool is_dagger, double error_rate);
    QError Reset(size_t qn);

    QError OracleGate(Qnum& qubits, QStat &matrix,
                      bool is_dagger);
    QError controlOracleGate(Qnum& qubits, const Qnum &controls,
                             QStat &matrix, bool is_dagger);

};


#endif // USE_CUDA
#endif // ! _GPU_QUANTUM_GATE_H

