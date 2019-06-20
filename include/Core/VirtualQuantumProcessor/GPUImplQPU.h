#ifndef _GPU_QUANTUM_GATE_H
#define _GPU_QUANTUM_GATE_H
#include "QPandaConfig.h"

#ifdef USE_CUDA

#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.h"

class GPUImplQPU : public QPUImpl
{
    size_t miQbitNum;
    bool mbIsInitQState;
    double* m_probgpu;
    double* m_resultgpu;
public:

    GPUImplQPU() :mbIsInitQState(false), m_probgpu(nullptr), m_resultgpu(nullptr) {}
    ~GPUImplQPU();

    GATEGPU::QState mvQuantumStat;
    GATEGPU::QState mvCPUQuantumStat;

    size_t getQStateSize();
    QStat getQState();
    QError P0(size_t qn, bool isConjugate, double error_rate);
    QError P0(size_t qn, Qnum& vControlBit, bool isConjugate, double error_rate);
    QError P1(size_t qn, bool isConjugate, double error_rate);
    QError P1(size_t qn, Qnum& vControlBit, bool isConjugate, double error_rate);
    QError Hadamard(size_t qn, bool isConjugate, double error_rate);
    QError Hadamard(size_t qn, Qnum& vControlBit, bool isConjugate, double error_rate);
    QError X(size_t qn, bool isConjugate, double error_rate);
    QError X(size_t qn, Qnum& vControlBit, bool isConjugate, double error_rate);
    QError Y(size_t qn, bool isConjugate, double error_rate);
    QError Y(size_t qn, Qnum& vControlBit, bool isConjugate, double error_rate);
    QError Z(size_t qn, bool isConjugate, double error_rate);
    QError Z(size_t qn, Qnum& vControlBit, bool isConjugate, double error_rate);
    QError T(size_t qn, bool isConjugate, double error_rate);
    QError T(size_t qn, Qnum& vControlBit, bool isConjugate, double error_rate);
    QError S(size_t qn, bool isConjugate, double error_rate);
    QError S(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QError RX_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);
    QError RX_GATE(size_t qn, double theta, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QError RY_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);
	QError RY_GATE(size_t qn, double theta, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QError RZ_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);
    QError RZ_GATE(size_t qn, double theta, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QError CNOT(size_t qn_0, size_t qn_1,
        bool isConjugate, double error_rate);
    QError CNOT(size_t qn_0, size_t qn_1, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QError CR(size_t qn_0, size_t qn_1, double theta,
        bool isConjugate, double error_rate);
    QError CR(size_t qn_0, size_t qn_1, Qnum& vControlBit, double theta,
        bool isConjugate, double error_rate);
    QError CZ(size_t qn_0, size_t qn_1, 
		bool isConjugate, double error_rate);
    QError CZ(size_t qn_0, size_t qn_1, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QError iSWAP(size_t qn_0, size_t qn_1, double theta,
		bool isConjugate, double error_rate);
    QError iSWAP(size_t qn_0, size_t qn_1, Qnum& vControlBit, double theta,
        bool isConjugate, double error_rate);
    QError iSWAP(size_t qn_0, size_t qn_1,
		bool isConjugate, double error_rate);
    QError iSWAP(size_t qn_0, size_t qn_1, Qnum& vControlBit, 
		bool isConjugate, double error_rate);
    QError SqiSWAP(size_t qn_0, size_t qn_1,
        bool isConjugate, double error_rate);
    QError SqiSWAP(size_t qn_0, size_t qn_1, Qnum& vControlBit,
        bool isConjugate, double error_rate);
    QError U1_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate);

    QError Reset(size_t qn);
    bool qubitMeasure(size_t qn);
    QError pMeasure(Qnum& qnum, std::vector<std::pair<size_t, double>> &mResult, 
		int select_max = -1);

    QError pMeasure(Qnum& qnum, std::vector<double> &mResult);
    QError initState(QuantumGateParam *);
    QError endGate(QuantumGateParam *pQuantumProParam, QPUImpl *pQGate);
    QError unitarySingleQubitGate(size_t qn, QStat& matrix,
        bool isConjugate, double error_rate, GateType type);

    QError controlunitarySingleQubitGate(size_t qn, Qnum& qnum, QStat& matrix,
        bool isConjugate, double error_rate, GateType type);
    QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1, QStat& matrix,
        bool isConjugate, double error_rate, GateType type);
    QError controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum& qnum, QStat& matrix, 
		bool isConjugate, double error_rate, GateType type);

    QError DiagonalGate(Qnum& vQubit, QStat & matrix,
        bool isConjugate, double error_rate);
    QError controlDiagonalGate(Qnum& vQubit, QStat & matrix, Qnum& vControlBit,
        bool isConjugate, double error_rate);
};

#endif // USE_CUDA
#endif // ! _GPU_QUANTUM_GATE_H

