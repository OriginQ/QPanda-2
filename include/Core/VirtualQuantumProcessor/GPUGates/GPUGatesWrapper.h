#ifndef GPU_GATES_WRAPPER_H
#define GPU_GATES_WRAPPER_H

#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.h"

#include <vector>
#include <algorithm>
#include <map>
#include <time.h>

namespace GATEGPU
{
    int devicecount();
    bool initstate(QState& psi, QState& psigpu, size_t);
    bool destroyState(QState& psi, QState& psigpu, size_t sQnum);
    bool clearState(QState& psi, QState& psigpu, size_t sQnum);
    bool Hadamard(QState& psi, size_t qn, bool isConjugate, double error_rate = 0);
    bool Hadamardnew(QState& psi, size_t qn, bool isConjugate, double error_rate = 0);
    bool controlHadamard(QState& psi, Qnum&, bool isConjugate, double error_rate = 0);

    bool X(QState& psi, size_t qn, bool isConjugate, double error_rate = 0);
    bool controlX(QState& psi, Qnum&, bool isConjugate, double error_rate = 0);
    bool Y(QState& psi, size_t qn, bool isConjugate, double error_rate = 0);
    bool controlY(QState& psi, Qnum&, bool isConjugate, double error_rate = 0);
    bool Z(QState& psi, size_t qn, bool isConjugate, double error_rate = 0);
    bool controlZ(QState& psi, Qnum&, bool isConjugate, double error_rate = 0);
    bool S(QState& psi, size_t qn, bool isConjugate, double error_rate = 0);
    bool controlS(QState& psi, Qnum&, bool isConjugate, double error_rate = 0);
    bool T(QState& psi, size_t qn, bool isConjugate, double error_rate = 0);
    bool controlT(QState& psi, Qnum&, bool isConjugate, double error_rate = 0);

    bool RX(QState& psi, size_t qn, double theta, bool isConjugate, double error_rate = 0);
    bool controlRX(QState& psi, Qnum&, double theta, bool isConjugate, double error_rate = 0);
    bool RY(QState& psi, size_t qn, double theta, bool isConjugate, double error_rate = 0);
    bool controlRY(QState& psi, Qnum&, double theta, bool isConjugate, double error_rate = 0);
    bool RZ(QState& psi, size_t qn, double theta, bool isConjugate, double error_rate = 0);
    bool controlRZ(QState& psi, Qnum&, double theta, bool isConjugate, double error_rate = 0);

    bool CNOT(QState& psi, size_t qn_0, size_t qn_1, bool isConjugate, double error_rate = 0);
    bool controlCNOT(QState& psi, size_t qn_0, size_t qn_1, Qnum&, bool isConjugate, double error_rate = 0);
    bool CZ(QState& psi, size_t qn_0, size_t qn_1, bool isConjugate, double error_rate = 0);
    bool controlCZ(QState& psi, size_t qn_0, size_t qn_1, Qnum&, bool isConjugate, double error_rate = 0);
    bool CR(QState& psi, size_t qn_0, size_t qn_1, double thete, bool isConjugate, double error_rate = 0);
    bool controlCR(QState& psi, size_t qn_0, size_t qn_1, Qnum&, double theta, bool isConjugate, double error_rate = 0);
    bool iSWAP(QState& psi, size_t qn_0, size_t qn_1, double thete, bool isConjugate, double error_rate = 0);
    bool controliSWAP(QState& psi, size_t qn_0, size_t qn_1, Qnum&, double theta, bool isConjugate, double error_rate = 0);

    bool unitarysingle(QState& psi, size_t qn, QState& matr, bool isConjugate, double error_rate = 0);
    bool controlunitarysingle(QState& psi, Qnum&, QState& matr, bool isConjugate, double error_rate = 0);
    bool unitarydouble(QState& psi, size_t qn_0, size_t qn_1, QState& matr, bool isConjugate, double error_rate = 0);
    bool controlunitarydouble(QState& psi, Qnum&, QState& matr, bool isConjugate, double error_rate = 0);

    bool qbReset(QState& psi, size_t, double error_rate = 0);
    bool pMeasurenew(QState&, touple_prob&, Qnum&, int);
    bool getState(QState &psi, QState &psigpu, size_t qnum);
    int  qubitmeasure(QState& psigpu, gpu_qsize_t Block, gpu_qstate_t *resultgpu, gpu_qstate_t *probgpu);
    bool pMeasure_no_index(QState&, vec_prob &mResult, Qnum&);
    void gpuFree(void* memory);
}

#endif // GPU_GATE_WRAPPER_H
