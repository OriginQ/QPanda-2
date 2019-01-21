#ifndef GPU_GATES_WRAPPER_H
#define GPU_GATES_WRAPPER_H

#include "GPUStruct.hpp"

#include <vector>
#include <algorithm>
#include <map>
#include <time.h>

namespace GATEGPU
{
    using std::pair;
    using std::vector;

    typedef pair<size_t, double> GPUPAIR;
    typedef vector<GPUPAIR> vecprob;                          //pmeasure ∑µªÿ¿‡–Õ
    typedef vector<size_t> Qnum;
    typedef vector<size_t> vecuint;

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
    bool pMeasure(QState&, vecprob&, size_t *block, size_t m);
    bool pMeasurenew(QState&, vector<pair<size_t, double>>&, Qnum&, int);
    bool getState(QState &psi, QState &psigpu, size_t qnum);
    int  qubitmeasure(QState& psi, size_t Block, double* &resultgpu, double* &probgpu);
    bool pMeasure_no_index(QState&, vector<double> &mResult, Qnum&);
    void gpuFree(double* memory);
}

#endif // GPU_GATE_WRAPPER_H