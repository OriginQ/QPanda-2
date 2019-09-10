#ifndef _GPU_STRUCT_H
#define _GPU_STRUCT_H

#include "Core/QuantumCircuit/QGlobalVariable.h"

typedef size_t gpu_qsize_t;
typedef float gpu_qstate_t;
typedef std::complex<gpu_qstate_t> gpu_qcomplex_t;
typedef std::pair<size_t, double> gpu_pair;
typedef std::vector<gpu_pair> touple_prob;
typedef std::vector<double> vec_prob;
typedef std::vector<gpu_qsize_t> Qnum;

const gpu_qsize_t kThreadDim = 1024;

namespace GATEGPU
{
    struct probability
    {
        gpu_qstate_t prob;
        int state;
    };

    struct QState
    {
        QState() : real(nullptr), imag(nullptr) {}
        gpu_qstate_t * real;
        gpu_qstate_t * imag;
        gpu_qsize_t qnum;
    };
}

struct Complex
{
    gpu_qstate_t real;
    gpu_qstate_t imag;
};
#endif // !_GPU_STRUCT_H



