#ifndef _GPU_STRUCT_H
#define _GPU_STRUCT_H

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/complex.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Core/QuantumCircuit/QGlobalVariable.h"


USING_QPANDA

typedef size_t gpu_qsize_t;
typedef double gpu_qstate_t;
typedef std::complex<gpu_qstate_t> gpu_qcomplex_t;
typedef std::pair<size_t, double> gpu_pair;
typedef std::vector<gpu_pair> touple_prob;
typedef std::vector<double> vec_prob;
typedef std::vector<gpu_qsize_t> Qnum;

using device_qsize_t = size_t;
const device_qsize_t kThreadDim = 1024;

using device_complex_t = thrust::complex<qstate_type>;
using device_state_t = thrust::device_vector<device_complex_t>;
using device_qubit_t = thrust::device_vector<device_qsize_t>;
using device_param_t = thrust::device_vector<qstate_type>;

using device_complex_ptr_t = thrust::device_ptr<device_complex_t>;
using device_double_ptr_t = thrust::device_ptr<qstate_type>;
using device_int_ptr_t = thrust::device_ptr<device_qsize_t>;

using host_state_t = QStat;
using host_param_t = thrust::host_vector<qstate_type>;
using host_qubit_t = std::vector<device_qsize_t>;

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



