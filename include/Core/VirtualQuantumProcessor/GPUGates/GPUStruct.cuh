#ifndef _GPU_STRUCT_H
#define _GPU_STRUCT_H

#include <map>
#include <cuda.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_device_runtime_api.h>
#include <thrust/transform_reduce.h>
#include <device_launch_parameters.h>
#include "Core/Utilities/Tools/Macro.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

USING_QPANDA

using device_qsize_t = size_t;
const device_qsize_t kThreadDim = 1024;

typedef size_t gpu_qsize_t;
typedef double gpu_qstate_t;
typedef std::vector<double> vec_prob;
typedef std::vector<gpu_qsize_t> Qnum;
typedef std::pair<size_t, double> gpu_pair;
typedef std::vector<gpu_pair> touple_prob;
typedef std::complex<gpu_qstate_t> gpu_qcomplex_t;

using host_state_t = QStat;
using host_qubit_t = std::vector<device_qsize_t>;
using host_param_t = thrust::host_vector<qstate_type>;

using device_complex_t = thrust::complex<qstate_type>;
using device_param_t = thrust::device_vector<qstate_type>;
using device_qubit_t = thrust::device_vector<device_qsize_t>;
using device_state_t = thrust::device_vector<device_complex_t>;

using device_int_ptr_t = thrust::device_ptr<device_qsize_t>;
using device_double_ptr_t = thrust::device_ptr<qstate_type>;
using device_complex_ptr_t = thrust::device_ptr<device_complex_t>;

struct Complex
{
    gpu_qstate_t imag;
    gpu_qstate_t real;
};

namespace GATEGPU
{
    struct probability
    {
        int state;
        gpu_qstate_t prob;
    };

    struct QState
    {
        gpu_qsize_t qnum;
        gpu_qstate_t* real;
        gpu_qstate_t* imag;
        QState() : real(nullptr), imag(nullptr) {}
    };
}

namespace QCuda
{
    struct device_status
    {
        int m_device = { 0 };
        size_t free_size = { 0 };
        size_t total_size = { 0 };
    };

    class device_data_ptr
    {
    public:
        size_t data_count = { 0 };
        size_t data_start = { 0 };
        device_complex_t* data_vector = { nullptr };
        device_data_ptr* next_data_ptr = { nullptr };

        device_data_ptr();
        ~device_data_ptr();
        __device__ device_complex_t& operator[](size_t id);
    };

    struct device_data
    {
        int device_id;
        size_t data_count;
        size_t data_start;
        cudaStream_t cuda_stream;
        device_state_t data_vector;
        device_state_t m_device_matrix;
        device_qubit_t m_device_qubits;
        thrust::device_vector<QCuda::device_data_ptr> device_data_ptr;
    };
}
#endif