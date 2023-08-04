#include <omp.h>
#include <algorithm>
#include <thrust/transform_reduce.h>
#include "Core/Utilities/Tools/Utils.h"
#include "Core/VirtualQuantumProcessor/GPUGates/FunGates.cuh"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUGatesWrapper.cuh"

__constant__ qstate_type kSqrt2 = 0.707106781186547524400844362104849039;

template <typename FunGate>
__global__ void exec_gate_kernel(FunGate fun, size_t size, size_t thread_start, size_t thread_count)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < thread_count && idx + thread_start < size)
    {
        fun[fun.insert(idx + thread_start)];
    }
}

template __global__ void exec_gate_kernel<XFun>(XFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<HFun>(HFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<YFun>(YFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<ZFun>(ZFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<SFun>(SFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<PFun>(PFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<U1Fun>(U1Fun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<RZFun>(RZFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<SingleGateFun>(SingleGateFun, size_t, size_t, size_t);

template __global__ void exec_gate_kernel<CZFun>(CZFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<CRFun>(CRFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<CPFun>(CPFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<CUFun>(CUFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<CNOTFun>(CNOTFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<SWAPFun>(SWAPFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<ISWAPFun>(ISWAPFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<ORACLEFun>(ORACLEFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<CORACLEFun>(CORACLEFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<ISWAPThetaFun>(ISWAPThetaFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel<DoubleGateFun>(DoubleGateFun, size_t, size_t, size_t);

template <typename FunGate>
__global__ void exec_gate_kernel_multi(FunGate fun, size_t size, size_t thread_start, size_t thread_count)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < thread_count && idx + thread_start < size)
    {
        fun(fun.insert(idx + thread_start));
    }
}

template __global__ void exec_gate_kernel_multi<XFun>(XFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<HFun>(HFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<YFun>(YFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<ZFun>(ZFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<SFun>(SFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<U1Fun>(U1Fun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<PFun>(PFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<RZFun>(RZFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<SingleGateFun>(SingleGateFun, size_t, size_t, size_t);

template __global__ void exec_gate_kernel_multi<CZFun>(CZFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<CRFun>(CRFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<CPFun>(CPFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<CUFun>(CUFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<CNOTFun>(CNOTFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<SWAPFun>(SWAPFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<ISWAPFun>(ISWAPFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<ORACLEFun>(ORACLEFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<CORACLEFun>(CORACLEFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<ISWAPThetaFun>(ISWAPThetaFun, size_t, size_t, size_t);
template __global__ void exec_gate_kernel_multi<DoubleGateFun>(DoubleGateFun, size_t, size_t, size_t);

namespace QCuda
{
    device_data_ptr::device_data_ptr() { PRINT_DEBUG_MESSAGE }

    device_data_ptr::~device_data_ptr() { PRINT_DEBUG_MESSAGE }

    __device__ device_complex_t& device_data_ptr::operator[](size_t id)
    {
        PRINT_CUDA_DEBUG_MESSAGE
        if (id < data_count)
        {
            return data_vector[id];
        }
        return next_data_ptr->operator[](id - data_count);
    };
}

BaseGateFun::BaseGateFun() { PRINT_DEBUG_MESSAGE }

BaseGateFun::~BaseGateFun(){ PRINT_DEBUG_MESSAGE }

void BaseGateFun::set_ptr(device_complex_t* data_ptr, device_qubit_t& device_qubits, device_state_t& device_matrix)
{
    PRINT_DEBUG_MESSAGE
    data_vector = data_ptr;
    m_device_matrix = thrust::raw_pointer_cast(device_matrix.data());
    m_device_opt_qubits = thrust::raw_pointer_cast(device_qubits.data());
}

void BaseGateFun::set_ptr(QCuda::device_data_ptr* data_ptr, device_qubit_t& device_qubits, device_state_t& device_matrix)
{
    PRINT_DEBUG_MESSAGE
    device_data_ptr = data_ptr;
    m_device_matrix = thrust::raw_pointer_cast(device_matrix.data());
    m_device_opt_qubits = thrust::raw_pointer_cast(device_qubits.data());
}

void BaseGateFun::set_qubit_num(device_qsize_t qubit_num)
{
    PRINT_DEBUG_MESSAGE
    m_qubit_num = qubit_num;
}

void SingleGateFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            qcomplex_t temp;
        temp = matrix[1];
        matrix[1] = matrix[2];
        matrix[2] = temp;
        for (size_t i = 0; i < 4; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

void SingleGateFun::set_qubits(const host_qubit_t& qubits, device_qsize_t opt_num, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_cmask = 0;
    m_opt_num = opt_num;
    m_offset0 = 1ll << qubits.back();
    CHECK_CUDA(cudaMemcpyAsync(m_device_opt_qubits, &qubits[qubits.size() - 1], sizeof(device_qsize_t), cudaMemcpyHostToDevice, stream));

    std::for_each(qubits.begin(), qubits.end() - 1, [&](device_qsize_t q)
        { m_cmask |= 1ll << q; });
}

__device__ size_t SingleGateFun::insert(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    int64_t number = 1ll << m_device_opt_qubits[0];
    if (i < number)
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return i;
    }
    int64_t mask = number - 1;
    int64_t x = mask & i;
    int64_t y = ~mask & i;
    return ((y << 1) | x);
}

__device__ double SingleGateFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }

    size_t i1 = i | m_offset0;
    device_complex_t alpha = device_data_ptr->operator[](i);
    device_complex_t beta = device_data_ptr->operator[](i1);
    device_data_ptr->operator[](i) = m_device_matrix[0] * alpha + m_device_matrix[1] * beta;
    device_data_ptr->operator[](i1) = m_device_matrix[2] * alpha + m_device_matrix[3] * beta;
    return 0.0;
}

__device__ double SingleGateFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }

    size_t i1 = i | m_offset0;
    device_complex_t alpha = data_vector[i];
    device_complex_t beta = data_vector[i1];
    data_vector[i] = m_device_matrix[0] * alpha + m_device_matrix[1] * beta;
    data_vector[i1] = m_device_matrix[2] * alpha + m_device_matrix[3] * beta;
    return 0.0;
}

void XFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double XFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    thrust::swap(device_data_ptr->operator[](i), device_data_ptr->operator[](i | m_offset0));
    return 0.0;
}

__device__ double XFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    thrust::swap(data_vector[i], data_vector[i | m_offset0]);
    return 0.0;
}

void YFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double YFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    size_t i1 = i | m_offset0;
    device_complex_t alpha = device_data_ptr->operator[](i);
    device_complex_t beta = device_data_ptr->operator[](i1);
    device_data_ptr->operator[](i) = device_complex_t(beta.imag(), -beta.real());
    device_data_ptr->operator[](i1) = device_complex_t(-alpha.imag(), alpha.real());
    return 0.0;
}

__device__ double YFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    size_t i1 = i | m_offset0;
    device_complex_t alpha = data_vector[i];
    device_complex_t beta = data_vector[i1];
    data_vector[i] = device_complex_t(beta.imag(), -beta.real());
    data_vector[i1] = device_complex_t(-alpha.imag(), alpha.real());
    return 0.0;
}

void ZFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double ZFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_data_ptr->operator[](i | m_offset0) *= -1;
    return 0.0;
}

__device__ double ZFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    data_vector[i | m_offset0] *= -1;
    return 0.0;
}

void SFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double SFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    size_t i1 = i | m_offset0;
    if (m_is_dagger)
    {
        PRINT_CUDA_DEBUG_MESSAGE
            device_data_ptr->operator[](i1) = device_complex_t(device_data_ptr->operator[](i1).imag(), -device_data_ptr->operator[](i1).real());
    }
    else
    {
        PRINT_CUDA_DEBUG_MESSAGE
            device_data_ptr->operator[](i1) = device_complex_t(-device_data_ptr->operator[](i1).imag(), device_data_ptr->operator[](i1).real());
    }
    return 0.0;
}

__device__ double SFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    size_t i1 = i | m_offset0;
    if (m_is_dagger)
    {
        PRINT_CUDA_DEBUG_MESSAGE
            data_vector[i1] = device_complex_t(data_vector[i1].imag(), -data_vector[i1].real());
    }
    else
    {
        PRINT_CUDA_DEBUG_MESSAGE
            data_vector[i1] = device_complex_t(-data_vector[i1].imag(), data_vector[i1].real());
    }
    return 0.0;
}

void HFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double HFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    size_t i1 = i | m_offset0;
    device_complex_t alpha = device_data_ptr->operator[](i);
    device_complex_t beta = device_data_ptr->operator[](i1);
    device_data_ptr->operator[](i) = (alpha + beta) * kSqrt2;
    device_data_ptr->operator[](i1) = (alpha - beta) * kSqrt2;
    return 0.0;
}

__device__ double HFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    size_t i1 = i | m_offset0;
    device_complex_t alpha = data_vector[i];
    device_complex_t beta = data_vector[i1];
    data_vector[i] = (alpha + beta) * kSqrt2;
    data_vector[i1] = (alpha - beta) * kSqrt2;
    return 0.0;
}

void U1Fun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double U1Fun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_data_ptr->operator[](i | m_offset0) *= m_device_matrix[3];
    return 0.0;
}

__device__ double U1Fun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    data_vector[i | m_offset0] *= m_device_matrix[3];
    return 0.0;
}

void PFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double PFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_data_ptr->operator[](i | m_offset0) *= m_device_matrix[3];
    return 0.0;
}

__device__ double PFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    data_vector[i | m_offset0] *= m_device_matrix[3];
    return 0.0;
}

void RZFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            matrix[0] = qcomplex_t(matrix[0].real(), -matrix[0].imag());
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double RZFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_data_ptr->operator[](i) *= m_device_matrix[0];
    device_data_ptr->operator[](i | m_offset0) *= m_device_matrix[3];
    return 0.0;
}

__device__ double RZFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    data_vector[i] *= m_device_matrix[0];
    data_vector[i | m_offset0] *= m_device_matrix[3];
    return 0.0;
}

void DoubleGateFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            qcomplex_t temp;
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = i + 1; j < 4; j++)
            {
                temp = matrix[4 * i + j];
                matrix[4 * i + j] = matrix[4 * j + i];
                matrix[4 * j + i] = temp;
            }
        }
        for (size_t i = 0; i < 16; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

void DoubleGateFun::set_qubits(const host_qubit_t& qubits, device_qsize_t opt_num, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_cmask = 0;
    m_opt_num = opt_num;
    m_offset0 = 1ll << *(qubits.end() - 2);
    m_offset1 = 1ll << *(qubits.end() - 1);
    cudaMemcpyAsync(m_device_opt_qubits, &qubits[qubits.size() - 2], 2 * sizeof(device_qsize_t), cudaMemcpyHostToDevice, stream);

    std::for_each(qubits.begin(), qubits.end() - 2, [&](device_qsize_t q)
        { m_cmask |= 1ll << q; });
}

__device__ size_t DoubleGateFun::insert(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t n1 = m_device_opt_qubits[0];
    size_t n2 = m_device_opt_qubits[1];
    if (n1 > n2)
    {
        PRINT_CUDA_DEBUG_MESSAGE
            thrust::swap(n1, n2);
    }
    int64_t mask1 = (1ll << n1) - 1;
    int64_t mask2 = (1ll << (n2 - 1)) - 1;
    int64_t z = i & mask1;
    int64_t y = ~mask1 & i & mask2;
    int64_t x = ~mask2 & i;
    return ((x << 2) | (y << 1) | z);
}

__device__ double DoubleGateFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi00 = device_data_ptr->operator[](i);
    device_complex_t phi01 = device_data_ptr->operator[](i | m_offset0);
    device_complex_t phi10 = device_data_ptr->operator[](i | m_offset1);
    device_complex_t phi11 = device_data_ptr->operator[](i | m_offset0 | m_offset1);

    device_data_ptr->operator[](i) = m_device_matrix[0] * phi00 + m_device_matrix[1] * phi01 + m_device_matrix[2] * phi10 + m_device_matrix[3] * phi11;
    device_data_ptr->operator[](i | m_offset0) = m_device_matrix[4] * phi00 + m_device_matrix[5] * phi01 + m_device_matrix[6] * phi10 + m_device_matrix[7] * phi11;
    device_data_ptr->operator[](i | m_offset1) = m_device_matrix[8] * phi00 + m_device_matrix[9] * phi01 + m_device_matrix[10] * phi10 + m_device_matrix[11] * phi11;
    device_data_ptr->operator[](i | m_offset0 | m_offset1) = m_device_matrix[12] * phi00 + m_device_matrix[13] * phi01 + m_device_matrix[14] * phi10 + m_device_matrix[15] * phi11;
    return 0.0;
}

__device__ double DoubleGateFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi00 = data_vector[i];
    device_complex_t phi01 = data_vector[i | m_offset0];
    device_complex_t phi10 = data_vector[i | m_offset1];
    device_complex_t phi11 = data_vector[i | m_offset0 | m_offset1];

    data_vector[i] = m_device_matrix[0] * phi00 + m_device_matrix[1] * phi01 + m_device_matrix[2] * phi10 + m_device_matrix[3] * phi11;
    data_vector[i | m_offset0] = m_device_matrix[4] * phi00 + m_device_matrix[5] * phi01 + m_device_matrix[6] * phi10 + m_device_matrix[7] * phi11;
    data_vector[i | m_offset1] = m_device_matrix[8] * phi00 + m_device_matrix[9] * phi01 + m_device_matrix[10] * phi10 + m_device_matrix[11] * phi11;
    data_vector[i | m_offset0 | m_offset1] = m_device_matrix[12] * phi00 + m_device_matrix[13] * phi01 + m_device_matrix[14] * phi10 + m_device_matrix[15] * phi11;
    return 0.0;
}

void CNOTFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double CNOTFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    thrust::swap(device_data_ptr->operator[](i | m_offset0), device_data_ptr->operator[](i | m_offset0 | m_offset1));
    return 0.0;
}

__device__ double CNOTFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    thrust::swap(data_vector[i | m_offset0], data_vector[i | m_offset0 | m_offset1]);
    return 0.0;
}

void CZFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double CZFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_data_ptr->operator[](i | m_offset0 | m_offset1) = -device_data_ptr->operator[](i | m_offset0 | m_offset1);
    return 0.0;
}

__device__ double CZFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    data_vector[i | m_offset0 | m_offset1] = -data_vector[i | m_offset0 | m_offset1];
    return 0.0;
}

void CRFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double CRFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_data_ptr->operator[](i | m_offset0 | m_offset1) *= m_device_matrix[15];
    return 0.0;
}

__device__ double CRFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    data_vector[i | m_offset0 | m_offset1] *= m_device_matrix[15];
    return 0.0;
}

void CPFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double CPFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_data_ptr->operator[](i | m_offset0 | m_offset1) *= m_device_matrix[15];
    return 0.0;
}

__device__ double CPFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    data_vector[i | m_offset0 | m_offset1] *= m_device_matrix[15];
    return 0.0;
}

void SWAPFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_is_dagger = is_dagger;
}

__device__ double SWAPFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    thrust::swap(device_data_ptr->operator[](i | m_offset1), device_data_ptr->operator[](i | m_offset0));
    return 0.0;
}

__device__ double SWAPFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    thrust::swap(data_vector[i | m_offset1], data_vector[i | m_offset0]);
    return 0.0;
}

void ISWAPFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            matrix[6] = { 0, 1 };
        matrix[9] = { 0, 1 };
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double ISWAPFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi01 = device_data_ptr->operator[](i | m_offset1);
    device_complex_t phi10 = device_data_ptr->operator[](i | m_offset0);
    device_data_ptr->operator[](i | m_offset1) = m_device_matrix[6] * phi10;
    device_data_ptr->operator[](i | m_offset0) = m_device_matrix[9] * phi01;
    return 0.0;
}

__device__ double ISWAPFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi01 = data_vector[i | m_offset1];
    device_complex_t phi10 = data_vector[i | m_offset0];
    data_vector[i | m_offset1] = m_device_matrix[6] * phi10;
    data_vector[i | m_offset0] = m_device_matrix[9] * phi01;
    return 0.0;
}

void ISWAPThetaFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            matrix[6] = { matrix[6].real(), -matrix[6].imag() };
        matrix[9] = { matrix[9].real(), -matrix[9].imag() };
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double ISWAPThetaFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi01 = device_data_ptr->operator[](i | m_offset1);
    device_complex_t phi10 = device_data_ptr->operator[](i | m_offset0);
    device_data_ptr->operator[](i | m_offset1) = m_device_matrix[5] * phi01 + m_device_matrix[6] * phi10;
    device_data_ptr->operator[](i | m_offset0) = m_device_matrix[9] * phi01 + m_device_matrix[10] * phi10;
    return 0.0;
}

__device__ double ISWAPThetaFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi01 = data_vector[i | m_offset1];
    device_complex_t phi10 = data_vector[i | m_offset0];
    data_vector[i | m_offset1] = m_device_matrix[5] * phi01 + m_device_matrix[6] * phi10;
    data_vector[i | m_offset0] = m_device_matrix[9] * phi01 + m_device_matrix[10] * phi10;
    return 0.0;
}

void CUFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
            auto tmp = matrix[11];
        matrix[10] = { matrix[10].real(), -matrix[10].imag() };
        matrix[11] = { matrix[14].real(), -matrix[14].imag() };
        matrix[14] = { tmp.real(), -tmp.imag() };
        matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ double CUFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi10 = device_data_ptr->operator[](i | m_offset0);
    device_complex_t phi11 = device_data_ptr->operator[](i | m_offset0 | m_offset1);
    device_data_ptr->operator[](i | m_offset0) = m_device_matrix[10] * phi10 + m_device_matrix[11] * phi11;
    device_data_ptr->operator[](i | m_offset0 | m_offset1) = m_device_matrix[14] * phi10 + m_device_matrix[15] * phi11;
    return 0.0;
}

__device__ double CUFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    device_complex_t phi10 = data_vector[i | m_offset0];
    device_complex_t phi11 = data_vector[i | m_offset0 | m_offset1];
    data_vector[i | m_offset0] = m_device_matrix[10] * phi10 + m_device_matrix[11] * phi11;
    data_vector[i | m_offset0 | m_offset1] = m_device_matrix[14] * phi10 + m_device_matrix[15] * phi11;
    return 0.0;
}

__device__ double MeasureFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t real_idx = insert(i);
    if (data_vector)
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return data_vector[real_idx].real() * data_vector[real_idx].real() + data_vector[real_idx].imag() * data_vector[real_idx].imag();
    }
    return device_data_ptr->operator[](real_idx).real() * device_data_ptr->operator[](real_idx).real() + device_data_ptr->operator[](real_idx).imag() * device_data_ptr->operator[](real_idx).imag();
}

NormlizeFun::NormlizeFun(double prob, bool measure_out) : SingleGateFun(), m_prob(prob), m_measure_out(measure_out){ PRINT_DEBUG_MESSAGE };

NormlizeFun::NormlizeFun() : SingleGateFun(), m_prob(0), m_measure_out(false){ PRINT_DEBUG_MESSAGE }

void NormlizeFun::set_measure_out(double prob, bool measure_out)
{
    PRINT_DEBUG_MESSAGE
    m_prob = prob;
    m_measure_out = measure_out;
}

__device__ double NormlizeFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t real_idx = insert(i);
    if (data_vector)
    {
        PRINT_CUDA_DEBUG_MESSAGE
            if (m_measure_out)
            {
                PRINT_CUDA_DEBUG_MESSAGE
                    data_vector[real_idx] = 0;
                data_vector[real_idx | m_offset0] *= m_prob;
            }
            else
            {
                PRINT_CUDA_DEBUG_MESSAGE
                    data_vector[real_idx] *= m_prob;
                data_vector[real_idx | m_offset0] = 0;
            }
    }
    else
    {
        PRINT_CUDA_DEBUG_MESSAGE
            if (m_measure_out)
            {
                PRINT_CUDA_DEBUG_MESSAGE
                    device_data_ptr->operator[](real_idx) = 0;
                device_data_ptr->operator[](real_idx | m_offset0) *= m_prob;
            }
            else
            {
                PRINT_CUDA_DEBUG_MESSAGE
                    device_data_ptr->operator[](real_idx) *= m_prob;
                device_data_ptr->operator[](real_idx | m_offset0) = 0;
            }
    }
    return 0.0;
}

double exec_measure(MeasureFun& fun, size_t size, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    auto iter = thrust::counting_iterator<size_t>(0);
    return thrust::transform_reduce(thrust::cuda::par.on(stream), iter, iter + size, fun, 0.0, thrust::plus<double>());
}

void exec_normalize(NormlizeFun& fun, size_t size, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    auto iter = thrust::counting_iterator<size_t>(0);
    thrust::for_each(thrust::cuda::par.on(stream), iter, iter + size, fun);
    PRINT_DEBUG_MESSAGE
}

void ORACLEFun::set_qubits(const host_qubit_t& qubits, device_qsize_t opt_num, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_opt_num = opt_num;
    cudaMemcpyAsync(m_device_opt_qubits, &qubits[0], m_opt_num * sizeof(device_qsize_t), cudaMemcpyHostToDevice, stream);
}

void ORACLEFun::set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    if (is_dagger)
    {
        PRINT_DEBUG_MESSAGE
        qcomplex_t temp;
        size_t line = (size_t)powf(2, m_opt_num);
        size_t count = line * line;
        for (size_t i = 0; i < line; i++)
        {
            for (size_t j = i + 1; j < line; j++)
            {
                temp = matrix[line * i + j];
                matrix[line * i + j] = matrix[line * j + i];
                matrix[line * j + i] = temp;
            }
        }
        for (size_t i = 0; i < count; i++)
        {
            matrix[i] = qcomplex_t(matrix[i].real(), -matrix[i].imag());
        }
    }
    m_is_dagger = is_dagger;
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size() * sizeof(device_complex_t), cudaMemcpyHostToDevice, stream);
}

__device__ size_t ORACLEFun::insert(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t ret=((1 << (m_device_opt_qubits[0])) - 1)&i;
    for (size_t id = 1; id < m_opt_num; id++)
    {
        ret += (~((1 << (m_device_opt_qubits[id - 1] - id + 1)) - 1) & i & ((1 << (m_device_opt_qubits[id] - id)) - 1)) << id;
    }
    ret+= (~((1 << (m_device_opt_qubits[m_opt_num-1] - m_opt_num + 1)) - 1) & i) << m_opt_num;
    return ret;
}

__device__ double ORACLEFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t realxx_idxes[1024];
    device_complex_t state_bak[1024];
    size_t dim = (size_t)powf(2, m_opt_num);
    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        size_t realxx_idx = i;
        size_t tmp = i_dim;
        for (size_t qubit_idx = 0; qubit_idx < m_opt_num; qubit_idx++)
        {
            tmp = i_dim >> qubit_idx;
            if (tmp > 0)
            {
                if (1ull & tmp)
                {
                    realxx_idx += 1ll << m_device_opt_qubits[qubit_idx];
                }

            }
            else
            {
                break;
            }
        }
        realxx_idxes[i_dim] = realxx_idx;
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        state_bak[i_dim] = data_vector[realxx_idxes[i_dim]];
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        device_data_ptr->operator[](realxx_idxes[i_dim]) = 0;
        for (size_t m_dim = 0; m_dim < dim; m_dim++)
        {
            device_data_ptr->operator[](realxx_idxes[i_dim]) += m_device_matrix[i_dim * dim + m_dim] * state_bak[m_dim];
        }
    }
    return 0.0;
}

__device__ double ORACLEFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    size_t realxx_idxes[1024];
    device_complex_t state_bak[1024];
    size_t dim = (size_t)powf(2, m_opt_num);
    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        size_t realxx_idx = i;
        size_t tmp = i_dim;
        for (size_t qubit_idx = 0; qubit_idx < m_opt_num; qubit_idx++)
        {
            tmp = i_dim >> qubit_idx;
            if (tmp > 0)
            {
                if (1ull & tmp)
                {
                    realxx_idx += 1ll << m_device_opt_qubits[qubit_idx];
                }

            }
            else
            {
                break;
            }
        }
        realxx_idxes[i_dim] = realxx_idx;
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        state_bak[i_dim] = data_vector[realxx_idxes[i_dim]];
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        data_vector[realxx_idxes[i_dim]] = 0;
        for (size_t m_dim = 0; m_dim < dim; m_dim++)
        {
            data_vector[realxx_idxes[i_dim]] += m_device_matrix[i_dim * dim + m_dim] * state_bak[m_dim];
        }
    }
    return 0.0;
}

void CORACLEFun::set_qubits(const host_qubit_t& qubits, const host_qubit_t& control, cudaStream_t& stream)
{
    PRINT_DEBUG_MESSAGE
    m_cmask = 0;
    m_opt_num = qubits.size();
    cudaMemcpyAsync(m_device_opt_qubits, &qubits[0], m_opt_num * sizeof(device_qsize_t), cudaMemcpyHostToDevice, stream);
    std::for_each(control.begin(), control.end() - m_opt_num, [&](device_qsize_t q)
        { m_cmask |= 1ll << q; });
}

__device__ double CORACLEFun::operator()(size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
            return 0.0;
    }
    size_t realxx_idxes[1024];
    device_complex_t state_bak[1024];
    size_t dim = (size_t)powf(2, m_opt_num);
    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        size_t realxx_idx = i;
        size_t tmp = i_dim;
        for (size_t qubit_idx = 0; qubit_idx < m_opt_num; qubit_idx++)
        {
            tmp = i_dim >> qubit_idx;
            if (tmp > 0)
            {
                if (1ull & tmp)
                {
                    realxx_idx += 1ll << m_device_opt_qubits[qubit_idx];
                }

            }
            else
            {
                break;
            }
        }
        realxx_idxes[i_dim] = realxx_idx;
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        state_bak[i_dim] = data_vector[realxx_idxes[i_dim]];
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        device_data_ptr->operator[](realxx_idxes[i_dim]) = 0;
        for (size_t m_dim = 0; m_dim < dim; m_dim++)
        {
            device_data_ptr->operator[](realxx_idxes[i_dim]) += m_device_matrix[i_dim * dim + m_dim] * state_bak[m_dim];
        }
    }
    return 0.0;
}

__device__ double CORACLEFun::operator[](size_t i)
{
    PRINT_CUDA_DEBUG_MESSAGE
    if (m_cmask != (m_cmask & i))
    {
        PRINT_CUDA_DEBUG_MESSAGE
        return 0.0;
    }
    size_t realxx_idxes[1024];
    device_complex_t state_bak[1024];
    size_t dim = (size_t)powf(2, m_opt_num);
    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        size_t realxx_idx = i;
        size_t tmp = i_dim;
        for (size_t qubit_idx = 0; qubit_idx < m_opt_num; qubit_idx++)
        {
            tmp = i_dim >> qubit_idx;
            if (tmp > 0)
            {
                if (1ull & tmp)
                {
                    realxx_idx += 1ll << m_device_opt_qubits[qubit_idx];
                }

            }
            else
            {
                break;
            }
        }
        realxx_idxes[i_dim] = realxx_idx;
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        state_bak[i_dim]= data_vector[realxx_idxes[i_dim]];
    }

    for (size_t i_dim = 0; i_dim < dim; i_dim++)
    {
        data_vector[realxx_idxes[i_dim]] = 0;
        for (size_t m_dim = 0; m_dim < dim; m_dim++)
        {
            data_vector[realxx_idxes[i_dim]] += m_device_matrix[i_dim * dim + m_dim] * state_bak[m_dim];
        }
    }
    return 0.0;
}