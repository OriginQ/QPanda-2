#include "Core/VirtualQuantumProcessor/GPUGates/FunGates.cuh"
#include <algorithm>
#include <thrust/transform_reduce.h>
#include <omp.h>
#include <algorithm>
#include "Core/Utilities/Tools/Utils.h"



__constant__ qstate_type kSqrt2 = 1 / 1.4142135623731;

template <typename FunGate> __global__
void exec_gate_kernel(FunGate fun, int64_t size)
{
    int64_t idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < size)
    {
        int64_t real_idx = fun.insert(idx);
        fun(real_idx);
    }

    return ;
}

template __global__
void exec_gate_kernel<XFun>(XFun, int64_t);
template __global__
void exec_gate_kernel<HFun>(HFun, int64_t);
template __global__
void exec_gate_kernel<YFun>(YFun, int64_t);
template __global__
void  exec_gate_kernel<ZFun>(ZFun, int64_t);
template __global__
void exec_gate_kernel<SFun>(SFun, int64_t);
template __global__
void exec_gate_kernel<U1Fun>(U1Fun, int64_t);
template __global__
void exec_gate_kernel<PFun>(PFun, int64_t);
template __global__
void exec_gate_kernel<RZFun>(RZFun, int64_t);
template __global__
void exec_gate_kernel<SingleGateFun>(SingleGateFun, int64_t);


template __global__
void exec_gate_kernel<CNOTFun>(CNOTFun, int64_t);
template __global__
void exec_gate_kernel<CZFun>(CZFun, int64_t);
template __global__
void exec_gate_kernel<CRFun>(CRFun, int64_t);
template __global__
void exec_gate_kernel<CPFun>(CPFun, int64_t);
template __global__
void exec_gate_kernel<SWAPFun>(SWAPFun, int64_t);
template __global__
void exec_gate_kernel<ISWAPFun>(ISWAPFun, int64_t);
template __global__
void exec_gate_kernel<ISWAPThetaFun>(ISWAPThetaFun, int64_t);
template __global__
void exec_gate_kernel<CUFun>(CUFun, int64_t);
template __global__
void exec_gate_kernel<DoubleGateFun>(DoubleGateFun, int64_t);



BaseGateFun::BaseGateFun()
{

}


void BaseGateFun::set_state(device_state_t &state)
{
    m_state = thrust::raw_pointer_cast(state.data());
}


void BaseGateFun::set_qubit_num(device_qsize_t qubit_num)
{
    m_qubit_num = qubit_num;
}

void BaseGateFun::set_device_prams(device_qubit_t &device_qubits,
                                   device_state_t &device_matrix)
{
    m_device_opt_qubits = thrust::raw_pointer_cast(device_qubits.data());
    m_device_matrix = thrust::raw_pointer_cast(device_matrix.data());
}

BaseGateFun::~BaseGateFun()
{
}


void SingleGateFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
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
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

void SingleGateFun::set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t stream)
{
    m_offset0 = 1ll << qubits.back();
//    thrust::copy_n(qubits.end() - 1, 1, m_opt_qubits.begin());
    cudaError status = cudaMemcpyAsync(m_device_opt_qubits, &qubits[qubits.size() - 1], sizeof(device_qsize_t),
                                 cudaMemcpyHostToDevice, stream);
    QPANDA_ASSERT(cudaSuccess != status, "Error: cudaMemcpyAsync\n");
    m_opt_num = opt_num;

    m_cmask = 0;
    std::for_each(qubits.begin(), qubits.end() - 1, [&](device_qsize_t q){
           m_cmask |= 1ll << q;
       });
}


__device__ int64_t SingleGateFun::insert(int64_t i)
{
    int64_t number = 1ll << m_device_opt_qubits[0];
    if (i < number)
    {
        return i;
    }

    int64_t mask = number - 1;
    int64_t x = mask & i;
    int64_t y = ~mask & i;
    return ((y << 1) | x);
}

__device__ double SingleGateFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    int64_t i1 = i | m_offset0;
    device_complex_t alpha = m_state[i];
    device_complex_t beta = m_state[i1];
    m_state[i] = m_device_matrix[0] * alpha + m_device_matrix[1] * beta;
    m_state[i1] = m_device_matrix[2] * alpha + m_device_matrix[3] * beta;

    return 0.0;
}


void XFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
}


__device__ double XFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    thrust::swap(m_state[i], m_state[i | m_offset0]);
    return 0.0;
}


void YFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
}


__device__ double YFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;
    int64_t i1 = i | m_offset0;
    device_complex_t alpha = m_state[i];
    device_complex_t beta = m_state[i1];
    m_state[i] = device_complex_t(beta.imag(), -beta.real());
    m_state[i1] = device_complex_t(-alpha.imag(), alpha.real());
    return 0.0;
}

void ZFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
}

__device__ double ZFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;
    int64_t i1 = i | m_offset0;
    m_state[i1] = -m_state[i1];
    return 0.0;
}

void SFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
}

__device__ double SFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    int64_t i1 = i | m_offset0;
    if (m_is_dagger)
    {
        m_state[i1] = device_complex_t(m_state[i1].imag(), -m_state[i1].real());
    }
    else
    {
        m_state[i1] = device_complex_t(-m_state[i1].imag(), m_state[i1].real());
    }
    return 0.0;
}

void HFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
}

__device__ double HFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return ;

    int64_t i1 = i | m_offset0;
    device_complex_t alpha = m_state[i];
    device_complex_t beta = m_state[i1];
    m_state[i] = (alpha + beta) * kSqrt2;
    m_state[i1] = (alpha - beta) * kSqrt2;
    return 0.0;
}


void U1Fun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

__device__ double U1Fun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    int64_t i1 = i | m_offset0;
    m_state[i1] *= m_device_matrix[3];
    return 0.0;
}

void PFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
}

__device__ double PFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    int64_t i1 = i | m_offset0;
    m_state[i1] *= m_device_matrix[3];
    return 0.0;
}

void RZFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        matrix[0] = qcomplex_t(matrix[0].real(), -matrix[0].imag());
        matrix[3] = qcomplex_t(matrix[3].real(), -matrix[3].imag());
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

__device__ double RZFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    int64_t i1 = i | m_offset0;
    m_state[i] *= m_device_matrix[0];
    m_state[i1] *= m_device_matrix[3];
    return 0.0;
}


void DoubleGateFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
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
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}


void DoubleGateFun::set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t stream)
{
    m_offset0 = 1ll << *(qubits.end() - 2);
    m_offset1 = 1ll << *(qubits.end() - 1);
    cudaMemcpyAsync(m_device_opt_qubits, &qubits[qubits.size() - 2], 2 * sizeof(device_qsize_t),
            cudaMemcpyHostToDevice, stream);
    m_opt_num = opt_num;

    m_cmask = 0;
    std::for_each(qubits.begin(), qubits.end() - 2, [&](device_qsize_t q){
        m_cmask |= 1ll << q;
    });
    return ;
}


__device__ int64_t DoubleGateFun::insert(int64_t i)
{
    size_t n1 = m_device_opt_qubits[0];
    size_t n2 = m_device_opt_qubits[1];

    if (n1 > n2)
    {
        thrust::swap(n1, n2);
    }

    int64_t mask1 = (1ll << n1) - 1;
    int64_t mask2 = (1ll << (n2 - 1)) - 1;
    int64_t z = i & mask1;
    int64_t y = ~mask1 & i & mask2;
    int64_t x = ~mask2 & i;

    return ((x << 2) | (y << 1) | z);
}

__device__ double DoubleGateFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    device_complex_t phi00 = m_state[i];
    device_complex_t phi01 = m_state[i | m_offset0];
    device_complex_t phi10 = m_state[i | m_offset1];
    device_complex_t phi11 = m_state[i | m_offset0 | m_offset1];

    m_state[i] = m_device_matrix[0] * phi00 + m_device_matrix[1] * phi01
        + m_device_matrix[2] * phi10 + m_device_matrix[3] * phi11;
    m_state[i | m_offset0] = m_device_matrix[4] * phi00 + m_device_matrix[5] * phi01
        + m_device_matrix[6] * phi10 + m_device_matrix[7] * phi11;
    m_state[i | m_offset1] = m_device_matrix[8] * phi00 + m_device_matrix[9] * phi01
        + m_device_matrix[10] * phi10 + m_device_matrix[11] * phi11;
    m_state[i | m_offset0 | m_offset1] = m_device_matrix[12] * phi00 + m_device_matrix[13] * phi01
        + m_device_matrix[14] * phi10 + m_device_matrix[15] * phi11;

    return 0.0;
}

void CNOTFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
}

__device__ double CNOTFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;
    thrust::swap(m_state[i | m_offset0], m_state[i | m_offset0 | m_offset1]);
    return 0.0;
}

void CZFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
}

__device__ double CZFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;
    m_state[i | m_offset0 | m_offset1] = -m_state[i | m_offset0 | m_offset1];
    return 0.0;
}

void CRFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

__device__ double CRFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;
    m_state[i | m_offset0 | m_offset1] *= m_device_matrix[15];
    return 0.0;
}

void CPFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

__device__ double CPFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;
    m_state[i | m_offset0 | m_offset1] *= m_device_matrix[15];
    return 0.0;
}

void SWAPFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    m_is_dagger = is_dagger;
    return ;
}

__device__ double SWAPFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return ;
    thrust::swap(m_state[i | m_offset1], m_state[i | m_offset0]);
    return ;
}

void ISWAPFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        matrix[6] = { 0, 1 };
        matrix[9] = { 0, 1 };
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

__device__ double ISWAPFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    device_complex_t phi01 = m_state[i | m_offset1];
    device_complex_t phi10 = m_state[i | m_offset0];
    m_state[i | m_offset1] = m_device_matrix[6] * phi10;
    m_state[i | m_offset0] = m_device_matrix[9] * phi01;
    return 0.0;
}

void ISWAPThetaFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        matrix[6] = { matrix[6].real(), -matrix[6].imag() };
        matrix[9] = { matrix[9].real(), -matrix[9].imag() };
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

__device__ double ISWAPThetaFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    device_complex_t phi01 = m_state[i | m_offset1];
    device_complex_t phi10 = m_state[i | m_offset0];
    m_state[i | m_offset1] = m_device_matrix[5] * phi01 + m_device_matrix[6] * phi10;
    m_state[i | m_offset0] = m_device_matrix[9] * phi01 + m_device_matrix[10] * phi10;
    return 0.0;
}


void CUFun::set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream)
{
    if (is_dagger)
    {
        auto tmp = matrix[11];
        matrix[10] = { matrix[10].real(), -matrix[10].imag() };
        matrix[11] = { matrix[14].real(), -matrix[14].imag() };
        matrix[14] = { tmp.real(), -tmp.imag() };
        matrix[15] = { matrix[15].real(), -matrix[15].imag() };
    }

    m_is_dagger = is_dagger;
    //thrust::copy_n(matrix.begin(), matrix.size(), m_matrix.begin());
    cudaMemcpyAsync(m_device_matrix, matrix.data(), matrix.size()*sizeof(device_complex_t),
            cudaMemcpyHostToDevice, stream);
    return ;
}

__device__ double CUFun::operator()(int64_t i)
{
    if (m_cmask != (m_cmask & i))
        return 0.0;

    device_complex_t phi10 = m_state[i | m_offset0];
    device_complex_t phi11 = m_state[i | m_offset0 | m_offset1];
    m_state[i | m_offset0] = m_device_matrix[10] * phi10 + m_device_matrix[11] * phi11;
    m_state[i | m_offset0 | m_offset1] = m_device_matrix[14] * phi10 + m_device_matrix[15] * phi11;
    return 0.0;
}



__device__ double MeasureFun::operator()(int64_t i)
{
    int64_t real_idx = insert(i);
    return m_state[real_idx].real()*m_state[real_idx].real() +
            m_state[real_idx].imag()*m_state[real_idx].imag();
    return 0.0;
}

NormlizeFun::NormlizeFun(double prob, bool measure_out)
    :SingleGateFun(), m_prob(prob), m_measure_out(measure_out)
{
}

NormlizeFun::NormlizeFun()
    :SingleGateFun(), m_prob(0), m_measure_out(false)
{

}

void NormlizeFun::set_measure_out(double prob, bool measure_out)
{
    m_prob = prob;
    m_measure_out = measure_out;
    return ;
}

__device__ double NormlizeFun::operator()(int64_t i)
{
    int64_t real_idx = insert(i);
    if (m_measure_out)
    {
        m_state[real_idx] = 0;
        m_state[real_idx | m_offset0] *= m_prob;
    }
    else
    {
        m_state[real_idx] *= m_prob;
        m_state[real_idx | m_offset0] = 0;
    }

    return 0.0;
}

ProbFun::ProbFun()
{
}

void ProbFun::set_state(device_complex_ptr_t state)
{
    m_state = thrust::raw_pointer_cast(state);
}

void ProbFun::set_idx(int64_t idx)
{
    m_idx = idx;
}

void ProbFun::set_qubits(const host_qubit_t &host_qubits, device_qsize_t *opt_qubits,
                         device_qsize_t opt_num, cudaStream_t stream)
{
    m_mask = 0;
    m_cmask = 0;

    for(int i = 0 ; i < opt_num; i++)
    {
        m_mask |= (1ull << host_qubits[i]);
        if(((m_idx >> i) & 1) != 0)
        {
            m_cmask |= (1ull << host_qubits[i]);
        }
    }

    m_device_opt_qubits = opt_qubits;
    m_opt_num = opt_num;
    return ;
}

__device__ double ProbFun::operator()(int64_t i)
{

    double ret;
    ret = 0.0;

    if((i & m_mask) == m_cmask)
    {
        ret = m_state[i].real()*m_state[i].real() +
                m_state[i].imag()*m_state[i].imag();
    }
    return ret;
}



void exec_probs_measure(const host_qubit_t &qubits,
                           device_state_t &state,
                           int64_t qubit_num,
                           cudaStream_t &stream,
                           prob_vec &probs)
{
#if 0
    int64_t num = qubits.size();
    int64_t dim = 1ll << num;
    //int64_t size  = 1ll << (qubit_num - num);
    int64_t size  = 1ll << qubit_num;
    probs.resize(dim, 0);

    device_qubit_t opt_qubits(qubits.size(), 0);
    thrust::copy_n(qubits.begin(), qubits.size(), opt_qubits.begin());
    device_qsize_t *device_qubits = thrust::raw_pointer_cast(opt_qubits.data());;

    ProbFun fun;
    auto iter = thrust::counting_iterator<int64_t>(0);
    for(int64_t i = 0; i < dim; i++)
    {
        fun.set_state(state.data());
        fun.set_idx(i);
        fun.set_qubits(qubits, device_qubits, num, stream);

        probs[i] = thrust::transform_reduce(thrust::cuda::par.on(stream),
                                            iter,
                                            iter + size,
                                            fun,
                                            0.0,
                                            thrust::plus<double>());
    }

#else

    int64_t size  = 1ll << qubit_num;
    int64_t dim = 1ll << qubits.size();

    QStat host_state(size, 0);
    cudaMemcpyAsync(host_state.data(),
                    thrust::raw_pointer_cast(state.data()),
                    state.size()*sizeof(device_complex_t),
                    cudaMemcpyDeviceToHost, stream);

    probs.resize(dim);
    for (int64_t i = 0; i < size; i++)
    {
        int64_t idx = 0;
        for (int64_t j = 0; j < qubits.size(); j++)
        {
            idx += (((i >> (qubits[j])) % 2) << j);
        }
        probs[idx] += host_state[i].real()*host_state[i].real() +
                host_state[i].imag()*host_state[i].imag();
    }
#endif
    return ;
}


void exec_probs_measure(const host_qubit_t &qubits,
                        device_state_t &state,
                        int64_t qubit_num,
                        cudaStream_t &stream,
                        prob_tuple &probs,
                        int select_max)
{
    int64_t size  = 1ll << qubit_num;
    int64_t dim = 1ll << qubits.size();
    QStat host_state(size, 0);
    cudaMemcpyAsync(host_state.data(),
                    thrust::raw_pointer_cast(state.data()),
                    state.size()*sizeof(device_complex_t),
                    cudaMemcpyDeviceToHost, stream);

    probs.resize(1ll << dim);
    for (int64_t i = 0; i < size; i++)
    {
        int64_t idx = 0;
        for (int64_t j = 0; j < qubits.size(); j++)
        {
            idx += (((i >> (qubits[j])) % 2) << j);
        }
        probs[idx].second += host_state[i].real()*host_state[i].real() +
                host_state[i].imag()*host_state[i].imag();
    }


    if (select_max == -1 || probs.size() <= select_max)
    {
        return ;
    }
    else
    {
        stable_sort(probs.begin(), probs.end(),
                    [](std::pair<size_t, double> a, std::pair<size_t, double> b){
            return a.second > b.second;
        });
        probs.erase(probs.begin() + select_max, probs.end());
    }
    return ;
}

double exec_measure(MeasureFun &fun, int64_t size, cudaStream_t &stream)
{
    auto iter = thrust::counting_iterator<size_t>(0);
    double dprob = thrust::transform_reduce(thrust::cuda::par.on(stream),
                                    iter,
                                    iter + size,
                                    fun,
                                    0.0,
                                    thrust::plus<double>());
    return dprob;
}


void exec_normalize(NormlizeFun &fun, int64_t size, cudaStream_t &stream)
{
    auto iter = thrust::counting_iterator<size_t>(0);
    thrust::for_each(thrust::cuda::par.on(stream),
                     iter, iter + size,
                     fun);
    return ;
}








