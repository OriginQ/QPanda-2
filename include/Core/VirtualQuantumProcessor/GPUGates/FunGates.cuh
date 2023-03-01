#ifndef _FUN_GATES_H_
#define _FUN_GATES_H_

#include "Core/Utilities/Tools/Macro.h"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.cuh"

class BaseGateFun
{
protected:
    size_t m_cmask = 0;
    size_t m_offset0 = 0;
    size_t m_offset1 = 0;
    bool m_is_dagger = false;
    device_qsize_t m_opt_num = 0;
    device_qsize_t m_qubit_num = 0;
    qstate_type *m_param = nullptr;
    cuda::device_data_ptr *device_data_ptr;
    device_complex_t *data_vector = nullptr;
    device_complex_t *m_device_matrix = nullptr;
    device_qsize_t *m_device_opt_qubits = nullptr;

public:
    BaseGateFun();
    virtual ~BaseGateFun();
    void set_qubit_num(device_qsize_t qubit_num);

    virtual __device__ size_t insert(size_t) = 0;
    virtual __device__ double operator()(size_t i) = 0;
    virtual __device__ double operator[](size_t i) = 0;
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream) = 0;
    virtual void set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t &stream) = 0;
    virtual void set_qubits(const host_qubit_t& qubits, const host_qubit_t& control, cudaStream_t& stream) {};

    void set_ptr(device_complex_t *data_ptr, device_qubit_t &device_qubits, device_state_t &device_matrix);
    void set_ptr(cuda::device_data_ptr *data_ptr, device_qubit_t &device_qubits, device_state_t &device_matrix);
};

class SingleGateFun : public BaseGateFun
{
public:
    virtual __device__ size_t insert(size_t i);
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
    virtual void set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t &stream);
};

class XFun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class YFun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class ZFun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class RZFun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class SFun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class HFun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class U1Fun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class PFun : public SingleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class DoubleGateFun : public BaseGateFun
{
public:
    virtual __device__ size_t insert(size_t i);
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
    virtual void set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t &stream);
};

class CNOTFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class CZFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class CRFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class CPFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class SWAPFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class ISWAPFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class ISWAPThetaFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class CUFun : public DoubleGateFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t &stream);
};

class ORACLEFun : public DoubleGateFun
{
public:
    virtual __device__ size_t insert(size_t i);
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_matrix(host_state_t& matrix, bool is_dagger, cudaStream_t& stream);
    virtual void set_qubits(const host_qubit_t& qubits, device_qsize_t opt_num, cudaStream_t& stream);
};

class CORACLEFun : public ORACLEFun
{
public:
    virtual __device__ double operator()(size_t i);
    virtual __device__ double operator[](size_t i);
    virtual void set_qubits(const host_qubit_t& qubits, const host_qubit_t& control, cudaStream_t& stream);
};

class MeasureFun : public SingleGateFun
{
public:
    __device__ double operator()(size_t i);
};

class NormlizeFun : public SingleGateFun
{
public:
    NormlizeFun();
    __device__ double operator()(size_t i);
    NormlizeFun(double prob, bool measure_out);
    void set_measure_out(double prob, bool measure_out);

private:
    double m_prob{0.0};
    bool m_measure_out{false};
};

double exec_measure(MeasureFun &fun, size_t size, cudaStream_t &stream);
void exec_normalize(NormlizeFun &fun, size_t size, cudaStream_t &stream);

template <typename FunGate>
__global__ void exec_gate_kernel(FunGate fun, size_t size, size_t thread_start, size_t thread_count);

template <typename FunGate>
__global__ void exec_gate_kernel_multi(FunGate fun, size_t size, size_t thread_start, size_t thread_count);

#endif