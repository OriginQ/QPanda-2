#ifndef _FUN_GATES_H_
#define _FUN_GATES_H_

#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.cuh"



class BaseGateFun {
protected:
    device_complex_t* m_state = nullptr;
    qstate_type* m_param = nullptr;
    device_complex_t* m_device_matrix = nullptr;
    device_qsize_t* m_device_opt_qubits = nullptr;

    device_qsize_t m_qubit_num = 0;
    device_qsize_t m_opt_num = 0;
    int64_t m_offset0 = 0;
    int64_t m_offset1 = 0;

    bool m_is_dagger = false;
    int64_t m_cmask = 0;
public:
    BaseGateFun();
    void set_state(device_state_t& state);
    void set_qubit_num(device_qsize_t qubit_num);
    void set_device_prams(device_qubit_t& device_qubits,
                          device_state_t& device_matrix);

    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream) = 0;
    virtual void set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t stream) = 0;
    virtual __device__ int64_t insert(int64_t) = 0;
    virtual __device__ double operator()(int64_t i) = 0;

    virtual ~BaseGateFun();
};

class SingleGateFun : public BaseGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual void set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t stream);
    virtual __device__ int64_t insert(int64_t i);
    virtual __device__ double operator()(int64_t i);
};


class XFun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};

class YFun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};

class ZFun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};


class RZFun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};

class SFun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};

class HFun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};

class U1Fun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};


class PFun : public SingleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__  double operator()(int64_t i);
};


class DoubleGateFun : public BaseGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual void set_qubits(const host_qubit_t &qubits, device_qsize_t opt_num, cudaStream_t stream);
    virtual __device__ int64_t insert(int64_t i);
    virtual __device__ double operator()(int64_t i);
};

class CNOTFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};

class CZFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};

class CRFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};

class CPFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};


class SWAPFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};

class ISWAPFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};

class ISWAPThetaFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};

class CUFun : public DoubleGateFun
{
public:
    virtual void set_matrix(host_state_t &matrix, bool is_dagger, cudaStream_t stream);
    virtual __device__ double operator()(int64_t i);
};



class ProbFun
{
protected:
    int64_t m_mask = 0;
    int64_t m_cmask = 0;
    int64_t m_idx;
    device_complex_t* m_state = nullptr;

    device_qsize_t* m_device_opt_qubits = nullptr;
    device_qsize_t m_qubit_num = 0;
    device_qsize_t m_opt_num = 0;
public:
    ProbFun();
    void set_state(device_complex_ptr_t state);
    void set_idx(int64_t idx);
    void set_qubits(const host_qubit_t &qubits, device_qsize_t *opt_qubits,
                    device_qsize_t opt_num, cudaStream_t stream);
    __device__ double operator()(int64_t);
};

class MeasureFun : public SingleGateFun
{
public:
   __device__ double operator()(int64_t i);
private:
};


class NormlizeFun : public SingleGateFun
{
public:
    NormlizeFun();
    NormlizeFun(double prob, bool measure_out);
    void set_measure_out(double prob, bool measure_out);
    __device__ double operator()(int64_t i);
private:
    double m_prob;
    bool m_measure_out;
};


void exec_probs_measure(const host_qubit_t &qubits,
                           device_state_t &state,
                           int64_t qubit_num,
                           cudaStream_t &stream,
                           prob_vec &probs);
void exec_probs_measure(const host_qubit_t &qubits,
                           device_state_t &state,
                           int64_t qubit_num,
                           cudaStream_t &stream,
                           prob_tuple &probs,
                           int select_max);

double exec_measure(MeasureFun &fun, int64_t size, cudaStream_t &stream);
void exec_normalize(NormlizeFun &fun, int64_t size, cudaStream_t &stream);


template <typename FunGate> __global__
void exec_gate_kernel(FunGate fun, int64_t size);

#endif // !_FUN_GATES_H_








































