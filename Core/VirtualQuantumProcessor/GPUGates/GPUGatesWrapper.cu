
/***********************************************************************
Copyright:
Author:Xue Cheng
Date:2017-12-13
Description: Definition of Encapsulation of GPU gates
************************************************************************/

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <thrust/transform_reduce.h>
#include <device_launch_parameters.h>
#include "Core/Utilities/Tools/Utils.h"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUGates.cuh"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUGatesWrapper.cuh"

USING_QPANDA
using namespace std;

DeviceQPU::DeviceQPU()
{

}

int DeviceQPU::device_count()
{
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

bool DeviceQPU::init()
{
    m_device_matrix.resize(m_max_matrix_size, 0);
    m_device_qubits.resize(m_max_qubit_num, 0);

    m_type_gate_fun.insert({GateType::I_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::BARRIER_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::ECHO_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});

    m_type_gate_fun.insert({GateType::PAULI_X_GATE, std::shared_ptr<BaseGateFun>(new XFun())});
    m_type_gate_fun.insert({GateType::PAULI_Y_GATE, std::shared_ptr<BaseGateFun>(new YFun())});
    m_type_gate_fun.insert({GateType::PAULI_Z_GATE, std::shared_ptr<BaseGateFun>(new ZFun())});

    m_type_gate_fun.insert({GateType::X_HALF_PI, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::Y_HALF_PI, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::Z_HALF_PI, std::shared_ptr<BaseGateFun>(new RZFun())});

    m_type_gate_fun.insert({GateType::RX_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::RY_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::RZ_GATE, std::shared_ptr<BaseGateFun>(new RZFun())});

    m_type_gate_fun.insert({ GateType::RXX_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
    m_type_gate_fun.insert({ GateType::RYY_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
    m_type_gate_fun.insert({ GateType::RZZ_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
    m_type_gate_fun.insert({ GateType::RZX_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });

    m_type_gate_fun.insert({GateType::S_GATE, std::shared_ptr<BaseGateFun>(new SFun())});
    m_type_gate_fun.insert({GateType::T_GATE, std::shared_ptr<BaseGateFun>(new U1Fun())});
    m_type_gate_fun.insert({GateType::P_GATE, std::shared_ptr<BaseGateFun>(new PFun())});

    m_type_gate_fun.insert({GateType::HADAMARD_GATE, std::shared_ptr<BaseGateFun>(new HFun())});
    m_type_gate_fun.insert({GateType::RPHI_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});

    m_type_gate_fun.insert({GateType::U1_GATE, std::shared_ptr<BaseGateFun>(new U1Fun())});
    m_type_gate_fun.insert({GateType::U2_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::U3_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});
    m_type_gate_fun.insert({GateType::U4_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun())});

    m_type_gate_fun.insert({GateType::CNOT_GATE, std::shared_ptr<BaseGateFun>(new CNOTFun())});
    m_type_gate_fun.insert({GateType::CZ_GATE, std::shared_ptr<BaseGateFun>(new CZFun())});
    m_type_gate_fun.insert({GateType::CPHASE_GATE, std::shared_ptr<BaseGateFun>(new CRFun())});
    m_type_gate_fun.insert({GateType::CP_GATE, std::shared_ptr<BaseGateFun>(new CPFun())});

    m_type_gate_fun.insert({GateType::SWAP_GATE, std::shared_ptr<BaseGateFun>(new SWAPFun())});
    m_type_gate_fun.insert({GateType::ISWAP_GATE, std::shared_ptr<BaseGateFun>(new ISWAPFun())});
    m_type_gate_fun.insert({GateType::ISWAP_THETA_GATE, std::shared_ptr<BaseGateFun>(new ISWAPThetaFun())});
    m_type_gate_fun.insert({GateType::SQISWAP_GATE, std::shared_ptr<BaseGateFun>(new ISWAPThetaFun())});

    m_type_gate_fun.insert({GateType::CU_GATE, std::shared_ptr<BaseGateFun>(new CUFun())});
    m_type_gate_fun.insert({GateType::TWO_QUBIT_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun())});
    m_measure_fun = std::shared_ptr<MeasureFun>(new MeasureFun());
    m_norm_fun = std::shared_ptr<NormlizeFun>(new NormlizeFun());
    return true;
}

bool DeviceQPU::init_state(size_t qnum, const QStat &state)
{
    set_device();
    if (nullptr == m_cuda_stream)
    {
        auto ret = cudaStreamCreateWithFlags(&m_cuda_stream, cudaStreamNonBlocking);
        QPANDA_ASSERT(cudaSuccess != ret, "Error: cudaStreamCreateWithFlags.");
    }
    

    if (0 == state.size())
    {
		m_qubit_num = qnum;
        m_device_state.resize(1ll << m_qubit_num);
        thrust::fill(m_device_state.begin(), m_device_state.end(), 0);
        m_device_state[0] = 1;
    }
    else
    {
		m_qubit_num = (int)std::log2(state.size());
        m_device_state = state;
    }

    return init();
    return true;
}


void DeviceQPU::set_device()
{
    this->m_device_id = 0;
    auto ret = cudaSetDevice(this->m_device_id);
    QPANDA_ASSERT(cudaSuccess != ret, "Error: cudaSetDevice.");
}

void DeviceQPU::device_barrier()
{
    auto ret = cudaStreamSynchronize(m_cuda_stream);
    QPANDA_ASSERT(cudaSuccess != ret, "Error: cudaStreamSynchronize.");
    return ;
}


void DeviceQPU::device_debug(const std::string &flag, device_state_t &device_data)
{
    std::cout << flag << std::endl;
    thrust::host_vector<thrust::complex<qstate_type>> state = m_device_state;
    for (auto val : state)
    {
        std::cout << val << std::endl;
    }
}

void DeviceQPU::exec_gate(std::shared_ptr<BaseGateFun> fun, GateType type, QStat &matrix,
                          const Qnum &qnum, size_t num, bool is_dagger)
{
    set_device();
    fun->set_state(m_device_state);
    fun->set_device_prams(m_device_qubits, m_device_matrix);
    fun->set_matrix(matrix, is_dagger, m_cuda_stream);
    fun->set_qubits(qnum, num, m_cuda_stream);

    size_t dim;
    int64_t size = 1ll << (m_qubit_num - num);
    dim = size / kThreadDim;
    dim = size % kThreadDim ? dim + 1 : dim;

    switch (type)
    {
    case GateType::I_GATE:
    case GateType::BARRIER_GATE:
    case GateType::ECHO_GATE:
        break;
    case GateType::PAULI_X_GATE:
        exec_gate_kernel<XFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<XFun>(fun), size);
        break;
    case GateType::PAULI_Y_GATE:
        exec_gate_kernel<YFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<YFun>(fun), size);
        break;
    case GateType::PAULI_Z_GATE:
        exec_gate_kernel<ZFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<ZFun>(fun), size);
        break;
    case GateType::S_GATE:
        exec_gate_kernel<SFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<SFun>(fun), size);
        break;
    case GateType::T_GATE:
    case GateType::U1_GATE:
        exec_gate_kernel<U1Fun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<U1Fun>(fun), size);
        break;
    case GateType::P_GATE:
        exec_gate_kernel<PFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<PFun>(fun), size);
        break;
    case GateType::RZ_GATE:
    case GateType::Z_HALF_PI:
        exec_gate_kernel<RZFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<RZFun>(fun), size);
        break;
    case GateType::HADAMARD_GATE:
        exec_gate_kernel<HFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<HFun>(fun), size);
        break;
    case GateType::X_HALF_PI:
    case GateType::Y_HALF_PI:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::U2_GATE:
    case GateType::U3_GATE:
    case GateType::U4_GATE:
    case GateType::RPHI_GATE:
        exec_gate_kernel<SingleGateFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<SingleGateFun>(fun), size);
        break;
    case GateType::CNOT_GATE:
        exec_gate_kernel<CNOTFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<CNOTFun>(fun), size);
        break;
    case GateType::CZ_GATE:
        exec_gate_kernel<CZFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<CZFun>(fun), size);
        break;
    case GateType::CPHASE_GATE:
        exec_gate_kernel<CRFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<CRFun>(fun), size);
        break;
    case GateType::CP_GATE:
        exec_gate_kernel<CPFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<CPFun>(fun), size);
        break;
    case GateType::SWAP_GATE:
        exec_gate_kernel<SWAPFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<SWAPFun>(fun), size);
        break;
    case GateType::ISWAP_GATE:
        exec_gate_kernel<ISWAPFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<ISWAPFun>(fun), size);
        break;
    case GateType::ISWAP_THETA_GATE:
    case GateType::SQISWAP_GATE:
        exec_gate_kernel<ISWAPThetaFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<ISWAPThetaFun>(fun), size);
        break;
    case GateType::CU_GATE:
        exec_gate_kernel<CUFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<CUFun>(fun), size);
        break;
    case GateType::RXX_GATE:
    case GateType::RYY_GATE:
    case GateType::RZZ_GATE:
    case GateType::RZX_GATE:
    case GateType::TWO_QUBIT_GATE:
        exec_gate_kernel<DoubleGateFun><<<dim, kThreadDim, 0, m_cuda_stream>>>(*dynamic_pointer_cast<DoubleGateFun>(fun), size);
        break;
    default:
        throw std::runtime_error("Error: gate type: " + std::to_string(type));
    }

    device_barrier();
    return ;
}


void DeviceQPU::exec_gate(GateType type, QStat &matrix, const Qnum &qnum, size_t num, bool is_dagger)
{
    auto iter = m_type_gate_fun.find(type);
    if (m_type_gate_fun.end() == iter)
    {
        throw std::runtime_error("gate type");
    }

    this->exec_gate(iter->second, type, matrix, qnum, num, is_dagger);
    return ;
}


void DeviceQPU::probs_measure(const Qnum &qnum,  prob_vec &probs)
{
    set_device();
    exec_probs_measure(qnum, m_device_state,
                          m_qubit_num,
                          m_cuda_stream,
                          probs);
    return ;
}

void DeviceQPU::probs_measure(const Qnum &qnum, prob_tuple &probs, int select_max)
{
    set_device();
    exec_probs_measure(qnum, m_device_state,
                          m_qubit_num,
                          m_cuda_stream,
                          probs,
                          select_max);
    return ;
}

bool DeviceQPU::qubit_measure(size_t qn)
{
    set_device();
    m_measure_fun->set_device_prams(m_device_qubits, m_device_matrix);
    m_measure_fun->set_state(m_device_state);
    m_measure_fun->set_qubits({qn}, 1, m_cuda_stream);
    int64_t size = 1ll << (m_qubit_num - 1);

    double dprob = exec_measure(*m_measure_fun, size, m_cuda_stream);
    device_barrier();

    bool measure_out = false;
    double fi = random_generator19937();
    if (fi > dprob)
    {
        measure_out = true;
    }

    dprob = measure_out ? 1 / sqrt(1 - dprob) : 1 / sqrt(dprob);
    m_norm_fun->set_device_prams(m_device_qubits, m_device_matrix);
    m_norm_fun->set_measure_out(dprob, measure_out);
    m_norm_fun->set_state(m_device_state);
    m_norm_fun->set_qubits({qn}, 1, m_cuda_stream);

    exec_normalize(*m_norm_fun, size, m_cuda_stream);
    device_barrier();
    return measure_out;
}


void DeviceQPU::get_qstate(QStat &state)
{
    state.resize(m_device_state.size(), 0);
    thrust::copy_n(m_device_state.begin(), m_device_state.size(), state.begin());
    return ;
}


void DeviceQPU::reset(size_t qn)
{
    auto measure_out = qubit_measure(qn);
    if (measure_out)
    {
        QStat matrix = {0, 1, 1, 0};
        this->exec_gate(GateType::PAULI_X_GATE, matrix, {qn}, 1, false);
    }

    return ;
}

DeviceQPU::~DeviceQPU()
{
    if (nullptr != m_cuda_stream)
    {
        cudaStreamDestroy(m_cuda_stream);
        m_cuda_stream = nullptr;
    }
}
