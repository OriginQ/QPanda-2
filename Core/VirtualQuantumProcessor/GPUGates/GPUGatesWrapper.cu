/***********************************************************************
Copyright:
Author:Xue Cheng
Date:2017-12-13
Description: Definition of Encapsulation of GPU gates

Author:Sun KunFeng
Date:2022-05-17
Description: add multi-gpu support
************************************************************************/
#include <chrono>
#include <csignal>
#include "Core/Utilities/Tools/Utils.h"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUGates.cuh"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUGatesWrapper.cuh"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
#include <unistd.h>
#endif

USING_QPANDA
using namespace std;

bool DeviceQPU::locked = { false };
std::vector<int> DeviceQPU::m_device_used = {};
SharedMemory* DeviceQPU::m_share = { nullptr };
struct GPU_USED* DeviceQPU::m_used = { nullptr };

DeviceQPU::DeviceQPU()
{
    PRINT_DEBUG_MESSAGE
    device_count();
}

DeviceQPU::~DeviceQPU()
{
    PRINT_DEBUG_MESSAGE
    device_data_unalloc();
    device_unlink();
    uninit();
}

void DeviceQPU::uninit()
{
    PRINT_DEBUG_MESSAGE
    if (m_share != nullptr)
    {
        PRINT_DEBUG_MESSAGE
        if (m_device_used.size() > 0)
        {
            PRINT_DEBUG_MESSAGE
            if (!locked)
            {
                PRINT_DEBUG_MESSAGE
#if defined(_WIN32) || defined(_WIN64)
                while (!m_used->m_mutex.try_lock())
                {
                    Sleep(1);
                }
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
                while (!m_used->m_mutex.try_lock())
                {
                    usleep(1);
                }
#endif
                locked = true;
            }
            for (int i = 0; i < m_device_used.size(); i++)
            {
                m_used->m_count += 1;
                m_used->m_device[m_device_used[i]] = m_device_used[i];
            }
            m_device_used.clear();
        }
        if (locked)
        {
            PRINT_DEBUG_MESSAGE
            locked = false;
            m_used->m_mutex.unlock();
        }
        if ((m_used->m_thread -= 1) == 0)
        {
            PRINT_DEBUG_MESSAGE
            m_share->memory_delete();
        }
        delete m_share, m_share = nullptr;
    }
}

void DeviceQPU::abort(int signals)
{
    PRINT_DEBUG_MESSAGE
    device_unlink();
    uninit();
    exit(0);
}

int DeviceQPU::device_count()
{
    PRINT_DEBUG_MESSAGE
    CHECK_CUDA(cudaGetDeviceCount(&m_device_num));
    return m_device_num;
}

void DeviceQPU::device_share(void)
{
    PRINT_DEBUG_MESSAGE
    std::signal(SIGFPE, abort);
    std::signal(SIGILL, abort);
    std::signal(SIGINT, abort);
    std::signal(SIGABRT, abort);
    std::signal(SIGSEGV, abort);
    std::signal(SIGTERM, abort);
    m_share = new SharedMemory(sizeof(struct GPU_USED), "GPU_USED");
    m_used = (struct GPU_USED*&)m_share->memory();
    m_used->m_init = false;
    if ((m_used->m_thread += 1) == 1)
    {
        PRINT_DEBUG_MESSAGE
        locked = false;
        std::mutex mm_mutex;
        memcpy(&m_used->m_mutex, &mm_mutex, sizeof(std::mutex));
    }
}

void DeviceQPU::device_links(void)
{
    PRINT_DEBUG_MESSAGE
    int device_link_enable;
    for (int i = 0; i < m_device_used.size(); i++)
    {
        for (int j = i + 1; j < m_device_used.size(); j++)
        {
            CHECK_CUDA(cudaDeviceCanAccessPeer(&device_link_enable, m_device_used[i], m_device_used[j]));
            if (device_link_enable)
            {
                CHECK_CUDA(cudaSetDevice(m_device_used[i]));
                CHECK_CUDA(cudaDeviceEnablePeerAccess(m_device_used[j], 0));
                CHECK_CUDA(cudaSetDevice(m_device_used[j]));
                CHECK_CUDA(cudaDeviceEnablePeerAccess(m_device_used[i], 0));
            }
        }
    }
}

void DeviceQPU::device_unlink(void)
{
    PRINT_DEBUG_MESSAGE
    int device_link_enable;
    if (m_device_used.size() > 1)
    {
        PRINT_DEBUG_MESSAGE
        for (int i = 0; i < m_device_used.size(); i++)
        {
            for (int j = i + 1; j < m_device_used.size(); j++)
            {
                CHECK_CUDA(cudaDeviceCanAccessPeer(&device_link_enable, m_device_used[i], m_device_used[j]));
                if (device_link_enable)
                {
                    CHECK_CUDA(cudaSetDevice(m_device_used[i]));
                    CHECK_CUDA(cudaDeviceDisablePeerAccess(m_device_used[j]));
                    CHECK_CUDA(cudaSetDevice(m_device_used[j]));
                    CHECK_CUDA(cudaDeviceDisablePeerAccess(m_device_used[i]));
                }
            }
        }
    }
}

void DeviceQPU::device_status_init(void)
{
    PRINT_DEBUG_MESSAGE
    for (int i = 0; i < m_device_num; i++)
    {
        if (m_used->m_device[i] >= 0)
        {
            CHECK_CUDA(cudaSetDevice(i));
            cuda_device_status.push_back(QCuda::device_status());
            cuda_device_status[cuda_device_status.size() - 1].m_device = i;
            CHECK_CUDA(cudaMemGetInfo(&cuda_device_status[cuda_device_status.size() - 1].free_size, &cuda_device_status[cuda_device_status.size() - 1].total_size));
        }
    }
}

bool DeviceQPU::device_data_alloc(size_t data_count)
{
    PRINT_DEBUG_MESSAGE
#if 1
    if (data_count > 0)
    {
        PRINT_DEBUG_MESSAGE
        device_share();
    cuda_find_device:
#if defined(_WIN32) || defined(_WIN64)
        while (!m_used->m_mutex.try_lock())
        {
            Sleep(1);
        }
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
        while (!m_used->m_mutex.try_lock())
        {
            usleep(1);
        }
#endif
        locked = true;
        if (m_used->m_count == 0)
        {
            PRINT_DEBUG_MESSAGE
            if (m_used->m_init)
            {
                PRINT_DEBUG_MESSAGE
                locked = false;
                m_used->m_mutex.unlock();
                goto cuda_find_device;
            }
            else
            {
                PRINT_DEBUG_MESSAGE
                if (m_device_num > 0)
                {
                    PRINT_DEBUG_MESSAGE
                    m_used->m_count = m_device_num;
                    for (int i = 0; i < DEVICE_COUNT; i++)
                    {
                        m_used->m_device[i] = -1;
                    }
                    for (int i = 0; i < m_device_num; i++)
                    {
                        m_used->m_device[i] = i;
                    }
                }
                else
                {
                    PRINT_DEBUG_MESSAGE
                    locked = false;
                    m_used->m_mutex.unlock();
                    m_share->memory_delete();
                    throw std::runtime_error("can't find gpu.");
                }
                m_used->m_init = true;
            }
        }
        device_status_init();
        size_t device_memory_unalloc = data_count * sizeof(device_complex_t);
        for (int i = 0; i < cuda_device_status.size(); i++)
        {
            if (cuda_device_status[i].free_size > free_size)
            {
                if (device_memory_unalloc > cuda_device_status[i].free_size - free_size)
                {
                    device_memory_unalloc -= cuda_device_status[i].free_size - free_size;
                }
                else
                {
                    device_memory_unalloc = 0;
                    break;
                }
            }
        }
        if (device_memory_unalloc == 0)
        {
            PRINT_DEBUG_MESSAGE
            for (int i = 0; i < cuda_device_status.size(); i++)
            {
                if (data_count > 0)
                {
                    if (cuda_device_status[i].free_size > free_size && (cuda_device_status[i].free_size - free_size) / sizeof(device_complex_t) > 0)
                    {
                        m_used->m_count -= 1;
                        CHECK_CUDA(cudaSetDevice(cuda_device_status[i].m_device));
                        cuda_device_state.push_back(new QCuda::device_state);
                        m_used->m_device[cuda_device_status[i].m_device] = -1;
                        m_device_used.push_back(cuda_device_status[i].m_device);
                        cuda_device_state[cuda_device_state.size() - 1]->device_data.device_id = cuda_device_status[i].m_device;
                        if (cuda_device_status[i].free_size - free_size >= data_count * sizeof(device_complex_t))
                        {
                            cuda_device_state[cuda_device_state.size() - 1]->device_data.data_count = data_count;
                        }
                        else
                        {
                            cuda_device_state[cuda_device_state.size() - 1]->device_data.data_count = (cuda_device_status[i].free_size - free_size) / sizeof(device_complex_t);
                        }
                        if (cuda_device_state.size() == 1)
                        {
                            cuda_device_state[cuda_device_state.size() - 1]->device_data.data_start = 0;
                        }
                        else
                        {
                            cuda_device_state[cuda_device_state.size() - 1]->device_data.data_start = cuda_device_state[cuda_device_state.size() - 2]->device_data.data_start + cuda_device_state[cuda_device_state.size() - 2]->device_data.data_count;
                        }
                        data_count -= cuda_device_state[cuda_device_state.size() - 1]->device_data.data_count;
                        CHECK_CUDA(cudaStreamCreateWithFlags(&cuda_device_state[cuda_device_state.size() - 1]->device_data.cuda_stream, cudaStreamNonBlocking));
                        cuda_device_state[cuda_device_state.size() - 1]->device_data.data_vector.resize(cuda_device_state[cuda_device_state.size() - 1]->device_data.data_count);
                        thrust::fill(cuda_device_state[cuda_device_state.size() - 1]->device_data.data_vector.begin(), cuda_device_state[cuda_device_state.size() - 1]->device_data.data_vector.end(), 0);
                    }
                }
                else
                {
                    break;
                }
            }
            for (int i = 0; i < cuda_device_state.size(); i++)
            {
                CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
                cuda_device_state[i]->device_data.device_data_ptr.resize(cuda_device_state.size());
                thrust::host_vector<QCuda::device_data_ptr> host_data_ptr(cuda_device_state.size());
                for (int j = 0; j < cuda_device_state.size(); j++)
                {
                    host_data_ptr[j].data_count = cuda_device_state[j]->device_data.data_count;
                    host_data_ptr[j].data_start = cuda_device_state[j]->device_data.data_start;
                    host_data_ptr[j].data_vector = thrust::raw_pointer_cast(cuda_device_state[j]->device_data.data_vector.data());
                    host_data_ptr[j].next_data_ptr = j < cuda_device_state.size() - 1 ? thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()) + j + 1 : nullptr;
                }
                cuda_device_state[i]->device_data.device_data_ptr = host_data_ptr;
            }
            locked = false;
            m_used->m_mutex.unlock();
            device_links();
            return true;
        }
        locked = false;
        m_used->m_mutex.unlock();
        return false;
    }
    return false;
#else
        if (m_device_num > 0 && data_count > 0)
        {
            int device_used_num = { -1 };
            std::vector<int>* device_malloc_used;
            std::vector<size_t> thread_count(m_device_num);
            std::vector<size_t> device_memory(m_device_num);
            std::vector<size_t> device_data_count(m_device_num);
            std::vector<size_t> device_memory_unalloc_max(m_device_num);
            std::vector<size_t> device_memory_unalloc_min(m_device_num);
            std::vector<std::vector<int>> device_used_max(m_device_num);
            std::vector<std::vector<int>> device_used_min(m_device_num);
            for (int i = 0; i < m_device_num; i++)
            {
                device_memory_unalloc_max[i] = data_count * sizeof(device_complex_t);
                device_memory_unalloc_min[i] = data_count * sizeof(device_complex_t);
                device_data_count[i] = size_t(ceill((long double)data_count / (i + 1)));
                thread_count[i] = size_t(ceill((long double)(data_count >> 1) / (i + 1)));
                device_memory[i] = device_data_count[i] * sizeof(device_complex_t) + free_size;
            }
            for (int i = 0; i < m_device_num; i++)
            {
                for (int j = 0; j < m_device_num; j++)
                {
                    if (cuda_device_status[i].free_size > device_memory[j])
                    {
                        device_used_max[j].push_back(i);
                        if (device_memory_unalloc_max[j] > 0)
                        {
                            if (device_memory_unalloc_max[j] > device_memory[j] - free_size)
                            {
                                device_memory_unalloc_max[j] -= device_memory[j] - free_size;
                            }
                            else
                            {
                                device_memory_unalloc_max[j] = 0;
                            }
                        }
                        if (cuda_device_status[i].thread_wait >= thread_count[j])
                        {
                            device_used_min[j].push_back(i);
                            if (device_memory_unalloc_min[j] > device_memory[j] - free_size)
                            {
                                device_memory_unalloc_min[j] -= device_memory[j] - free_size;
                            }
                            else
                            {
                                device_used_num = j;
                                device_malloc_used = &device_used_min[device_used_num];
                                goto cuda_device_data_alloc;
                            }
                        }
                    }
                }
            }
            if (device_memory_unalloc_max[m_device_num - 1] == 0)
            {
                device_used_num = m_device_num - 1;
                device_malloc_used = &device_used_max[device_used_num];
            cuda_device_data_alloc:
                for (int i = 0; i < device_malloc_used->size(); i++)
                {
                    CHECK_CUDA(cudaSetDevice(device_malloc_used->operator[](i)));
                    cuda_device_state.push_back(new QCuda::device_state);
                    cuda_device_state[i]->device_data.device_id = device_malloc_used->operator[](i);
                    cuda_device_state[i]->device_data.device_data_ptr.resize(device_malloc_used->size());
                    CHECK_CUDA(cudaStreamCreateWithFlags(&cuda_device_state[i]->device_data.cuda_stream, cudaStreamNonBlocking))
                        cuda_device_state[i]->device_data.data_start = i == 0 ? 0 : cuda_device_state[cuda_device_state.size() - 2]->device_data.data_start + cuda_device_state[cuda_device_state.size() - 2]->device_data.data_count;
                    if (i < device_malloc_used->size() - 1 || device_malloc_used->size() == 1)
                    {
                        cuda_device_state[i]->device_data.data_count = device_data_count[device_used_num];
                        cuda_device_state[i]->device_data.data_vector.resize(cuda_device_state[i]->device_data.data_count);
                        thrust::fill(cuda_device_state[i]->device_data.data_vector.begin(), cuda_device_state[i]->device_data.data_vector.end(), 0);
                    }
                }
                CHECK_CUDA(cudaSetDevice(device_malloc_used->operator[](device_malloc_used->size() - 1)));
                if (device_malloc_used->size() > 1)
                {
                    cuda_device_state[cuda_device_state.size() - 1]->device_data.data_count = data_count - cuda_device_state[cuda_device_state.size() - 2]->device_data.data_count;
                    cuda_device_state[cuda_device_state.size() - 1]->device_data.data_vector.resize(cuda_device_state[cuda_device_state.size() - 1]->device_data.data_count);
                    thrust::fill(cuda_device_state[cuda_device_state.size() - 1]->device_data.data_vector.begin(), cuda_device_state[cuda_device_state.size() - 1]->device_data.data_vector.end(), 0);
                }

                for (int i = 0; i < device_malloc_used->size(); i++)
                {
                    thrust::host_vector<QCuda::device_data_ptr> host_data_ptr(device_malloc_used->size());
                    CHECK_CUDA(cudaSetDevice(device_malloc_used->operator[](i)));
                    for (int j = 0; j < device_malloc_used->size(); j++)
                    {
                        host_data_ptr[j].device_id = cuda_device_state[j]->device_data.device_id;
                        host_data_ptr[j].data_count = cuda_device_state[j]->device_data.data_count;
                        host_data_ptr[j].data_start = cuda_device_state[j]->device_data.data_start;
                        host_data_ptr[j].data_vector = thrust::raw_pointer_cast(cuda_device_state[j]->device_data.data_vector.data());
                        host_data_ptr[j].next_data_ptr = j < device_malloc_used->size() - 1 ? thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()) + j + 1 : nullptr;
                    }
                    cuda_device_state[i]->device_data.device_data_ptr = host_data_ptr;
                }
                return true;
            }
        }
    return false;
#endif
}

void DeviceQPU::device_data_unalloc(void)
{
    PRINT_DEBUG_MESSAGE
    if (!cuda_device_state.empty())
    {
        PRINT_DEBUG_MESSAGE
        for (int i = 0; i < cuda_device_state.size(); i++)
        {
            if (cuda_device_state[i])
            {
                CHECK_CUDA(cudaStreamDestroy(cuda_device_state[i]->device_data.cuda_stream))
                    delete cuda_device_state[i], cuda_device_state[i] = nullptr;
            }
        }
        cuda_device_state.clear(), cuda_device_state.shrink_to_fit();
    }
}

void DeviceQPU::device_data_init(void)
{
    PRINT_DEBUG_MESSAGE
    if (!cuda_device_state.empty())
    {
        PRINT_DEBUG_MESSAGE
        for (int i = 0; i < cuda_device_state.size(); i++)
        {
            if (cuda_device_state[i])
            {
                thrust::fill(cuda_device_state[i]->device_data.data_vector.begin(), cuda_device_state[i]->device_data.data_vector.end(), 0);
            }
        }
    }
}

bool DeviceQPU::init_state(size_t qnum, const QStat& state)
{
    PRINT_DEBUG_MESSAGE
    if (state.size() == 0)
    {
        PRINT_DEBUG_MESSAGE
        m_qubit_num == qnum ? device_data_init() : (is_init = false, device_data_unalloc());
        if (!is_init && (m_qubit_num = qnum) && (device_status_size = 1ll << m_qubit_num) && !device_data_alloc(device_status_size))
        {
            throw(std::runtime_error("memory out of range"));
        }
        for (int i = 0; i < cuda_device_state.size(); i++)
        {
            CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
            cuda_device_state[i]->device_data.m_device_qubits.resize(m_max_qubit_num, 0);
            cuda_device_state[i]->device_data.m_device_matrix.resize(m_max_matrix_size, 0);
        }
        cuda_device_state[0]->device_data.data_vector[0] = { 1.0,0.0 };
    }
    else
    {
        PRINT_DEBUG_MESSAGE
        size_t alloc_pos = { 0 };
        qnum = (int)std::log2(state.size());
        m_qubit_num == qnum ? device_data_init() : (is_init = false, device_data_unalloc());
        if (!is_init && (m_qubit_num = qnum) && (device_status_size = 1ll << m_qubit_num) && !device_data_alloc(device_status_size))
        {
            throw(std::runtime_error("memory out of range"));
        }
        for (int i = 0; i < cuda_device_state.size(); i++)
        {
            cuda_device_state[i]->device_data.data_vector.assign(state.begin() + alloc_pos, state.begin() + alloc_pos + cuda_device_state[i]->device_data.data_count);
            alloc_pos += cuda_device_state[i]->device_data.data_count;
        }
    }
    // PRINT_DEBUG_CUDAINFO
    return is_init == true ? true : (is_init = init());
}

bool DeviceQPU::init()
{
    PRINT_DEBUG_MESSAGE
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::S_GATE, std::shared_ptr<BaseGateFun>(new SFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::P_GATE, std::shared_ptr<BaseGateFun>(new PFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::T_GATE, std::shared_ptr<BaseGateFun>(new U1Fun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::U1_GATE, std::shared_ptr<BaseGateFun>(new U1Fun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RZ_GATE, std::shared_ptr<BaseGateFun>(new RZFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::Z_HALF_PI, std::shared_ptr<BaseGateFun>(new RZFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::HADAMARD_GATE, std::shared_ptr<BaseGateFun>(new HFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::PAULI_X_GATE, std::shared_ptr<BaseGateFun>(new XFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::PAULI_Y_GATE, std::shared_ptr<BaseGateFun>(new YFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::PAULI_Z_GATE, std::shared_ptr<BaseGateFun>(new ZFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::I_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::ECHO_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::BARRIER_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::P0_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::P1_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RX_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RY_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::U2_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::U3_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::U4_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::X_HALF_PI, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::Y_HALF_PI, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RPHI_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::CP_GATE, std::shared_ptr<BaseGateFun>(new CPFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::CZ_GATE, std::shared_ptr<BaseGateFun>(new CZFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::CNOT_GATE, std::shared_ptr<BaseGateFun>(new CNOTFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::CPHASE_GATE, std::shared_ptr<BaseGateFun>(new CRFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::CU_GATE, std::shared_ptr<BaseGateFun>(new CUFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::SWAP_GATE, std::shared_ptr<BaseGateFun>(new SWAPFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::ISWAP_GATE, std::shared_ptr<BaseGateFun>(new ISWAPFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::SQISWAP_GATE, std::shared_ptr<BaseGateFun>(new ISWAPThetaFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::ISWAP_THETA_GATE, std::shared_ptr<BaseGateFun>(new ISWAPThetaFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::P00_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::P11_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RXX_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RYY_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RZZ_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::RZX_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::TWO_QUBIT_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::MS_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::ORACLE_GATE, std::shared_ptr<BaseGateFun>(new ORACLEFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::CORACLE_GATE, std::shared_ptr<BaseGateFun>(new CORACLEFun()) });

        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::NoiseSingle_GATE, std::shared_ptr<BaseGateFun>(new SingleGateFun()) });
        cuda_device_state[i]->m_type_gate_fun.insert({ GateType::NoiseDouble_GATE, std::shared_ptr<BaseGateFun>(new DoubleGateFun()) });

        cuda_device_state[i]->m_measure_fun = std::shared_ptr<MeasureFun>(new MeasureFun());
        cuda_device_state[i]->m_norm_fun = std::shared_ptr<NormlizeFun>(new NormlizeFun());

        if (cuda_device_state.size() > 1)
        {
            cuda_device_state[i]->m_type_gate_fun[GateType::S_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::P_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::T_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U1_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::RZ_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::Z_HALF_PI]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::HADAMARD_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::PAULI_X_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::PAULI_Y_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::PAULI_Z_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::I_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::ECHO_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::BARRIER_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::P0_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::P1_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RX_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RY_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U2_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U3_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U4_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::X_HALF_PI]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::Y_HALF_PI]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RPHI_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::CP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CZ_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CNOT_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CPHASE_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::CU_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::SWAP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::ISWAP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::SQISWAP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::ISWAP_THETA_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::P00_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::P11_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RXX_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RYY_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RZZ_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RZX_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::MS_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::TWO_QUBIT_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::ORACLE_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CORACLE_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::NoiseSingle_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::NoiseDouble_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_measure_fun->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_norm_fun->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.device_data_ptr.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
        }
        else
        {
            cuda_device_state[i]->m_type_gate_fun[GateType::S_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::P_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::T_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U1_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::RZ_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::Z_HALF_PI]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::HADAMARD_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::PAULI_X_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::PAULI_Y_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::PAULI_Z_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::I_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::ECHO_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::BARRIER_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::P0_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::P1_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RX_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RY_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U2_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U3_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::U4_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::X_HALF_PI]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::Y_HALF_PI]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RPHI_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::CP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CZ_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CNOT_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CPHASE_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::CU_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::SWAP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::ISWAP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::SQISWAP_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::ISWAP_THETA_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::P00_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::P11_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RXX_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RYY_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RZZ_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::RZX_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::MS_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::TWO_QUBIT_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::ORACLE_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::CORACLE_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_type_gate_fun[GateType::NoiseSingle_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_type_gate_fun[GateType::NoiseDouble_GATE]->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);

            cuda_device_state[i]->m_measure_fun->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
            cuda_device_state[i]->m_norm_fun->set_ptr(thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.m_device_qubits, cuda_device_state[i]->device_data.m_device_matrix);
        }
    }
    return true;
}

void DeviceQPU::device_debug(const std::string& flag, device_state_t& device_data)
{
    PRINT_DEBUG_MESSAGE
    std::cout << flag << std::endl;
    thrust::host_vector<thrust::complex<qstate_type>> state = {};
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        CHECK_CUDA(cudaStreamSynchronize(cuda_device_state[i]->device_data.cuda_stream));
        state.insert(state.end(), cuda_device_state[i]->device_data.data_vector.begin(), cuda_device_state[i]->device_data.data_vector.end());
    }
    for (auto val : state)
    {
        std::cout << val << std::endl;
    }
}

void DeviceQPU::exec_gate(GateType type, QStat& matrix, const Qnum& qnum, size_t num, bool is_dagger)
{
    PRINT_DEBUG_MESSAGE
    size_t measure_size = 1ll << (m_qubit_num - num);
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        auto iter = cuda_device_state[i]->m_type_gate_fun.find(type);
        if (cuda_device_state[i]->m_type_gate_fun.end() == iter)
        {
            throw std::runtime_error("gate type");
        }
        this->exec_gate(iter->second, type, matrix, qnum, num, is_dagger, measure_size, i);
    }
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaStreamSynchronize(cuda_device_state[i]->device_data.cuda_stream));
    }
}

void DeviceQPU::exec_gate(GateType type, QStat& matrix, const Qnum& qnum, const Qnum& control, bool is_dagger)
{
    PRINT_DEBUG_MESSAGE
    size_t measure_size = 1ll << (m_qubit_num - qnum.size());
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        auto iter = cuda_device_state[i]->m_type_gate_fun.find(type);
        if (cuda_device_state[i]->m_type_gate_fun.end() == iter)
        {
            throw std::runtime_error("gate type");
        }
        this->exec_gate(iter->second, type, matrix, qnum, control, is_dagger, measure_size, i);
    }
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaStreamSynchronize(cuda_device_state[i]->device_data.cuda_stream));
    }
}

void DeviceQPU::exec_gate(std::shared_ptr<BaseGateFun> fun, GateType type, QStat& matrix, const Qnum& qnum, size_t num, bool is_dagger, size_t& measure_size, int id)
{
    PRINT_DEBUG_MESSAGE
    size_t grid, block;
    size_t thread_count;
    size_t thread_start;
    fun->set_qubits(qnum, num, cuda_device_state[id]->device_data.cuda_stream);
    fun->set_matrix(matrix, is_dagger, cuda_device_state[id]->device_data.cuda_stream);

    thread_count = cuda_device_state[id]->device_data.data_count % (1ll << num) == 0 ? cuda_device_state[id]->device_data.data_count / (1ll << num) : cuda_device_state[id]->device_data.data_count / (1ll << num) + 1;
    thread_start = cuda_device_state[id]->device_data.data_start % (1ll << num) == 0 ? cuda_device_state[id]->device_data.data_start / (1ll << num) : cuda_device_state[id]->device_data.data_start / (1ll << num) + 1;

    block = measure_size > kThreadDim ? kThreadDim : size_t(ceilf(measure_size / 32.0));
    grid = measure_size % block == 0 ? measure_size / block : measure_size / block + 1;

    if (cuda_device_state.size() > 1)
    {
        PRINT_DEBUG_MESSAGE
        switch (type)
        {
        case GateType::I_GATE:
        case GateType::ECHO_GATE:
        case GateType::BARRIER_GATE:
            break;
        case GateType::PAULI_X_GATE:
            exec_gate_kernel_multi<XFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<XFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::PAULI_Y_GATE:
            exec_gate_kernel_multi<YFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<YFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::PAULI_Z_GATE:
            exec_gate_kernel_multi<ZFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ZFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::S_GATE:
            exec_gate_kernel_multi<SFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<SFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::T_GATE:
        case GateType::U1_GATE:
            exec_gate_kernel_multi<U1Fun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<U1Fun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::P_GATE:
            exec_gate_kernel_multi<PFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<PFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::RZ_GATE:
        case GateType::Z_HALF_PI:
            exec_gate_kernel_multi<RZFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<RZFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::HADAMARD_GATE:
            exec_gate_kernel_multi<HFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<HFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::P0_GATE:
        case GateType::P1_GATE:
        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::U2_GATE:
        case GateType::U3_GATE:
        case GateType::U4_GATE:
        case GateType::X_HALF_PI:
        case GateType::Y_HALF_PI:
        case GateType::RPHI_GATE:
        case GateType::NoiseSingle_GATE:
            exec_gate_kernel_multi<SingleGateFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<SingleGateFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CNOT_GATE:
            exec_gate_kernel_multi<CNOTFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CNOTFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CZ_GATE:
            exec_gate_kernel_multi<CZFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CZFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CPHASE_GATE:
            exec_gate_kernel_multi<CRFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CRFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CP_GATE:
            exec_gate_kernel_multi<CPFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CPFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::SWAP_GATE:
            exec_gate_kernel_multi<SWAPFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<SWAPFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::ISWAP_GATE:
            exec_gate_kernel_multi<ISWAPFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ISWAPFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::SQISWAP_GATE:
        case GateType::ISWAP_THETA_GATE:
            exec_gate_kernel_multi<ISWAPThetaFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ISWAPThetaFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CU_GATE:
            exec_gate_kernel_multi<CUFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CUFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::P00_GATE:
        case GateType::P11_GATE:
        case GateType::RXX_GATE:
        case GateType::RYY_GATE:
        case GateType::RZZ_GATE:
        case GateType::RZX_GATE:
        case GateType::MS_GATE:
        case GateType::TWO_QUBIT_GATE:
        case GateType::NoiseDouble_GATE:
            exec_gate_kernel_multi<DoubleGateFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<DoubleGateFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::ORACLE_GATE:
            exec_gate_kernel_multi<ORACLEFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ORACLEFun>(fun), measure_size, thread_start, thread_count);
            break;
        default:
            throw std::runtime_error("Error: gate type: " + std::to_string(type));
        }
    }
    else
    {
        PRINT_DEBUG_MESSAGE
        switch (type)
        {
        case GateType::I_GATE:
        case GateType::ECHO_GATE:
        case GateType::BARRIER_GATE:
            break;
        case GateType::PAULI_X_GATE:
            exec_gate_kernel<XFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<XFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::PAULI_Y_GATE:
            exec_gate_kernel<YFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<YFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::PAULI_Z_GATE:
            exec_gate_kernel<ZFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ZFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::S_GATE:
            exec_gate_kernel<SFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<SFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::T_GATE:
        case GateType::U1_GATE:
            exec_gate_kernel<U1Fun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<U1Fun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::P_GATE:
            exec_gate_kernel<PFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<PFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::RZ_GATE:
        case GateType::Z_HALF_PI:
            exec_gate_kernel<RZFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<RZFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::HADAMARD_GATE:
            exec_gate_kernel<HFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<HFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::P0_GATE:
        case GateType::P1_GATE:
        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::U2_GATE:
        case GateType::U3_GATE:
        case GateType::U4_GATE:
        case GateType::X_HALF_PI:
        case GateType::Y_HALF_PI:
        case GateType::RPHI_GATE:
        case GateType::NoiseSingle_GATE:
            exec_gate_kernel<SingleGateFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<SingleGateFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CNOT_GATE:
            exec_gate_kernel<CNOTFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CNOTFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CZ_GATE:
            exec_gate_kernel<CZFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CZFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CPHASE_GATE:
            exec_gate_kernel<CRFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CRFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CP_GATE:
            exec_gate_kernel<CPFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CPFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::SWAP_GATE:
            exec_gate_kernel<SWAPFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<SWAPFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::ISWAP_GATE:
            exec_gate_kernel<ISWAPFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ISWAPFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::SQISWAP_GATE:
        case GateType::ISWAP_THETA_GATE:
            exec_gate_kernel<ISWAPThetaFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ISWAPThetaFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::CU_GATE:
            exec_gate_kernel<CUFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CUFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::P00_GATE:
        case GateType::P11_GATE:
        case GateType::RXX_GATE:
        case GateType::RYY_GATE:
        case GateType::RZZ_GATE:
        case GateType::RZX_GATE:
        case GateType::MS_GATE:
        case GateType::TWO_QUBIT_GATE:
        case GateType::NoiseDouble_GATE:
            exec_gate_kernel<DoubleGateFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<DoubleGateFun>(fun), measure_size, thread_start, thread_count);
            break;
        case GateType::ORACLE_GATE:
            exec_gate_kernel<ORACLEFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<ORACLEFun>(fun), measure_size, thread_start, thread_count);
            break;
        default:
            throw std::runtime_error("Error: gate type: " + std::to_string(type));
        }
    }
}

void DeviceQPU::exec_gate(std::shared_ptr<BaseGateFun> fun, GateType type, QStat& matrix, const Qnum& qnum, const Qnum& control, bool is_dagger, size_t& measure_size, int id)
{
    PRINT_DEBUG_MESSAGE
    size_t grid, block;
    size_t thread_count;
    size_t thread_start;
    fun->set_qubits(qnum, control, cuda_device_state[id]->device_data.cuda_stream);
    fun->set_matrix(matrix, is_dagger, cuda_device_state[id]->device_data.cuda_stream);

    thread_count = cuda_device_state[id]->device_data.data_count % (1ll << qnum.size()) == 0 ? cuda_device_state[id]->device_data.data_count / (1ll << qnum.size()) : cuda_device_state[id]->device_data.data_count / (1ll << qnum.size()) + 1;
    thread_start = cuda_device_state[id]->device_data.data_start % (1ll << qnum.size()) == 0 ? cuda_device_state[id]->device_data.data_start / (1ll << qnum.size()) : cuda_device_state[id]->device_data.data_start / (1ll << qnum.size()) + 1;

    block = measure_size > kThreadDim ? kThreadDim : size_t(ceilf(measure_size / 32.0));
    grid = measure_size % block == 0 ? measure_size / block : measure_size / block + 1;

    if (cuda_device_state.size() > 1)
    {
        PRINT_DEBUG_MESSAGE
        switch (type)
        {
        case GateType::CORACLE_GATE:
            exec_gate_kernel_multi<CORACLEFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CORACLEFun>(fun), measure_size, thread_start, thread_count);
            break;
        default:
            throw std::runtime_error("Error: gate type: " + std::to_string(type));
        }
    }
    else
    {
        PRINT_DEBUG_MESSAGE
        switch (type)
        {
        case GateType::CORACLE_GATE:
            exec_gate_kernel<CORACLEFun> << <grid, block, 0, cuda_device_state[id]->device_data.cuda_stream >> > (*dynamic_pointer_cast<CORACLEFun>(fun), measure_size, thread_start, thread_count);
            break;
        default:
            throw std::runtime_error("Error: gate type: " + std::to_string(type));
        }
    }
}

void DeviceQPU::probs_measure(const Qnum& qnum, prob_vec& probs)
{
    PRINT_DEBUG_MESSAGE
    int64_t size = { 0 };
    probs.resize(1ll << qnum.size());
    QStat host_state(device_status_size, 0);
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        CHECK_CUDA(cudaMemcpyAsync(host_state.data() + size, thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.data_count * sizeof(device_complex_t), cudaMemcpyDeviceToHost, cuda_device_state[i]->device_data.cuda_stream));
        size += cuda_device_state[i]->device_data.data_count;
    }

    for (int64_t i = 0; i < device_status_size; i++)
    {
        int64_t idx = 0;
        for (int64_t j = 0; j < qnum.size(); j++)
        {
            idx += (((i >> (qnum[j])) % 2) << j);
        }
        probs[idx] += host_state[i].real() * host_state[i].real() + host_state[i].imag() * host_state[i].imag();
    }
}

void DeviceQPU::probs_measure(const Qnum& qnum, prob_tuple& probs, int select_max)
{
    PRINT_DEBUG_MESSAGE
    size_t size = { 0 };
    probs.resize(1ll << qnum.size());
    QStat host_state(device_status_size, 0);
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        CHECK_CUDA(cudaMemcpyAsync(host_state.data() + size, thrust::raw_pointer_cast(cuda_device_state[i]->device_data.data_vector.data()), cuda_device_state[i]->device_data.data_count * sizeof(device_complex_t), cudaMemcpyDeviceToHost, cuda_device_state[i]->device_data.cuda_stream));
        size += cuda_device_state[i]->device_data.data_count;
    }

    for (int64_t i = 0; i < device_status_size; i++)
    {
        int64_t idx = 0;
        for (int64_t j = 0; j < qnum.size(); j++)
        {
            idx += (((i >> (qnum[j])) % 2) << j);
        }
        probs[idx].second += host_state[i].real() * host_state[i].real() + host_state[i].imag() * host_state[i].imag();
    }

    if (select_max != -1 && probs.size() > select_max)
    {
        PRINT_DEBUG_MESSAGE
        stable_sort(probs.begin(), probs.end(), [](std::pair<size_t, double> a, std::pair<size_t, double> b)
                { return a.second > b.second; });
        probs.erase(probs.begin() + select_max, probs.end());
    }
}

bool DeviceQPU::qubit_measure(size_t qn)
{
    PRINT_DEBUG_MESSAGE
    int i = 0, count = cuda_device_state[i]->device_data.data_count;
    int64_t measure_size = 1ll << (m_qubit_num - 1);
    while (count < 1ll << qn && i < cuda_device_state.size())
    {
        i++;
        count += cuda_device_state[i]->device_data.data_count;
    }
    if (i < cuda_device_state.size() && count >= 1ll << qn)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        cuda_device_state[i]->m_measure_fun->set_qubits({ qn }, 1, cuda_device_state[i]->device_data.cuda_stream);
        double dprob = exec_measure(*cuda_device_state[i]->m_measure_fun, measure_size, cuda_device_state[i]->device_data.cuda_stream);
        CHECK_CUDA(cudaStreamSynchronize(cuda_device_state[i]->device_data.cuda_stream));

        bool measure_out = random_generator19937() > dprob ? true : false;
        dprob = measure_out == true ? (1 / sqrt(1 - dprob)) : (1 / sqrt(dprob));
        cuda_device_state[i]->m_norm_fun->set_measure_out(dprob, measure_out);
        cuda_device_state[i]->m_norm_fun->set_qubits({ qn }, 1, cuda_device_state[i]->device_data.cuda_stream);
        exec_normalize(*cuda_device_state[i]->m_norm_fun, measure_size, cuda_device_state[i]->device_data.cuda_stream);
        CHECK_CUDA(cudaStreamSynchronize(cuda_device_state[i]->device_data.cuda_stream));
        return measure_out;
    }
    throw(std::runtime_error("qubit measure error."));
}

void DeviceQPU::get_qstate(QStat& state)
{
    PRINT_DEBUG_MESSAGE
    size_t size = { 0 };
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        size += cuda_device_state[i]->device_data.data_vector.size();
    }
    state.resize(size, 0), size = 0;
    for (int i = 0; i < cuda_device_state.size(); i++)
    {
        CHECK_CUDA(cudaSetDevice(cuda_device_state[i]->device_data.device_id));
        thrust::copy_n(cuda_device_state[i]->device_data.data_vector.begin(), cuda_device_state[i]->device_data.data_vector.size(), state.begin() + size);
        size += cuda_device_state[i]->device_data.data_count;
    }
}

void DeviceQPU::reset(size_t qn)
{
    PRINT_DEBUG_MESSAGE
    auto measure_out = qubit_measure(qn);
    if (measure_out)
    {
        PRINT_DEBUG_MESSAGE
            QStat matrix = { 0, 1, 1, 0 };
        this->exec_gate(GateType::PAULI_X_GATE, matrix, { qn }, 1, false);
    }
}