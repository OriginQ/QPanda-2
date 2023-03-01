#ifndef GPU_GATES_WRAPPER_H
#define GPU_GATES_WRAPPER_H

#include "Core/Utilities/Tools/Macro.h"
#include "Core/Utilities/Tools/SharedMemory.h"
#include "Core/VirtualQuantumProcessor/GPUGates/FunGates.cuh"
#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.cuh"

#define DEVICE_COUNT 32

namespace cuda
{
    struct device_state
    {
        cuda::device_data device_data;
        std::shared_ptr<NormlizeFun> m_norm_fun;
        std::shared_ptr<MeasureFun> m_measure_fun;
        std::map<GateType, std::shared_ptr<BaseGateFun>> m_type_gate_fun;
    };
}

struct GPU_USED
{
    int m_count;
    bool m_init;
    size_t m_thread;
    std::mutex m_mutex;
    int m_device[DEVICE_COUNT];
};

class DeviceQPU
{
public:
    DeviceQPU();
    virtual ~DeviceQPU();

    static bool locked;
    static SharedMemory* m_share;
    static struct GPU_USED* m_used;
    static void abort(int signals);
    static std::vector<int> m_device_used;

    int device_count();
    void reset(size_t qn);
    void get_qstate(QStat& state);
    bool init_state(size_t qnum, const QStat& state = {});
    void device_debug(const std::string& flag, device_state_t& device_data);

    bool qubit_measure(size_t qn);
    void probs_measure(const Qnum& qnum, prob_vec& probs);
    void probs_measure(const Qnum& qnum, prob_tuple& probs, int select_max);

    void exec_gate(GateType type, QStat& matrix, const Qnum& qnum, size_t num, bool is_dagger);
    void exec_gate(GateType type, QStat& matrix, const Qnum& qnum, const Qnum& control, bool is_dagger);
    void exec_gate(std::shared_ptr<BaseGateFun> fun, GateType type, QStat& matrix, const Qnum& qnum, size_t num, bool is_dagger, size_t& measure_size, int id);
    void exec_gate(std::shared_ptr<BaseGateFun> fun, GateType type, QStat& matrix, const Qnum& qnum, const Qnum& control, bool is_dagger, size_t& measure_size, int id);

protected:
    bool init();

    static void uninit();
    static void device_unlink(void);

    void device_share(void);
    void device_links(void);
    void device_data_init(void);
    void device_status_init(void);
    void device_data_unalloc(void);
    bool device_data_alloc(size_t alloc_count);

private:
    device_qsize_t m_qubit_num{ 0 };
    device_state_t m_device_state;

    qstate_type* m_reduce_buffer;
    std::vector<bool> m_peer_access;

    const size_t m_max_qubit_num = 64;
    const size_t m_max_matrix_size = 1024;

    bool is_init = { false };
    int m_device_num = { 0 };
    size_t device_status_size;
    const size_t free_size = { 1024 * 1024 * 300 };
    std::vector<cuda::device_state*> cuda_device_state;
    std::vector<cuda::device_status> cuda_device_status;
};
#endif