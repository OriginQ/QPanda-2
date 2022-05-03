#ifndef GPU_GATES_WRAPPER_H
#define GPU_GATES_WRAPPER_H

#include "Core/VirtualQuantumProcessor/GPUGates/GPUStruct.cuh"
#include "Core/VirtualQuantumProcessor/GPUGates/FunGates.cuh"

#include <vector>
#include <algorithm>
#include <map>
#include <time.h>


class DeviceQPU
{
public:
    DeviceQPU();
    int device_count();
    bool init();
    bool init_state(size_t qnum, const QStat &state = {});

    void exec_gate(std::shared_ptr<BaseGateFun> fun, GateType type, QStat &matrix,
                   const Qnum &qnum, size_t num, bool is_dagger);

    void exec_gate(GateType type, QStat &matrix,
                   const Qnum &qnum, size_t num, bool is_dagger);

    void probs_measure(const Qnum &qnum,  prob_vec &probs);
    void probs_measure(const Qnum &qnum,  prob_tuple &probs, int select_max);
    bool qubit_measure(size_t qn);

    void device_debug(const std::string &flag, device_state_t &device_data);
    void set_device();
    void device_barrier();
    void get_qstate(QStat &state);

    void reset(size_t qn);

    virtual ~DeviceQPU();
private:
    int m_device_id = { 0 };
    device_qsize_t m_qubit_num{0};
    device_state_t m_device_state;
    device_state_t m_device_matrix;
    device_qubit_t m_device_qubits;

    const size_t m_max_qubit_num = 64;
    const size_t m_max_matrix_size = 1024;

    qstate_type* m_reduce_buffer;
    std::vector<bool> m_peer_access;
    int m_device_num{ 0 };
    cudaStream_t m_cuda_stream{nullptr};
    std::map<GateType, std::shared_ptr<BaseGateFun>> m_type_gate_fun;
    std::shared_ptr<MeasureFun> m_measure_fun;
    std::shared_ptr<NormlizeFun> m_norm_fun;
};


#endif // GPU_GATE_WRAPPER_H





















