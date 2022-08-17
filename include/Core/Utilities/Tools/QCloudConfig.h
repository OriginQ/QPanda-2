#ifndef QCLOUD_CONFIG_H
#define QCLOUD_CONFIG_H
#include <string>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Module/DataStruct.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "ThirdParty/rabbit/rabbit.hpp"

QPANDA_BEGIN

#define MAX_FULL_AMPLITUDE_QUBIT_NUM 35
#define MAX_PARTIAL_AMPLITUDE_QUBIT_NUM 64

#define  QCLOUD_BATCH_COMPUTE_API_POSTFIX  "/taskApi/debug/submitTask.json"
#define  QCLOUD_BATCH_INQUIRE_API_POSTFIX  "/taskApi/debug/getTaskResultById.json"

#define  QCLOUD_COMPUTE_API_POSTFIX  "/api/taskApi/submitTask.json"
#define  QCLOUD_INQUIRE_API_POSTFIX  "/api/taskApi/getTaskDetail.json"

#define DEFAULT_CLUSTER_COMPUTEAPI    "http://pyqanda-admin.qpanda.cn/api/taskApi/submitTask.json"
#define DEFAULT_CLUSTER_INQUIREAPI     "http://pyqanda-admin.qpanda.cn/api/taskApi/getTaskDetail.json"

#define DEFAULT_REAL_CHIP_TASK_COMPUTEAPI     "https://qcloud.originqc.com.cn/api/taskApi/submitTask.json"
#define DEFAULT_REAL_CHIP_TASK_INQUIREAPI      "https://qcloud.originqc.com.cn/api/taskApi/getTaskDetail.json"

enum class CloudQMchineType : uint32_t
{
    Full_AMPLITUDE,
    NOISE_QMACHINE,
    PARTIAL_AMPLITUDE,
    SINGLE_AMPLITUDE,
    CHEMISTRY,
    REAL_CHIP,
    QST,
    FIDELITY
};

enum class RealChipType : uint32_t
{
    ORIGIN_WUYUAN_D3 = 7, //wuyuan no.3
    ORIGIN_WUYUAN_D4 = 5, //wuyuan no.2
    ORIGIN_WUYUAN_D5 = 2 //wuyuan no.1
};

enum class TaskStatus
{
    WAITING = 1,
    COMPUTING,
    FINISHED,
    FAILED,
    QUEUING,

    //The next status only appear in real chip backend
    SENT_TO_BUILD_SYSTEM,
    BUILD_SYSTEM_ERROR,
    SEQUENCE_TOO_LONG,
    BUILD_SYSTEM_RUN
};

enum class ReasultType
{
    PROBABILITY_MAP = 1, //full amplitude / noise -> measure/pmeasure
    MULTI_AMPLITUDE, //partial amplitude
    SINGLE_AMPLTUDE, //single amplitude
    SINGLE_PROBABILITY, //expectation, fidelity
};

struct NoiseConfigs
{
    std::string noise_model;
    double single_gate_param;
    double double_gate_param;

    double single_p2;
    double double_p2;

    double single_pgate;
    double double_pgate;
};


std::string to_string_array(const Qnum values);
std::string to_string_array(const std::vector<std::string> values);

void real_chip_task_validation(int shots, QProg &prog);

void params_verification(std::string dec_amplitude, size_t qubits_num);
void params_verification(std::vector<std::string> dec_amplitudes, size_t qubits_num);

size_t recv_json_data(void *ptr, size_t size, size_t nmemb, void *stream);

void json_string_transfer_encoding(std::string& str);

void construct_multi_prog_json(QuantumMachine* qvm, rabbit::array& code_array, size_t& code_len, std::vector<QProg>& prog_array);



void construct_cluster_task_json(
    rabbit::document& doc, 
    std::string prog_str,
    std::string token, 
    size_t qvm_type,
    size_t qubit_num,
    size_t cbit_num,
    size_t measure_type,
    std::string name);

void construct_real_chip_task_json(
    rabbit::document& doc,
    std::string prog_str,
    std::string token,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    size_t qvm_type,
    size_t qubit_num,
    size_t cbit_num,
    size_t measure_type,
    size_t shot,
    size_t chip_id,
    std::string name);

std::string hamiltonian_to_json(const QHamiltonian& hamiltonian);
QHamiltonian json_to_hamiltonian(const std::string& hamiltonian_json);

QPANDA_END

#endif // ! QCLOUD_CONFIG_H
