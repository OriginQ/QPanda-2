#pragma once

#include <ctime>
#include <string>
#include "Core/QuantumCloud/QCurl.h"
#include "Core/QuantumCloud/QRabbit.h"

#include "Core/Module/DataStruct.h"
#include "Core/Utilities/QPandaNamespace.h"

#include "QPandaConfig.h"
#if defined(USE_OPENSSL) && defined(USE_CURL)

QPANDA_BEGIN

#define  BATCH_COMPUTE_API_POSTFIX "/taskApi/debug/submitTask.json"
#define  BATCH_INQUIRE_API_POSTFIX "/taskApi/debug/getTaskResultById.json"

#define  DEFAULT_COMPUTE_API_POSTFIX "/api/taskApi/submitTask.json"
#define  DEFAULT_INQUIRE_API_POSTFIX "/api/taskApi/getTaskDetail.json"

#define  DEFAULT_OQCS_COMPUTE_API_POSTFIX "/qcal/oqcs/task/submitTask.json"
#define  DEFAULT_OQCS_INQUIRE_API_POSTFIX "/qcal/oqcs/task/getTaskDetailForQpanda.json"

#define  DEFAULT_URL "http://pyqanda-admin.qpanda.cn"

enum ClusterTaskType : uint32_t
{
    CLUSTER_MEASURE = 1,
    CLUSTER_PMEASURE = 2,
    CLUSTER_EXPECTATION
};

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
    ORIGIN_WUYUAN_D5 = 2  //wuyuan no.1
};

enum class TaskStatus : int
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

enum class QCloudExceptionType
{
    CURL_REQUEST_ERROR,
    JSON_PARSE_ERROR,
    TASK_PROCESS_ERROR
};

class QCloudException : public std::exception 
{
public:

    QCloudException(QCloudExceptionType type, const std::string& message)
    {
        switch (type)
        {
        case QPanda::QCloudExceptionType::CURL_REQUEST_ERROR:
            m_exception_message = "curl performed failed : " + message;
            break;
        case QPanda::QCloudExceptionType::JSON_PARSE_ERROR:
            m_exception_message = "json parse failed : " + message;
            break;
        case QPanda::QCloudExceptionType::TASK_PROCESS_ERROR:
            m_exception_message = "Task execution failed : " + message;
            break;
        default:
            break;
        }
    }

    virtual const char* what() const noexcept override 
    {
        return m_exception_message.c_str();
    }

private:
    std::string m_exception_message;
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

class QCloudMachineImp
{
public:

    QCloudMachineImp();
    ~QCloudMachineImp();

    void init(std::string user_token, bool is_logged = false);
    void set_qcloud_api(std::string cloud_url);

    void execute_full_amplitude_measure(
        std::map<std::string, double>& result,
        int shots);

    void execute_noise_measure(
        std::map<std::string, double>& result,
        int shots, 
        NoiseConfigs noisy_args);

    void execute_full_amplitude_pmeasure(
        std::map<std::string, double>& result,
        Qnum qubit_vec);

    void execute_partial_amplitude_pmeasure(
        std::map<std::string, qcomplex_t>& result,
        std::vector<std::string> amplitudes);

    void execute_single_amplitude_pmeasure(
        qcomplex_t& result,
        std::string amplitude);

    void execute_real_chip_measure(
        std::map<std::string, double>& result,
        int shots,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    void execute_get_state_tomography_density(
        std::vector<QStat>& result,
        int shot,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    void execute_get_state_fidelity(
        double& result,
        int shot,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    void execute_get_expectation(
        double& result,
        const QHamiltonian& hamiltonian,
        const Qnum& qubits);

    void execute_full_amplitude_measure_batch(
        std::vector<std::map<std::string, double>>& result,
        std::vector<std::string>& prog_vector,
        int shots);

    void execute_full_amplitude_pmeasure_batch(
        std::vector<std::map<std::string, double>>& result,
        std::vector<std::string>& prog_vector,
        Qnum qubits);

    void execute_partial_amplitude_pmeasure_batch(
        std::vector<std::map<std::string, qcomplex_t>>& result,
        std::vector<std::string>& prog_vector,
        std::vector<std::string> amplitudes);

    void execute_single_amplitude_pmeasure_batch(
        std::vector<qcomplex_t>& result,
        std::vector<std::string>& prog_vector,
        std::string amplitudes);

    void execute_noise_measure_batch(
        std::vector<std::map<std::string, double>>& result,
        std::vector<std::string>& prog_vector,
        int shots,
        NoiseConfigs noisy_args);

    void execute_real_chip_measure_batch(
        std::vector<std::map<std::string, double>>& result,
        std::vector<std::string>& prog_vector,
        int shots, 
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D3,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true);

public:

    template <typename T>
    void sumbit_and_query(std::string submit_json, T& result)
    {
        //update signature use ecdsa with sh256
        check_and_update_signature();

        //post execute task json to qcloud
        m_curl.post(m_compute_url, submit_json);

        //parse taskid from submit json result
        std::string taskid;
        parse_submit_json(taskid, m_curl.get_response_body());

        //post inquire result json to qcloud
        std::string result_string;
        query_result_json(taskid, result_string);

        //parse task result from submit json result
        parse_result<T>(result_string, result);

        return;
    }

    template <typename T>
    void batch_sumbit_and_query(std::string submit_json, std::vector<T>& result_array)
    {
        //update signature use ecdsa with sh256
        check_and_update_signature();

        //post execute task json to qcloud
        m_curl.post(m_batch_compute_url, submit_json);

        //parse taskid from batch job submit json result
        std::map<size_t, std::string> taskid_map;
        parse_submit_json(taskid_map, m_curl.get_response_body());

        //post inquire result json to qcloud
        std::vector<std::string> result_string_array;
        query_result_json(taskid_map, result_string_array);

        //parse task result from submit json result
        for (auto i = 0; i < result_string_array.size(); ++i)
        {
            T result;
            parse_result<T>(result_string_array[i], result);
            result_array.emplace_back(result);
        }

        return;
    }

public:

    void parse_submit_json(std::string& taskid, const std::string& submit_recv_string);
    void parse_submit_json(std::map<size_t, std::string>& taskid, const std::string& submit_recv_string);

    void query_result_json(std::string& taskid, std::string& result_string);
    void query_result_json(std::map<size_t, std::string>& taskid_map, std::vector<std::string>& result_string_array);

    void cyclic_query(const std::string& recv_json, bool& is_retry_again, std::string& result_string);
    void cyclic_query(const std::string& recv_json, bool& is_retry_again, std::vector<std::string>& result_array);

    void object_init(uint32_t qbits_num, uint32_t cbits_num, std::string& prog, std::string& name);
    void object_init(uint32_t qbits_num, uint32_t cbits_num, std::vector<std::string>& prog_array, std::string& name);

    template<typename T>
    void object_append(const std::string& key, const T& value)
    {
        m_object.insert(key, value);
    }

    void object_append_chip_args(RealChipType chip_id, bool is_amend, bool is_mapping, bool is_optimization)
    {
        m_object.insert("chipId", (size_t)chip_id);
        m_object.insert("isAmend", (int)!is_amend);
        m_object.insert("mappingFlag", (int)!is_mapping);
        m_object.insert("circuitOptimization", (int)!is_optimization);
        return;
    }

    void check_and_update_signature();

private:

    //lib curl
    QCurl m_curl;

    //rabbit object for submit task
    rabbit::object m_object;

    //origin qcloud user token 
    std::string m_user_token;

    //cloud url
    std::string m_inquire_url;
    std::string m_compute_url;

    std::string m_batch_inquire_url;
    std::string m_batch_compute_url;

    //time_t for update signature
    std::time_t m_signature_time = std::time(nullptr);
};

QPANDA_END

#endif

