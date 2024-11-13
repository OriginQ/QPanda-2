#pragma once

#include <ctime>
#include <string>
#include <thread>
#include "Core/QuantumCloud/QCurl.h"
#include "Core/QuantumCloud/QRabbit.h"
#include "Core/QuantumCloud/QCloudLog.h"

#include "Core/Module/DataStruct.h"
#include "Core/Utilities/QPandaNamespace.h"

#include "QPandaConfig.h"

QPANDA_BEGIN

#define  CHIP_CONFIG_API_POSTFIX "/api/taskApi/getFullConfig.json"

#define  BATCH_COMPUTE_API_POSTFIX "/oqcs/batch/submitTask.json"
#define  BIG_DATA_BATCH_COMPUTE_API_POSTFIX "/oqcs/batch/submitDaTask.json"

#define  BATCH_INQUIRE_API_POSTFIX "/oqcs/batch/taskInfo.json"

#define  DEFAULT_COMPUTE_API_POSTFIX "/api/taskApi/submitTask.json"
#define  DEFAULT_INQUIRE_API_POSTFIX "/api/taskApi/getTaskDetail.json"

#define  DEFAULT_ESTIMATE_API_POSTFIX "/oqcs/task/estimate.json"
#define  DEFAULT_OQCS_COMPUTE_API_POSTFIX "/oqcs/task/submitTask.json"
#define  DEFAULT_OQCS_INQUIRE_API_POSTFIX "/oqcs/task/getTaskDetailForQpanda.json"

#define  PQC_INIT_API "/oqcs/task/getPublicKeyQpanda.json"
#define  PQC_KEYS_API "/oqcs/task/generateKeyQpanda.json"

#define  PQC_COMPUTE_API "/oqcs/task/decrySubmitTaskQpanda.json"
#define  PQC_INQUIRE_API "/oqcs/task/encryTaskDetailQpanda.json"

#define  PQC_BATCH_COMPUTE_API "/oqcs/batch/decrySubmitTaskQpadan.json"
#define  PQC_BATCH_INQUIRE_API "/oqcs/batch/encryTaskInfoQpadan.json"

#define  DEFAULT_URL "http://pyqanda-admin.qpanda.cn"

enum ClusterTaskType : uint32_t
{
    CLUSTER_MEASURE = 1,
    CLUSTER_PMEASURE = 2,
    CLUSTER_EXPECTATION
};

enum EmMethod : int
{
    ZNE = 0,
    PEC = 1,
    READ_OUT = 2
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
    ORIGIN_WUYUAN_D5 = 2, //wuyuan no.1
    ORIGIN_72 = 72  //wuyuan no.1
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

#if defined(USE_CURL)

class QCloudMachineImp
{
public:

    QCloudMachineImp();
    ~QCloudMachineImp();

    void init(std::string user_token, 
        bool is_logged, 
        bool use_bin_or_hex_format, 
        bool m_enable_pqc_encryption,
        std::string m_random_num);

    bool is_enable_pqc_encryption() { return m_enable_pqc_encryption; }

    void set_qcloud_url(std::string cloud_url);

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

    void execute_error_mitigation(
        std::vector<double>& result,
        int shots,
        RealChipType chip_id,
        std::vector<std::string> expectations,
        const std::vector<double>& noise_strength,
        EmMethod qemMethod);

    void read_out_error_mitigation(
        std::map<std::string, double>& result,
        int shots,
        RealChipType chip_id,
        std::vector<std::string> expectations,
        const std::vector<double>& noise_strength,
        EmMethod qem_method);

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
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    std::string async_execute_real_chip_measure_batch(
        std::vector<std::string>& prog_vector,
        int shots,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    double estimate_price(size_t qubit_num,
        size_t shot,
        size_t qprogCount,
        size_t epoch);

    double parse_estimate_json(const std::string& estimate_recv_string);

public:

    void init_pqc_api(std::string url);

    std::string submit(std::string submit_json, bool is_batch_task = false)
    {
        if (is_batch_task)
        {
            if (m_enable_pqc_encryption)
                m_curl.post(m_pqc_batch_compute_url, submit_json);
            else
                m_curl.post(m_batch_compute_url, submit_json);
        }
        else
        {
            m_curl.post(m_compute_url, submit_json);
        }

        //parse taskid from submit json result
        std::string taskid;
        parse_submit_json(taskid, m_curl.get_response_body());

        return taskid;
    }

    double get_estimate_price(std::string estimate_json)
    {
        m_curl.post(m_estimate_url, estimate_json);
        return parse_estimate_json(m_curl.get_response_body());
    }

    std::map<std::string, double> query_state_result(std::string task_id)
    {
        if (m_enable_pqc_encryption)
        {
            auto batch_result = query_batch_state_result(task_id);
            return batch_result.empty() ? std::map<std::string, double>() : batch_result[0];
        }

        rabbit::object obj;

        obj.insert("taskId", task_id);
        obj.insert("apiKey", m_user_token);

        m_curl.post(m_inquire_url, obj.str());

        bool is_retry_again = false;
        std::string result_recv_string;
        cyclic_query(m_curl.get_response_body(), is_retry_again, result_recv_string);

        std::map<std::string, double> temp_result;
        if (is_retry_again)
            std::cout << "Task " << task_id << " Running..." << std::endl;
        else
            parse_result<std::map<std::string, double>>(result_recv_string, temp_result);

        auto result = convert_map_format(temp_result, m_measure_qubits_num[0]);
        return result;
    }

    std::vector<std::map<std::string, double>> query_batch(std::string task_id, bool open_loop)
    {
        rabbit::object obj;

        obj.insert("taskId", task_id);

        auto inquire_url = m_enable_pqc_encryption ? m_pqc_batch_inquire_url : m_batch_inquire_url;

        m_curl.post(inquire_url, obj.str());

        bool is_retry_again = false;
        std::vector<std::string> result_recv_string;
        cyclic_query(m_curl.get_response_body(), is_retry_again, result_recv_string);

        std::vector<std::map<std::string, double>> result;

        if (open_loop)
        {
            while (is_retry_again)
            {
                m_curl.post(inquire_url, obj.str());
                cyclic_query(m_curl.get_response_body(), is_retry_again, result_recv_string);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
        else
        {
            if (is_retry_again)
            {
                std::string task_status_info = "Task " + task_id + " Running...";
                QCLOUD_LOG_INFO(task_status_info);
            }
        }

        if (!is_retry_again)
        {
            for (auto i = 0; i < result_recv_string.size(); ++i)
            {
                std::map<std::string, double> temp_result;
                parse_result<std::map<std::string, double>>(result_recv_string[i], temp_result);

                result.emplace_back(convert_map_format(temp_result, m_measure_qubits_num[i]));
            }
        }

        return result;
    }

    std::vector<std::map<std::string, double>> query_batch_state_result(std::string task_id, bool open_loop = false)
    {
        if (!m_pqc_init_completed  && m_enable_pqc_encryption)
        {
            pqc_init();
            m_pqc_init_completed = true;
        }

        int retry = 0;
        while (retry < 20)
        {
            try
            {
                return query_batch(task_id, open_loop);
            }
            catch (const std::exception& e)
            {
                std::string error_message = "Catch exception : " + std::string(e.what())  + ",retry request attempts : " + std::to_string(++retry);
                QCLOUD_LOG_WARNING(error_message);
            }
            catch (...)
            {
                std::string error_message = "Catch exception, retry request attempts : " + std::to_string(++retry);
                QCLOUD_LOG_WARNING(error_message);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }   

        if (retry >= 20)
        {
            std::string error_message = "Max attempts reached. unable to complete the query task.";
            throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, error_message);
        }
    }

    template <typename T>
    void sumbit_and_query(std::string submit_json, T& result)
    {
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
        //post execute task json to qcloud
        m_curl.post(m_batch_compute_url, submit_json);

        //parse taskid from batch job submit json result
        std::string taskid;
        parse_submit_json(taskid, m_curl.get_response_body());

        //post inquire result json to qcloud
        std::vector<std::string> result_string_array;
        query_result_json(taskid, result_string_array);

        //parse task result from submit json result
        for (auto i = 0; i < result_string_array.size(); ++i)
        {
            T result;
            parse_result<T>(result_string_array[i], result);
            result_array.emplace_back(result);
        }

        return;
    }

    std::string compress_data(std::string submit_string);
    std::string object_string() { return m_object.str(); }

public:

    void parse_submit_json(std::string& taskid, const std::string& submit_recv_string);
    void parse_submit_json(std::map<size_t, std::string>& taskid, const std::string& submit_recv_string);
     
    void query_result_json(std::string& taskid, std::string& result_string);
    void query_result_json(std::string& taskid, std::vector<std::string>& result_string_array);

    void cyclic_query(const std::string& recv_json, bool& is_retry_again, std::string& result_string);
    void cyclic_query(const std::string& recv_json, bool& is_retry_again, std::vector<std::string>& result_array);

    void object_init(uint32_t qbits_num, uint32_t cbits_num, std::string& name);
    void object_init(uint32_t qbits_num, uint32_t cbits_num, std::string& prog, std::string& name);

    template<typename T>
    void object_append(const std::string& key, const T& value)
    {
        m_object.insert(key, value);
    }

    void object_append_chip_args(RealChipType chip_id, 
        bool is_amend, 
        bool is_mapping, 
        bool is_optimization)
    {
        m_object.insert("chipId", (size_t)chip_id);
        m_object.insert("isAmend", is_amend);
        m_object.insert("mappingFlag", is_mapping);
        m_object.insert("circuitOptimization", is_optimization);
        //m_object.insert("compileLevel", compile_level);
        return;
    }

    void object_append_em_args(RealChipType chip_id, 
        std::vector<std::string> expectations,
        const std::vector<double>& noise_strength,
        rabbit::array& noise_strength_array,
        EmMethod qem_method)
    {
        switch (qem_method)
        {
        case QPanda::ZNE:
        {
            rabbit::array exp_array;
            for (auto i = 0; i < expectations.size(); ++i)
                exp_array.push_back(expectations[i]);

            for (auto i = 0; i < noise_strength.size(); ++i)
                noise_strength_array.push_back(noise_strength[i]);

            m_object.insert("expectations", exp_array.str());
            //m_object.insert("noiseStrength", noise_strength_array);
            break;
        }

        case QPanda::PEC:
        case QPanda::READ_OUT:
        {
            rabbit::array exp_array, noise_strength_array;
            for (auto i = 0; i < expectations.size(); ++i)
                exp_array.push_back(expectations[i]);

            m_object.insert("expectations", exp_array.str());
            break;
        }

        default:
            break;
        }

        m_object.insert("chipId", (size_t)chip_id);
        m_object.insert("isEm", 1);
        m_object.insert("qemMethod", qem_method);
        
        return;
    }

    template <typename T>
    std::map<std::string, T> convert_map_format(const std::map<std::string, T>& input_map, size_t digit_num)
    {
        if (m_use_bin_or_hex_format) //use bin format
        {
            auto is_hex = [](const std::string& str)
            {
                size_t pos = (str.size() > 2 && str.substr(0, 2) == "0X") ? 2 : 0;
                return str.find_first_not_of("0123456789abcdefABCDEF", pos) == std::string::npos;
            };

            auto hex_to_binary = [](const std::string& hex_str, size_t binary_length)
            {
                size_t pos = (hex_str.size() > 2 && hex_str.substr(0, 2) == "0X") ? 2 : 0;
                std::string hex_digits = hex_str.substr(pos);

                std::stringstream ss;
                ss << std::hex << hex_digits;
                unsigned long hex_value;
                ss >> hex_value;

                // Construct std::bitset with specified size
                std::bitset<128> bits(hex_value);

                // Convert to binary string
                std::string binary_str = bits.to_string();

                if(binary_length > binary_str.length())
                    binary_str = std::string(binary_length - binary_str.length(), '0') + binary_str;
                else
                    binary_str = std::string(binary_str.end() - binary_length, binary_str.end());

                // Ensure the binary string has the specified length by padding with leading zeros
                return binary_str;
            };

            std::map<std::string, T> result;

            for (const auto& entry : input_map)
            {
                std::string key = entry.first;
                std::transform(key.begin(), key.end(), key.begin(), ::toupper);

                if (is_hex(key))
                    key = hex_to_binary(key, digit_num);

                result[key] = entry.second;
            }

            return result;
        }
        else //use hex format
        {
            auto is_binary = [](const std::string& str)
            {
                return str.find_first_not_of("01") == std::string::npos;
            };

            auto binary_to_hex = [](const std::string& binary_str)
            {
                std::bitset<8> bits(std::stoi(binary_str, nullptr, 2));
                std::stringstream hex_stream;
                hex_stream << "0X" << std::hex << bits.to_ulong();
                return hex_stream.str();
            };

            std::map<std::string, T> result;

            for (const auto& entry : input_map)
            {
                std::string key = entry.first;

                if (is_binary(key))
                    key = binary_to_hex(key);

                result[key] = entry.second;
            }

            return result;
        }

        
    }

    void pqc_init();

#if defined(USE_QHETU) 

    std::string sm4_encode(std::string_view key, std::string_view IV, std::string_view data);
    std::string sm4_decode(std::string_view key, std::string_view IV, std::string_view enc_data);
    std::vector<std::string> enc_hybrid(std::string_view pk_str, std::string& rdnum);

#endif

private:

    //lib curl
    QCurl m_curl;

    //True  -> binary
    //False -> hex
    bool m_use_bin_or_hex_format;

    bool m_pqc_init_completed = false;
    bool m_enable_pqc_encryption = false;
    std::string m_random_num;

    //m_measure_qubits_num[0]  -> normal task measure_qubit_num
    //m_measure_qubits_num     -> batch  task measure_qubit_num
    Qnum m_measure_qubits_num;

    //rabbit object for submit task
    rabbit::object m_object;

    //origin qcloud user token 
    std::string m_user_token;

    //PQC Cryption
    std::string m_iv;
    std::string m_sym_key;

    std::string m_pqc_init_url;
    std::string m_pqc_keys_url;

    std::string m_pqc_inquire_url;
    std::string m_pqc_compute_url;
    std::string m_pqc_batch_inquire_url;
    std::string m_pqc_batch_compute_url;

    //cloud url
    std::string m_inquire_url;
    std::string m_estimate_url;
    std::string m_compute_url;

    std::string m_batch_inquire_url;
    std::string m_batch_compute_url;
    std::string m_big_data_batch_compute_url;

    //chip config
    std::string m_chip_config_url;
};

#endif

QPANDA_END


