
#include "Core/Utilities/Tools/Utils.h"
#include "Core/QuantumCloud/QCloudLog.h"
#include "Core/QuantumCloud/Signature.h"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/QuantumCloud/QCloudMachineImp.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"

#include "QPandaConfig.h"
#if defined(USE_OPENSSL) && defined(USE_CURL)

using namespace std;
USING_QPANDA

static std::string to_string_array(const Qnum values)
{
    std::string string_array;
    for (auto val : values)
    {
        string_array.append(to_string(val));
        if (val != values.back())
        {
            string_array.append(",");
        }
    }

    return string_array;
}


static std::string to_string_array(const std::vector<string> values)
{
    std::string string_array;
    for (auto val : values)
    {
        string_array.append(val);
        if (val != values.back())
        {
            string_array.append(",");
        }
    }

    return string_array;
}

void QCloudMachineImp::object_init(uint32_t qbits_num, uint32_t cbits_num, std::string& prog, std::string& name)
{
    m_object.clear();

    m_object.insert("apiKey", m_user_token);

    m_object.insert("code", prog);
    m_object.insert("codeLen", prog.size());

    m_object.insert("taskFrom", 4);
    m_object.insert("qubitNum", qbits_num);
    m_object.insert("classicalbitNum", cbits_num);

    m_object.insert("taskName", name);

    return;
}

void QCloudMachineImp::object_init(uint32_t qbits_num, uint32_t cbits_num, std::vector<std::string>& prog_array, std::string& name)
{
    m_object.clear();

    std::vector<string> originir_array;

    size_t code_len = 0;
    for (auto& val : prog_array)
        code_len += val.size();

    rabbit::array code_array;
    for (auto i = 0; i < prog_array.size(); ++i)
    {
        rabbit::object code_value;
        code_value.insert("code", originir_array[i]);
        code_value.insert("id", (size_t)i);
        code_value.insert("step", (size_t)i);
        code_value.insert("breakPoint", "0");
        code_value.insert("isNow", (size_t)!i);
        code_array.push_back(code_value);
    }

    m_object.insert("codeArr", code_array);
    m_object.insert("apiKey", m_user_token);

    m_object.insert("codeLen", to_string(code_len));
    m_object.insert("qubitNum", qbits_num);
    m_object.insert("classicalbitNum", cbits_num);
    m_object.insert("taskFrom", 4);
    m_object.insert("taskName", name);

    return;
}


QCloudMachineImp::QCloudMachineImp()
{
    //OpenSSL_add_all_algorithms();
}


QCloudMachineImp::~QCloudMachineImp()
{
    //EVP_cleanup();
}

void QCloudMachineImp::init(std::string user_token, bool is_logged /* = false */)
{
    m_curl.init();
    m_user_token = user_token;

    if (is_logged)
        QCloudLogger::get_instance().enable();

    std::string compute_api_postfix, inquire_api_postfix;
    if (user_token.find('/') != std::string::npos)
    {
        //set curl header
        auto signature = qcloud_signature(user_token);
        m_curl.set_curl_header(signature);

        compute_api_postfix = DEFAULT_OQCS_COMPUTE_API_POSTFIX;
        inquire_api_postfix = DEFAULT_OQCS_INQUIRE_API_POSTFIX;
    }
    else
    {
        //do noting, use default POSTFIX
        compute_api_postfix = DEFAULT_COMPUTE_API_POSTFIX;
        inquire_api_postfix = DEFAULT_INQUIRE_API_POSTFIX;
    }

    JsonConfigParam config;
    if (!config.load_config(CONFIG_PATH))
    {
        m_compute_url = std::string(DEFAULT_URL) + compute_api_postfix;
        m_inquire_url = std::string(DEFAULT_URL) + inquire_api_postfix;

        LOG_WARNING("load config failed use default compute_url : " + m_compute_url);
        LOG_WARNING("load config failed use default inquire_url : " + m_inquire_url);
    }
    else
    {
        std::map<string, string> QCloudConfig;
        bool is_success = config.getQuantumCloudConfig(QCloudConfig);
        if (!is_success)
        {
            m_compute_url = std::string(DEFAULT_URL) + compute_api_postfix;
            m_inquire_url = std::string(DEFAULT_URL) + inquire_api_postfix;
        }
        else
        {
            set_qcloud_api(QCloudConfig["QCloudAPI"]);
        }
    }

    return;
}

void QCloudMachineImp::set_qcloud_api(std::string cloud_url)
{
    if (m_user_token.find('/') != std::string::npos)
    {
        m_compute_url = cloud_url + DEFAULT_OQCS_COMPUTE_API_POSTFIX;
        m_inquire_url = cloud_url + DEFAULT_OQCS_INQUIRE_API_POSTFIX;
    }
    else
    {
        //do noting, use default POSTFIX
        m_compute_url = cloud_url + DEFAULT_COMPUTE_API_POSTFIX;
        m_inquire_url = cloud_url + DEFAULT_INQUIRE_API_POSTFIX;
    }

    m_batch_compute_url = cloud_url + BATCH_COMPUTE_API_POSTFIX;
    m_batch_inquire_url = cloud_url + BATCH_INQUIRE_API_POSTFIX;

    return;
}

void QCloudMachineImp::check_and_update_signature()
{
    auto current_time = std::time(nullptr);
    std::time_t time_diff = current_time - m_signature_time;

    if (time_diff >= 100) 
    {
        auto signature = qcloud_signature(m_user_token);

        LOG_INFO("Signature: " + signature);

        m_curl.update_curl_header("Authorization", signature);
        m_signature_time = current_time;
    }

    return;
}


void QCloudMachineImp::parse_submit_json(std::string& taskid,const std::string& submit_recv_string)
{
    try
    {
        LOG_INFO("qcloud submit json recv : " + submit_recv_string);

        rabbit::document doc;
        doc.parse(submit_recv_string);

        if (!doc["success"].as_bool())
        {
            auto message = doc["message"].as_string();
            throw std::runtime_error(message.c_str());
        }
        else
        {
            taskid = doc["obj"]["taskId"].as_string();

            LOG_INFO("qcloud taskid : " + taskid);
        }
    }
    catch (const std::exception& e) 
    {
        throw QCloudException(QCloudExceptionType::JSON_PARSE_ERROR, e.what());
    }

    return;
}

void QCloudMachineImp::query_result_json(std::string& taskid, std::string& result_recv_string)
{
    bool is_retry_again = false;

    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        rabbit::object obj;

        obj.insert("taskId", taskid);
        obj.insert("apiKey", m_user_token);

        check_and_update_signature();
        m_curl.post(m_inquire_url, obj.str());

        cyclic_query(m_curl.get_response_body(), is_retry_again, result_recv_string);

    } while (is_retry_again);

    return;
}

void QCloudMachineImp::parse_submit_json(std::map<size_t, std::string>& taskid_map, const std::string& submit_recv_string)
{
    rabbit::document doc;

    try
    {
        doc.parse(submit_recv_string);

        if (!doc["success"].as_bool())
        {
            auto message = doc["message"].as_string();
            throw std::runtime_error(message.c_str());
        }

        for (auto i = 0; i < doc["obj"]["stepTaskResultList"].size(); ++i)
        {
            auto step_id = doc["obj"]["stepTaskResultList"][i]["step"].as_string();
            auto task_id = doc["obj"]["stepTaskResultList"][i]["taskId"].as_string();

            taskid_map.insert(make_pair(stoi(step_id), task_id));
        }
    }
    catch (const exception& e)
    {
        throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, e.what());
    }
}

void QCloudMachineImp::query_result_json(std::map<size_t, std::string>& taskid_map, std::vector<std::string>& result_string_array)
{
    bool is_retry_again = false;

    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        std::string string_array;
        for (auto val : taskid_map)
            string_array.append(val.second).append(";");

        rabbit::object obj;
        obj.insert("taskIds", string_array);
        obj.insert("apiKey", m_user_token);

        check_and_update_signature();
        m_curl.post(m_inquire_url, obj.str());

        cyclic_query(m_curl.get_response_body(), is_retry_again, result_string_array);

    } while (is_retry_again);

    return;
}

void QCloudMachineImp::cyclic_query(const std::string& recv_json, bool& is_retry_again, std::string& result_string)
{
    rabbit::document recv_doc;
    recv_doc.parse(recv_json.c_str());

    if (!recv_doc["success"].as_bool())
    {
        auto message = recv_doc["message"].as_string();
        throw QCloudException(QCloudExceptionType::JSON_PARSE_ERROR, message);
    }

    auto list = recv_doc["obj"]["qcodeTaskNewVo"]["taskResultList"];
    std::string state = list[0]["taskState"].as_string();
    std::string qtype = list[0]["rQMachineType"].as_string();

    auto status = static_cast<TaskStatus>(atoi(state.c_str()));
    auto backend_type = static_cast<CloudQMchineType>(atoi(qtype.c_str()));
    switch (status)
    {
        case TaskStatus::FINISHED:
        {
            is_retry_again = false;

            switch (backend_type)
            {
            case CloudQMchineType::REAL_CHIP:
            case CloudQMchineType::NOISE_QMACHINE:
            case CloudQMchineType::Full_AMPLITUDE:
            case CloudQMchineType::PARTIAL_AMPLITUDE:
            case CloudQMchineType::SINGLE_AMPLITUDE:
                result_string = list[0]["taskResult"].as_string();
                break;

            case CloudQMchineType::QST:
                result_string = list[0]["qstresult"].as_string();
                break;

            case CloudQMchineType::FIDELITY:
            {
                rabbit::object obj;
                obj.insert("value", stod(list[0]["qstfidelity"].as_string()));

                result_string = obj.str();
                break;
            }

            default: throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, "wrong backend type");
            }

            LOG_INFO("qcloud result string : " + result_string);

            break;
        }
    
        case TaskStatus::FAILED:
            if (!list[0].has("errorMessage"))
                throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, "wrong status : unknown error");
            else
                throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, list[0]["errorMessage"].str());

        //The next status only appear in real chip backend
        case TaskStatus::BUILD_SYSTEM_ERROR: 
            throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, "real chip build system error");

        case TaskStatus::SEQUENCE_TOO_LONG:
            throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, "real chip maximum timing sequence");

        default:
        {
            is_retry_again = true;
            return;
        }
    }

    return;
}

void QCloudMachineImp::cyclic_query(const std::string& recv_json, bool& is_retry_again, std::vector<std::string>& result_array)
{
    rabbit::document doc;
    doc.parse(recv_json);

    if (!doc["success"].as_bool())
    {
        auto message = doc["message"].as_string();
        throw QCloudException(QCloudExceptionType::TASK_PROCESS_ERROR, message);
    }

    for (int i = 0; i < doc["obj"].size(); i++)
    {
        auto step = doc["obj"][i]["step"].as_string();
        auto stat = doc["obj"][i]["taskState"].as_string();

        switch ((TaskStatus)stoi(stat))
        {
        case TaskStatus::FINISHED:
        {
            auto result_string = doc["obj"][i]["taskResult"].as_string();
            result_array.emplace_back(result_string);
            break;
        }

        case TaskStatus::BUILD_SYSTEM_ERROR: QCERR_AND_THROW(run_fail, "build system error");
        case TaskStatus::SEQUENCE_TOO_LONG: QCERR_AND_THROW(run_fail, "exceeding maximum timing sequence");
        case TaskStatus::FAILED: QCERR_AND_THROW(run_fail, "task failed");
        default: break;;
        }
    }

    return;
}

void QCloudMachineImp::execute_full_amplitude_measure(
    std::map<std::string, double>& result,
    int shots)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);
    object_append("shot", (size_t)shots);

    return sumbit_and_query(m_object.str(), result);
}
 
void QCloudMachineImp::execute_noise_measure(
    std::map<std::string, double>& result,
    int shots,
    NoiseConfigs noisy_args)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::NOISE_QMACHINE);

    object_append("shot", (size_t)shots);
    object_append("noisemodel", noisy_args.noise_model);
    object_append("singleGate", noisy_args.single_gate_param);
    object_append("doubleGate", noisy_args.double_gate_param);

    if ("DECOHERENCE_KRAUS_OPERATOR" == noisy_args.noise_model)
    {
        object_append("singleP2", noisy_args.single_p2);
        object_append("doubleP2", noisy_args.double_p2);
        object_append("singlePgate", noisy_args.single_pgate);
        object_append("doublePgate", noisy_args.double_pgate);
    }

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_full_amplitude_pmeasure(
    std::map<std::string, double>& result,
    Qnum qubits)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);
    object_append("qubits", to_string_array(qubits));

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_partial_amplitude_pmeasure(
    std::map<std::string, qcomplex_t>& result,
    std::vector<std::string> amplitudes)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::PARTIAL_AMPLITUDE);
    object_append("Amplitude", to_string_array(amplitudes));

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_single_amplitude_pmeasure(
    qcomplex_t& result,
    std::string amplitude)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::SINGLE_AMPLITUDE);
    object_append("Amplitude", amplitude);

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_real_chip_measure(
    std::map<std::string, double>& result,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization)
{
    object_append_chip_args(chip_id, is_amend, is_mapping, is_optimization);

    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::REAL_CHIP);
    object_append("shot", (size_t)shots);

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_get_state_tomography_density(
    std::vector<QStat>& result,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization)
{
    object_append_chip_args(chip_id, is_amend, is_mapping, is_optimization);

    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::QST);
    object_append("shot", (size_t)shots);

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_get_state_fidelity(
    double& result,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization)
{
    object_append_chip_args(chip_id, is_amend, is_mapping, is_optimization);

    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::FIDELITY);
    object_append("shot", (size_t)shots);

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_get_expectation(
    double& result,
    const QHamiltonian& hamiltonian,
    const Qnum& qubits)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_EXPECTATION);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);

    object_append("qubits", to_string_array(qubits));
    object_append("hamiltonian", hamiltonian_to_json(hamiltonian));

    return sumbit_and_query(m_object.str(), result);
}


void QCloudMachineImp::execute_full_amplitude_measure_batch(
    std::vector<std::map<std::string, double>>& result,
    std::vector<std::string> & prog_vector,
    int shots)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);
    object_append("shot", (size_t)shots);

    return batch_sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_full_amplitude_pmeasure_batch(
    std::vector<std::map<std::string, double>>& result,
    std::vector<std::string>& prog_vector,
    Qnum qubits)
{
    object_append("qubits", to_string_array(qubits));
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);

    return batch_sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_partial_amplitude_pmeasure_batch(
    std::vector<std::map<std::string, qcomplex_t>>& result,
    std::vector<std::string>& prog_vector,
    std::vector<std::string> amplitudes)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::PARTIAL_AMPLITUDE);
    object_append("Amplitude", to_string_array(amplitudes));

    return batch_sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_single_amplitude_pmeasure_batch(
    std::vector<qcomplex_t>& result,
    std::vector<std::string>& prog_vector,
    std::string amplitude)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::SINGLE_AMPLITUDE);
    object_append("Amplitude", amplitude);

    return batch_sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_noise_measure_batch(
    std::vector<std::map<std::string, double>>& result,
    std::vector<std::string>& prog_vector,
    int shots,
    NoiseConfigs noisy_args)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::NOISE_QMACHINE);

    object_append("shot", (size_t)shots);
    object_append("noisemodel", noisy_args.noise_model);
    object_append("singleGate", noisy_args.single_gate_param);
    object_append("doubleGate", noisy_args.double_gate_param);

    if ("DECOHERENCE_KRAUS_OPERATOR" == noisy_args.noise_model)
    {
        object_append("singleP2", noisy_args.single_p2);
        object_append("doubleP2", noisy_args.double_p2);
        object_append("singlePgate", noisy_args.single_pgate);
        object_append("doublePgate", noisy_args.double_pgate);
    }

    return batch_sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::execute_real_chip_measure_batch(
    std::vector<std::map<std::string, double>>& result,
    std::vector<std::string>& prog_vector,
    int shot, 
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization)
{
    object_append_chip_args(chip_id, is_amend, is_mapping, is_optimization);

    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::REAL_CHIP);
    object_append("shot", (size_t)shot);

    return batch_sumbit_and_query(m_object.str(), result);
}

#endif