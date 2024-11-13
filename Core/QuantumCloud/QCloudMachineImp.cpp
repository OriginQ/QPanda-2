
#include <cctype> 
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/QuantumCloud/QCloudMachineImp.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "bz2/bzlib.h"

#include "QPandaConfig.h"

#if defined(USE_CURL)

using namespace std;
USING_QPANDA

static std::string random_num_process(const std::string& hex_str) 
{
    if (!std::all_of(hex_str.begin(), hex_str.end(), [](char c) { return std::isxdigit(c); })) 
        throw std::invalid_argument("Provided string is not a valid hexadecimal string");

    std::string processed_data;
    size_t length = hex_str.length();

    if (length > 192) 
    {
        processed_data = hex_str.substr(0, 192);
    }
    else if (length < 192) 
    {
        processed_data = hex_str + std::string(192 - length, '0');
    }
    else 
    {
        processed_data = hex_str;
    }

    return processed_data;
}

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

void QCloudMachineImp::object_init(uint32_t qbits_num, uint32_t cbits_num, std::string& name)
{
    m_object.clear();

    m_object.insert("apiKey", m_user_token);

    m_object.insert("taskFrom", 4);
    m_object.insert("taskName", name);

    return;
}


QCloudMachineImp::QCloudMachineImp() {}


QCloudMachineImp::~QCloudMachineImp() {}

void QCloudMachineImp::init_pqc_api(std::string url)
{
    m_pqc_compute_url = url + PQC_COMPUTE_API;
    m_pqc_inquire_url = url + PQC_INQUIRE_API;

    m_pqc_batch_compute_url = url + PQC_BATCH_COMPUTE_API;
    m_pqc_batch_inquire_url = url + PQC_BATCH_INQUIRE_API;

    m_pqc_init_url = url + PQC_INIT_API;
    m_pqc_keys_url = url + PQC_KEYS_API;
}

void QCloudMachineImp::init(std::string user_token,
                            bool is_logged /* = false */,
                            bool use_bin_or_hex_format /* = false */,
                            bool enable_pqc_encryption,
                            std::string random_num)
{
    m_curl.init(user_token);
    m_user_token = user_token;
    m_use_bin_or_hex_format = use_bin_or_hex_format;

    m_enable_pqc_encryption = enable_pqc_encryption;
    m_random_num = random_num_process(random_num);

    if (is_logged)
        QCloudLogger::get_instance().enable();

    std::string compute_api_postfix, inquire_api_postfix;
    if (user_token.find('/') != std::string::npos)
    {
        compute_api_postfix = DEFAULT_OQCS_COMPUTE_API_POSTFIX;
        inquire_api_postfix = DEFAULT_OQCS_INQUIRE_API_POSTFIX;
    }
    else
    {
        //do noting, use default POSTFIX
        compute_api_postfix = DEFAULT_COMPUTE_API_POSTFIX;
        inquire_api_postfix = DEFAULT_INQUIRE_API_POSTFIX;
    }

    m_compute_url = std::string(DEFAULT_URL) + compute_api_postfix;
    m_inquire_url = std::string(DEFAULT_URL) + inquire_api_postfix;
    m_estimate_url = std::string(DEFAULT_URL) + std::string(DEFAULT_ESTIMATE_API_POSTFIX);

    m_batch_compute_url = std::string(DEFAULT_URL) + BATCH_COMPUTE_API_POSTFIX;
    m_big_data_batch_compute_url = std::string(DEFAULT_URL) + BIG_DATA_BATCH_COMPUTE_API_POSTFIX;
    m_batch_inquire_url = std::string(DEFAULT_URL) + BATCH_INQUIRE_API_POSTFIX;

    m_chip_config_url = std::string(DEFAULT_URL) + std::string(CHIP_CONFIG_API_POSTFIX);

    init_pqc_api(std::string(DEFAULT_URL));
    return;
}

void QCloudMachineImp::set_qcloud_url(std::string cloud_url)
{
    if (m_user_token.find('/') != std::string::npos)
    {
        m_compute_url = cloud_url + DEFAULT_OQCS_COMPUTE_API_POSTFIX;
        m_inquire_url = cloud_url + DEFAULT_OQCS_INQUIRE_API_POSTFIX;
    }
    else
    {
        //do noting, use default postfix
        m_compute_url = cloud_url + DEFAULT_COMPUTE_API_POSTFIX;
        m_inquire_url = cloud_url + DEFAULT_INQUIRE_API_POSTFIX;
    }

    m_batch_compute_url = cloud_url + BATCH_COMPUTE_API_POSTFIX;
    m_batch_inquire_url = cloud_url + BATCH_INQUIRE_API_POSTFIX;
    m_big_data_batch_compute_url = cloud_url + BIG_DATA_BATCH_COMPUTE_API_POSTFIX;
    m_estimate_url = cloud_url + std::string(DEFAULT_ESTIMATE_API_POSTFIX);

    init_pqc_api(cloud_url);
    return;
}

void QCloudMachineImp::parse_submit_json(std::string& taskid, const std::string& submit_recv_string)
{
    try
    {
        QCLOUD_LOG_INFO("qcloud submit json recv : " + submit_recv_string);

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

            QCLOUD_LOG_INFO("qcloud taskid : " + taskid);
        }
    }
    catch (const std::exception& e) 
    {
        throw QCloudException(QCloudExceptionType::JSON_PARSE_ERROR, e.what());
    }

    return;
}

double QCloudMachineImp::parse_estimate_json(const std::string& estimate_recv_string)
{
    try
    {
        QCLOUD_LOG_INFO("qcloud estimate json recv : " + estimate_recv_string);

        rabbit::document doc;
        doc.parse(estimate_recv_string);

        if (!doc["success"].as_bool())
        {
            auto message = doc["message"].as_string();
            throw std::runtime_error(message.c_str());
        }
        else
        {
            auto estimate_value_str = doc["obj"].as_string();
            auto estimate_value = std::stof(estimate_value_str.c_str());

            QCLOUD_LOG_INFO("qcloud estimate price : " + estimate_value_str);

            return estimate_value;
        }
    }
    catch (const std::exception& e)
    {
        throw QCloudException(QCloudExceptionType::JSON_PARSE_ERROR, e.what());
    }
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

void QCloudMachineImp::query_result_json(std::string& taskid, std::vector<std::string>& result_string_array)
{
    bool is_retry_again = false;

    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        rabbit::object obj;
        obj.insert("taskId", taskid);
        obj.insert("apiKey", m_user_token);

        m_curl.post(m_batch_inquire_url, obj.str());

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
            m_measure_qubits_num.clear();
            if (!list[0].has("measureQubitSize"))
                QCERR_AND_THROW(run_fail, "measureQubitSize not found");

            for (size_t i = 0; i < list[0]["measureQubitSize"].size(); i++)
            {
                auto qubit_addr = list[0]["measureQubitSize"][i].as_int();
                m_measure_qubits_num.emplace_back(qubit_addr);
            }

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

            QCLOUD_LOG_INFO("qcloud result string : " + result_string);

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

    std::string state_str = doc["obj"]["taskStatus"].as_string();

    auto status = static_cast<TaskStatus>(atoi(state_str.c_str()));

    switch ((TaskStatus)status)
    {
        case TaskStatus::FINISHED:
        {
            is_retry_again = false;

            m_measure_qubits_num.clear();
            if (!doc["obj"].has("measureQubitSize"))
                QCERR_AND_THROW(run_fail, "measureQubitSize not found");

            for (size_t i = 0; i < doc["obj"]["measureQubitSize"].size(); i++)
            {
                auto qubit_addr = doc["obj"]["measureQubitSize"][i].as_int();
                m_measure_qubits_num.emplace_back(qubit_addr);
            }

             
            if (m_enable_pqc_encryption)
            {
#if defined(USE_QHETU) 

                auto result_string = sm4_decode(m_sym_key, m_iv, doc["obj"]["taskResult"].str());
                
                rabbit::document result_doc;
                result_doc.parse(result_string);

                for (size_t i = 0; i < result_doc.size(); i++)
                    result_array.emplace_back(result_doc[i].str());
#else
                QCERR_AND_THROW(std::runtime_error, " QHETU Lib not found, enable pqc encryption failed.");
#endif
            }
            else
            {
                for (size_t i = 0; i < doc["obj"]["taskResult"].size(); i++)
                {
                    auto result_string = doc["obj"]["taskResult"][i].as_string();
                    result_array.emplace_back(result_string);
                }
            }

            break;
        }

        case TaskStatus::FAILED:
        {
            if (doc["obj"].has("errorDetail"))
            {
                QCERR_AND_THROW(run_fail, doc["obj"]["errorDetail"].as_string());
            }
            else
            {
                QCERR_AND_THROW(run_fail, "task failed");
            }
        }
        default: 
        {
            is_retry_again = true;
            return;
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

    std::map<std::string, double> origin_result;
    sumbit_and_query(m_object.str(), origin_result);

    result = origin_result;
    //result = convert_map_format(origin_result, m_measure_qubits_num[0]);
    return;
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

    std::map<std::string, double> origin_result;
    sumbit_and_query(m_object.str(), origin_result);

    result = origin_result;
    //result = convert_map_format(origin_result, m_measure_qubits_num[0]);

    return;
}

void QCloudMachineImp::execute_full_amplitude_pmeasure(
    std::map<std::string, double>& result,
    Qnum qubits)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);
    object_append("qubits", to_string_array(qubits));

    std::map<std::string, double> origin_result;
    sumbit_and_query(m_object.str(), origin_result);

    result = origin_result;
    //result = convert_map_format(origin_result, m_measure_qubits_num[0]);

    return;
}

void QCloudMachineImp::execute_partial_amplitude_pmeasure(
    std::map<std::string, qcomplex_t>& result,
    std::vector<std::string> amplitudes)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::PARTIAL_AMPLITUDE);
    object_append("Amplitude", to_string_array(amplitudes));

    std::map<std::string, qcomplex_t> origin_result;
    sumbit_and_query(m_object.str(), origin_result);

    result = origin_result;
    //result = convert_map_format(origin_result, m_measure_qubits_num[0]);


    return;
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


void QCloudMachineImp::execute_error_mitigation(
    std::vector<double>& result,
    int shots,
    RealChipType chip_id,
    std::vector<std::string> expectations,
    const std::vector<double>& noise_strength,
    EmMethod qem_method)
{
    rabbit::array noise_strength_array;
    object_append_em_args(chip_id, expectations, noise_strength, noise_strength_array, qem_method);

    if (qem_method == EmMethod::ZNE)
        m_object.insert("noiseStrength", noise_strength_array);

    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::REAL_CHIP);
    object_append("shot", (size_t)shots);

    return sumbit_and_query(m_object.str(), result);
}

void QCloudMachineImp::read_out_error_mitigation(
    std::map<std::string, double>& result,
    int shots,
    RealChipType chip_id,
    std::vector<std::string> expectations,
    const std::vector<double>& noise_strength,
    EmMethod qem_method)
{
    rabbit::array noise_strength_array;
    object_append_em_args(chip_id, expectations, noise_strength, noise_strength_array, qem_method);

    if (qem_method == EmMethod::ZNE)
        m_object.insert("noiseStrength", noise_strength_array);

    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::REAL_CHIP);
    object_append("shot", (size_t)shots);

    std::map<std::string, double> origin_result;
    sumbit_and_query(m_object.str(), origin_result);

    result = convert_map_format(origin_result, m_measure_qubits_num[0]);

    return;
}

void QCloudMachineImp::execute_real_chip_measure(
    std::map<std::string, double>& result,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization)
{
    object_append_chip_args(chip_id, 
        is_amend, 
        is_mapping, 
        is_optimization);

    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::REAL_CHIP);
    object_append("shot", (size_t)shots);

    std::map<std::string, double> origin_result;
    sumbit_and_query(m_object.str(), origin_result);

    result = convert_map_format(origin_result, m_measure_qubits_num[0]);

    return;
}

void QCloudMachineImp::execute_get_state_tomography_density(
    std::vector<QStat>& result,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization)
{
    object_append_chip_args(chip_id, 
        is_amend, 
        is_mapping, 
        is_optimization);

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
    object_append_chip_args(chip_id,
        is_amend,
        is_mapping,
        is_optimization);

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

    std::vector<std::map<std::string, double>> origin_result;
    batch_sumbit_and_query(m_object.str(), origin_result);

    result.clear();
    for (size_t i = 0; i < origin_result.size(); i++)
        result.emplace_back(origin_result[i]);
        //result.emplace_back(convert_map_format(origin_result[i], m_measure_qubits_num[i]));

    return;
}

void QCloudMachineImp::execute_full_amplitude_pmeasure_batch(
    std::vector<std::map<std::string, double>>& result,
    std::vector<std::string>& prog_vector,
    Qnum qubits)
{
    object_append("qubits", to_string_array(qubits));
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);

    std::vector<std::map<std::string, double>> origin_result;
    batch_sumbit_and_query(m_object.str(), origin_result);

    result.clear();
    for (size_t i = 0; i < origin_result.size(); i++)
        result.emplace_back(origin_result[i]);
        //result.emplace_back(convert_map_format(origin_result[i], m_measure_qubits_num[i]));

    return;
}

void QCloudMachineImp::execute_partial_amplitude_pmeasure_batch(
    std::vector<std::map<std::string, qcomplex_t>>& result,
    std::vector<std::string>& prog_vector,
    std::vector<std::string> amplitudes)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::PARTIAL_AMPLITUDE);
    object_append("Amplitude", to_string_array(amplitudes));

    std::vector<std::map<std::string, qcomplex_t>> origin_result;
    batch_sumbit_and_query(m_object.str(), origin_result);

    result.clear();
    for (size_t i = 0; i < origin_result.size(); i++)
        result.emplace_back(origin_result[i]);
        //result.emplace_back(convert_map_format(origin_result[i], m_measure_qubits_num[i]));

    return;
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

    std::vector<std::map<std::string, double>> origin_result;
    batch_sumbit_and_query(m_object.str(), origin_result);

    result.clear();
    for (size_t i = 0; i < origin_result.size(); i++)
        result.emplace_back(origin_result[i]);
        //result.emplace_back(convert_map_format(origin_result[i], m_measure_qubits_num[i]));

    return;
}

std::string QCloudMachineImp::compress_data(std::string submit_string)
{
    const uint32_t compress_buf_size = 1024 * 1024 * 64;

    std::vector<char> compress_buf(compress_buf_size, 0);

    unsigned int compress_output_len = compress_buf_size;
    const auto compress_ret = BZ2_bzBuffToBuffCompress(
        compress_buf.data(),
        &compress_output_len,
        const_cast<char*>(submit_string.c_str()),
        submit_string.length(),
        9,
        0,
        0);

    std::string compress_data;
    if (compress_ret == BZ_OK)
    {
        QCLOUD_LOG_INFO("bz2 compress succeeded.");
        compress_buf.resize(compress_output_len);
        compress_data = std::string(compress_buf.begin(), compress_buf.end());
    }
    else
    {
        QCLOUD_LOG_INFO("bz2 compress failed: " + compress_ret);
        compress_data = submit_string;
    }

    return compress_data;
}

void QCloudMachineImp::pqc_init()
{
#if defined(USE_QHETU) 

    auto pqc_init_url = m_pqc_init_url;
    m_curl.get(pqc_init_url);
    auto pqc_data = m_curl.get_response_body();

    rabbit::document doc;
    doc.parse(pqc_data);

    if (!doc["success"].as_bool())
    {
        auto message = doc["message"].as_string();
        throw std::runtime_error(message.c_str());
    }
    else
    {
        auto pk0_value = doc["obj"]["pk0"].str();
        auto pkId_value = doc["obj"]["pkId"].str();
        auto enc_data = enc_hybrid(pk0_value, m_random_num);

        rabbit::object obj;
        obj.insert("cipher1", enc_data[0]);
        obj.insert("cipher2", enc_data[1]);
        obj.insert("pkId", pkId_value);

        m_curl.post(m_pqc_keys_url, obj.str());
        auto pqc_init_msg = m_curl.get_response_body();

        rabbit::document pqc_init_doc;
        pqc_init_doc.parse(pqc_init_msg);

        if (!pqc_init_doc["success"].as_bool())
        {
            auto init_err_message = pqc_init_doc["message"].as_string();
            throw std::runtime_error(init_err_message.c_str());
        }

        m_sym_key = enc_data[2];
        m_iv = enc_data[3];

        QCLOUD_LOG_INFO("qcloud pqc m_sym_key : " + m_sym_key);
        QCLOUD_LOG_INFO("qcloud pqc m_iv : " + m_iv);
    }

#else

    QCERR_AND_THROW(std::runtime_error, " QHETU Lib not found, enable pqc encryption failed.");

#endif

}


std::string QCloudMachineImp::async_execute_real_chip_measure_batch(
        std::vector<std::string>& prog_vector,
        int shot, 
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization)
{
    rabbit::array code_array;

    size_t code_array_bytes = 0;
    for (auto i = 0; i < prog_vector.size(); ++i)
    {
        code_array_bytes += prog_vector[i].size();
        code_array.push_back(prog_vector[i]);
    }

    object_append_chip_args(chip_id,
        is_amend,
        is_mapping,
        is_optimization);

    object_append("qmachineType", (size_t)CloudQMchineType::REAL_CHIP);
    //object_append("qprogArr", code_array);
    object_append("shot", (size_t)shot);

    if (m_enable_pqc_encryption)
    {
#if defined(USE_QHETU) 

        if (!m_pqc_init_completed)
        {
            pqc_init();
            m_pqc_init_completed = true;
        }

        auto pqc_data = sm4_encode(m_sym_key, m_iv, code_array.str());

        object_append("qprogStr", pqc_data);

        auto task_id = submit(m_object.str(), true);
        return task_id;

#else
        QCERR_AND_THROW(std::runtime_error, " QHETU Lib not found, enable pqc encryption failed.");

#endif
    }

    if (code_array_bytes >= 1024 * 1024)
    {
        auto temp_batch_url = m_batch_compute_url;
        m_batch_compute_url = m_big_data_batch_compute_url;

        rabbit::object big_data_doc;
        big_data_doc.insert("QProg", code_array);

        auto compress_data_str = compress_data(big_data_doc.str());

        //m_object.erase("qprogArr");
        std::string configuration_header_data = "configuration: " + m_object.str();

        m_curl.set_curl_header(configuration_header_data);

        auto task_id = submit(compress_data_str, true);

        m_curl.del_curl_header("configuration");
        m_batch_compute_url = temp_batch_url;
        return task_id;
    }
    else
    {
        object_append("qprogArr", code_array);
        auto task_id = submit(m_object.str(), true);
        return task_id;
    }

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
    auto batch_id = async_execute_real_chip_measure_batch(prog_vector,
        shot,
        chip_id,
        is_amend,
        is_mapping,
        is_optimization);

    std::vector<std::map<std::string, double>> origin_result;
    origin_result = query_batch_state_result(batch_id, true);

    result.clear();
    for (size_t i = 0; i < origin_result.size(); i++)
        result.emplace_back(convert_map_format(origin_result[i], m_measure_qubits_num[i]));

    return;
}


double QCloudMachineImp::estimate_price(size_t qubit_num,
    size_t shot,
    size_t qprogCount,
    size_t epoch)
{
    object_append("qubitNum", (size_t)qubit_num);
    object_append("shot", (size_t)shot);
    object_append("qprogCount", (size_t)qprogCount);
    object_append("epoch", (size_t)epoch);

    return get_estimate_price(m_object.str());
}


#if defined(USE_QHETU) 
#include "QHetu/qhetu.h"

std::string QCloudMachineImp::sm4_encode(std::string_view key, std::string_view IV, std::string_view data)
{
    return QHetu::sm4_enc(key, IV, data);
}
std::string QCloudMachineImp::sm4_decode(std::string_view key, std::string_view IV, std::string_view enc_data)
{
    return QHetu::sm4_dec(key, IV, enc_data);
}
std::vector<std::string> QCloudMachineImp::enc_hybrid(std::string_view pk_str, std::string& rdnum)
{
    auto [c_text1, c_text2, sym_key, sym_IV] = QHetu::enc_hybrid(pk_str, rdnum);
    std::vector<std::string> enc_datas = { c_text1, c_text2, sym_key, sym_IV };
    return enc_datas;
}

#endif

#endif