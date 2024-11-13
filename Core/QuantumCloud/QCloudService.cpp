#include "Core/Core.h"
#include "bz2/bzlib.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/QuantumCloud/QCloudLog.h"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/QuantumCloud/QCloudService.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"

#include "QPandaConfig.h"

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

void QCloudService::object_init(uint32_t qbits_num, 
    uint32_t cbits_num, 
    std::string& prog, 
    std::string& name, 
    int task_form)
{
    m_object.clear();

    m_object.insert("apiKey", m_user_token);

    m_object.insert("code", prog);
    m_object.insert("codeLen", prog.size());

    m_object.insert("taskFrom", task_form);
    m_object.insert("qubitNum", qbits_num);
    m_object.insert("classicalbitNum", cbits_num);

    m_object.insert("taskName", name);

    return;
}

void QCloudService::object_init(uint32_t qbits_num, 
    uint32_t cbits_num, 
    std::vector<std::string>& prog_array, 
    std::string& name, 
    int task_form)
{
    m_object.clear();

    rabbit::array code_array;
    for (auto i = 0; i < prog_array.size(); ++i)
        code_array.push_back(prog_array[i]);

    object_append("qmachineType", (size_t)CloudQMchineType::REAL_CHIP);
    object_append("qprogArr", code_array);

    object_append("taskFrom", task_form);
    object_append("taskName", name);

    return;
}


QCloudService::QCloudService() {}


QCloudService::~QCloudService() {}


void QCloudService::init_pqc_api(std::string url)
{
    m_pqc_compute_url = url + PQC_COMPUTE_API;
    m_pqc_inquire_url = url + PQC_INQUIRE_API;

    m_pqc_batch_compute_url = url + PQC_BATCH_COMPUTE_API;
    m_pqc_batch_inquire_url = url + PQC_BATCH_INQUIRE_API;

    m_pqc_init_url = url + PQC_INIT_API;
    m_pqc_keys_url = url + PQC_KEYS_API;
}


void QCloudService::init(std::string user_token, 
                         bool is_logged /* = false */)
{
    QVM::init();
    _start();
    _QMachine_type = QMachineType::QCloud;

    m_user_token = user_token;

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
    m_batch_inquire_url = std::string(DEFAULT_URL) + BATCH_INQUIRE_API_POSTFIX;
    m_big_data_batch_compute_url = std::string(DEFAULT_URL) + BIG_DATA_BATCH_COMPUTE_API_POSTFIX;

    m_chip_config_url = std::string(DEFAULT_URL) + std::string(CHIP_CONFIG_API_POSTFIX);

    init_pqc_api(std::string(DEFAULT_URL));
    return;
}

static std::map<NOISE_MODEL, std::string> cloud_noise_model_mapping =
{
  {NOISE_MODEL::BITFLIP_KRAUS_OPERATOR,"BITFLIP_KRAUS_OPERATOR"},
  {NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR,"BIT_PHASE_FLIP_OPRATOR"},
  {NOISE_MODEL::DAMPING_KRAUS_OPERATOR,"DAMPING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR,"DECOHERENCE_KRAUS_OPERATOR"},
  {NOISE_MODEL::DEPHASING_KRAUS_OPERATOR,"DEPHASING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR,"DEPOLARIZING_KRAUS_OPERATOR"},
  {NOISE_MODEL::PHASE_DAMPING_OPRATOR,"PHASE_DAMPING_OPRATOR"}
};

void QCloudService::set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params)
{
    auto iter = cloud_noise_model_mapping.find(model);
    if (cloud_noise_model_mapping.end() == iter || single_params.empty() || double_params.empty())
        QCERR_AND_THROW(run_fail, "NOISE MODEL ERROR");

    m_noisy_args.noise_model = iter->second;
    m_noisy_args.single_gate_param = single_params[0];
    m_noisy_args.double_gate_param = double_params[0];

    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR == iter->first)
    {
        if (3 != single_params.size())
            QCERR_AND_THROW(run_fail, "DECOHERENCE_KRAUS_OPERATOR PARAM SIZE ERROR");

        if (3 != double_params.size())
            QCERR_AND_THROW(run_fail, "DECOHERENCE_KRAUS_OPERATOR PARAM SIZE ERROR");

        m_noisy_args.single_p2 = single_params[1];
        m_noisy_args.double_p2 = double_params[1];

        m_noisy_args.single_pgate = single_params[2];
        m_noisy_args.double_pgate = double_params[2];
    }

    return;
}



void QCloudService::set_qcloud_url(std::string cloud_url)
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
    m_estimate_url = cloud_url + DEFAULT_ESTIMATE_API_POSTFIX;
    m_big_data_batch_compute_url = cloud_url + BIG_DATA_BATCH_COMPUTE_API_POSTFIX;

    init_pqc_api(cloud_url);
    return;
}

void QCloudService::parse_submit_json(std::string& taskid, const std::string& submit_recv_string)
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

void QCloudService::parse_submit_json(std::map<size_t, std::string>& taskid_map, const std::string& submit_recv_string)
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



void QCloudService::cyclic_query(const std::string& recv_json, bool& is_retry_again, std::string& result_string)
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

void QCloudService::batch_cyclic_query(const std::string& recv_json, bool& is_retry_again, std::vector<std::string>& result_array)
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
            m_measure_qubits_num.clear();
            if (!doc["obj"].has("measureQubitSize"))
                QCERR_AND_THROW(run_fail, "measureQubitSize not found");

            for (size_t i = 0; i < doc["obj"]["measureQubitSize"].size(); i++)
            {
                auto qubit_addr = doc["obj"]["measureQubitSize"][i].as_int();
                m_measure_qubits_num.emplace_back(qubit_addr);
            }

            is_retry_again = false;

            for (size_t i = 0; i < doc["obj"]["taskResult"].size(); i++)
            {
                auto result_string = doc["obj"]["taskResult"][i].as_string();
                result_array.emplace_back(result_string);
            }

            break;
        }

        case TaskStatus::FAILED:
        {
            if (doc["obj"].has("errorMessage"))
            {
                std::string error_detail = "error message : " + doc["obj"]["errorMessage"].as_string();
                QCERR_AND_THROW(run_fail, error_detail);
            }
            else
            {
                QCERR_AND_THROW(run_fail, "task failed");
            }
        }
        case TaskStatus::BUILD_SYSTEM_ERROR: QCERR_AND_THROW(run_fail, "build system error");
        case TaskStatus::SEQUENCE_TOO_LONG: QCERR_AND_THROW(run_fail, "exceeding maximum timing sequence");
        default:
        {
            is_retry_again = true;
            return;
        }
    }

    return;
}


void QCloudService::build_init_object(QProg& prog, std::string task_name, int task_from)
{
    auto prog_info = count_prog_info(prog);
    if (prog_info.layer_num > 500)
        QCERR_AND_THROW(std::runtime_error, "The number of layers in the quantum circuit exceeds the limit");

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name, task_from);
}

void QCloudService::build_init_object(std::string& originir, std::string task_name, int task_from)
{
    object_init(getAllocateQubitNum(), getAllocateCMem(), originir, task_name, task_from);
}


void QCloudService::build_init_object_batch(std::vector<string>& prog_strings, 
    std::string task_name, 
    int task_from)
{
    object_init(getAllocateQubitNum(), getAllocateCMem(), prog_strings, task_name, task_from);
}


void QCloudService::build_init_object_batch(std::vector<QProg>& prog_vector, 
    std::string task_name,
    int task_from)
{
    //convert prog to originir 
    std::vector<string> prog_strs;

    for (size_t i = 0; i < prog_vector.size(); i++)
    {
        auto prog_info = count_prog_info(prog_vector[i]);
        if (prog_info.layer_num > 500)
            QCERR_AND_THROW(std::runtime_error, "The number of layers in the quantum circuit exceeds the limit");

        auto prog_str = convert_qprog_to_originir(prog_vector[i], this);
        prog_strs.emplace_back(prog_str);
    }

    object_init(getAllocateQubitNum(), getAllocateCMem(), prog_strs, task_name, task_from);
}


string QCloudService::build_full_amplitude_measure(int shots)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);
    object_append("shot", (size_t)shots);

    return m_object.str();
}
 
string QCloudService::build_noise_measure(int shots)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::NOISE_QMACHINE);

    object_append("shot", (size_t)shots);
    object_append("noisemodel", m_noisy_args.noise_model);
    object_append("singleGate", m_noisy_args.single_gate_param);
    object_append("doubleGate", m_noisy_args.double_gate_param);

    if ("DECOHERENCE_KRAUS_OPERATOR" == m_noisy_args.noise_model)
    {
        object_append("singleP2", m_noisy_args.single_p2);
        object_append("doubleP2", m_noisy_args.double_p2);
        object_append("singlePgate", m_noisy_args.single_pgate);
        object_append("doublePgate", m_noisy_args.double_pgate);
    }

    return m_object.str();
}

string QCloudService::build_full_amplitude_pmeasure(
    Qnum qubits)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);
    object_append("qubits", to_string_array(qubits));

    return m_object.str();
}

string QCloudService::build_partial_amplitude_pmeasure(
    std::vector<std::string> amplitudes)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::PARTIAL_AMPLITUDE);
    object_append("Amplitude", to_string_array(amplitudes));

    return m_object.str();
}

string QCloudService::build_single_amplitude_pmeasure(
    std::string amplitude)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    object_append("QMachineType", (size_t)CloudQMchineType::SINGLE_AMPLITUDE);
    object_append("Amplitude", amplitude);

    return m_object.str();
}


string QCloudService::build_error_mitigation(
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

    return m_object.str();
}

string QCloudService::build_read_out_mitigation(
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

    return m_object.str();
}

string QCloudService::build_real_chip_measure(
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

    return m_object.str();
}

string QCloudService::build_get_state_tomography_density(
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

    return m_object.str();
}

string QCloudService::build_get_state_fidelity(
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

    return m_object.str();
}

string QCloudService::build_get_expectation(
    const QHamiltonian& hamiltonian,
    const Qnum& qubits)
{
    object_append("measureType", (size_t)ClusterTaskType::CLUSTER_EXPECTATION);
    object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);

    object_append("qubits", to_string_array(qubits));
    object_append("hamiltonian", hamiltonian_to_json(hamiltonian));

    return m_object.str();
}

std::string QCloudService::compress_data(std::string submit_string)
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


string QCloudService::build_real_chip_measure_batch(
    std::vector<std::string>& prog_strs,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    bool enable_compress_check,
    std::string batch_id,
    int task_from)
{
    if (prog_strs.size() > 200)
        QCERR_AND_THROW(std::runtime_error, "Exceed batch job limit");

    m_object.clear();

    size_t code_array_bytes = 0;
    rabbit::array code_array;
    for (auto i = 0; i < prog_strs.size(); ++i)
    {
        code_array.push_back(prog_strs[i]);
        code_array_bytes += prog_strs[i].size();
    }

    object_append("qmachineType", (size_t)CloudQMchineType::REAL_CHIP);
    //object_append("qprogArr", code_array);
    object_append("taskFrom", task_from);
    object_append("batchNo", batch_id);
    object_append("shot", (size_t)shots);

    object_append_chip_args(chip_id,
        is_amend,
        is_mapping,
        is_optimization);

    if (enable_compress_check && (code_array_bytes >= 1024 * 1024))
    {
        QCLOUD_LOG_INFO("large batch data, enable data compression.");

        rabbit::object big_data_doc;
        big_data_doc.insert("QProg", code_array);

        m_use_compress_data = true;
        m_configuration_header_data = m_object.str();

        return big_data_doc.str();

    }
    else
    {
        m_use_compress_data = false;
        object_append("qprogArr", code_array);
        return m_object.str();
    }

}

string QCloudService::build_real_chip_measure_batch(
    std::vector<QProg>& prog_vector,
    int shots, 
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    bool enable_compress_check,
    std::string batch_id,
    int task_from)
{
    std::vector<string> prog_strs;
    for (size_t i = 0; i < prog_vector.size(); i++)
    {
        auto gate_num = prog_vector[i].get_qgate_num();

        QVec qubits;
        auto qubits_num = prog_vector[i].get_used_qubits(qubits);

        if (gate_num > (MAX_LAYER_LIMIT * qubits_num))
            QCERR_AND_THROW(std::runtime_error, "The number of layers in the quantum circuit exceeds the limit");

        prog_strs.emplace_back(convert_qprog_to_originir(prog_vector[i], this));
    }

    return build_real_chip_measure_batch(prog_strs,
        shots, 
        chip_id, 
        is_amend, 
        is_mapping,
        is_optimization,
        enable_compress_check,
        batch_id,
        task_from);
}


#if defined(USE_QHETU) 
#include "QHetu/qhetu.h"

std::string QCloudService::sm4_encode(std::string_view key, std::string_view IV, std::string_view data)
{
    return QHetu::sm4_enc(key, IV, data);
}
std::string QCloudService::sm4_decode(std::string_view key, std::string_view IV, std::string_view enc_data)
{
    return QHetu::sm4_dec(key, IV, enc_data);
}
std::vector<std::string> QCloudService::enc_hybrid(std::string_view pk_str, std::string& rdnum)
{
    auto [c_text1, c_text2, sym_key, sym_IV] = QHetu::enc_hybrid(pk_str, rdnum);
    std::vector<std::string> enc_datas = { c_text1, c_text2, sym_key, sym_IV };
    return enc_datas;
}

#endif


