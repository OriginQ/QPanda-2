#include <fstream>
#include <algorithm>
#include <string.h>
#include "Core/Core.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "ThirdParty/rabbit/rabbit.hpp"

USING_QPANDA
using namespace std;
using namespace Base64;
using namespace rapidjson;

static std::map<NOISE_MODEL, std::string> noise_model_mapping =
{
  {NOISE_MODEL::BITFLIP_KRAUS_OPERATOR,"BITFLIP_KRAUS_OPERATOR"},
  {NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR,"BIT_PHASE_FLIP_OPRATOR"},
  {NOISE_MODEL::DAMPING_KRAUS_OPERATOR,"DAMPING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR,"DECOHERENCE_KRAUS_OPERATOR"},
  {NOISE_MODEL::DEPHASING_KRAUS_OPERATOR,"DEPHASING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR,"DEPOLARIZING_KRAUS_OPERATOR"},
  //{NOISE_MODEL::KRAUS_MATRIX_OPRATOR,"KRAUS_MATRIX_OPRATOR"},
  //{NOISE_MODEL::MIXED_UNITARY_OPRATOR,"MIXED_UNITARY_OPRATOR"},
  //{NOISE_MODEL::PAULI_KRAUS_MAP,"PAULI_KRAUS_MAP"},
  {NOISE_MODEL::PHASE_DAMPING_OPRATOR,"PHASE_DAMPING_OPRATOR"}
};

static std::string  rabbit_json_extract(const rabbit::document& result_doc, std::string value)
{
    transform(value.begin(), value.end(), value.begin(), ::toupper);

    for (auto iter = result_doc.member_begin(); iter != result_doc.member_end(); ++iter)
    {
        std::string iter_name = iter->name();
        std::string iter_name_upper = iter->name();

        transform(iter_name_upper.begin(), iter_name_upper.end(), iter_name_upper.begin(), ::toupper);
        if (0 == strcmp(iter_name_upper.c_str(), value.c_str()))
            return std::string(iter_name);
    }

    QCERR_AND_THROW(std::runtime_error, "result json incorrect,no key or value found");
}

QCloudMachine::QCloudMachine()
{
#ifdef USE_CURL

    curl_global_init(CURL_GLOBAL_ALL);

    m_post_curl = curl_easy_init();
    m_headers = curl_slist_append(m_headers, "Content-Type: application/json;charset=UTF-8");
    m_headers = curl_slist_append(m_headers, "Connection: keep-alive");
    m_headers = curl_slist_append(m_headers, "Server: nginx/1.16.1");
    m_headers = curl_slist_append(m_headers, "Transfer-Encoding: chunked");
    m_headers = curl_slist_append(m_headers, "origin-language: en");

    curl_easy_setopt(m_post_curl, CURLOPT_HTTPHEADER, m_headers);
    curl_easy_setopt(m_post_curl, CURLOPT_TIMEOUT, 60);
    curl_easy_setopt(m_post_curl, CURLOPT_CONNECTTIMEOUT, 30);
    curl_easy_setopt(m_post_curl, CURLOPT_HEADER, 0);
    curl_easy_setopt(m_post_curl, CURLOPT_POST, 1);
    curl_easy_setopt(m_post_curl, CURLOPT_SSL_VERIFYHOST, 0);
    curl_easy_setopt(m_post_curl, CURLOPT_SSL_VERIFYPEER, 0);

    curl_easy_setopt(m_post_curl, CURLOPT_READFUNCTION, nullptr);
    curl_easy_setopt(m_post_curl, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(m_post_curl, CURLOPT_WRITEFUNCTION, recv_json_data);
#else
    QCERR_AND_THROW(run_fail, "Need support the curl libray");
#endif
}

QCloudMachine::~QCloudMachine()
{
#ifdef USE_CURL
    curl_slist_free_all(m_headers);
    curl_easy_cleanup(m_post_curl);
    curl_global_cleanup();
#else
    QCERR_AND_THROW(run_fail, "need support the curl libray");
#endif
}

void QCloudMachine::set_qcloud_api(std::string url)
{
    m_compute_url = url + QCLOUD_COMPUTE_API_POSTFIX;
    m_inquire_url = url + QCLOUD_INQUIRE_API_POSTFIX;

    m_batch_compute_url = url + QCLOUD_BATCH_COMPUTE_API_POSTFIX;
    m_batch_inquire_url = url + QCLOUD_BATCH_INQUIRE_API_POSTFIX;

    return;
}

void QCloudMachine::init(string token, bool is_logged)
{
    JsonConfigParam config;
    try
    {
        m_token = token;
        m_is_logged = is_logged;
        _start();

        if (!config.load_config(CONFIG_PATH))
        {
            if (m_is_logged) std::cout << "config warning: can not find config file, use default config" << endl;

            m_compute_url = DEFAULT_CLUSTER_COMPUTEAPI;
            m_inquire_url = DEFAULT_CLUSTER_INQUIREAPI;
        }
        else
        {
            std::map<string, string> QCloudConfig;
            bool is_success = config.getQuantumCloudConfig(QCloudConfig);
            if (!is_success)
            {
                if (m_is_logged) std::cout << "config warning: get quantum cloud config failed, use default config" << endl;

                m_compute_url = DEFAULT_CLUSTER_COMPUTEAPI;
                m_inquire_url = DEFAULT_CLUSTER_INQUIREAPI;
            }
            else
            {
                set_qcloud_api(QCloudConfig["QCloudAPI"]);
            }
        }
    }
    catch (std::exception &e)
    {
        finalize();

        if (m_is_logged) std::cout << "config warning: load config file catch exception, use default config" << endl;

        m_compute_url = DEFAULT_CLUSTER_COMPUTEAPI;
        m_inquire_url = DEFAULT_CLUSTER_INQUIREAPI;

        QCERR_AND_THROW(run_fail, e.what());
    }
}

void QCloudMachine::set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params)
{
    auto iter = noise_model_mapping.find(model);
    if (noise_model_mapping.end() == iter || single_params.empty() || double_params.empty())
    {
        QCERR_AND_THROW(run_fail, "NOISE MODEL ERROR");
    }

    m_noise_params.noise_model = iter->second;
    m_noise_params.single_gate_param = single_params[0];
    m_noise_params.double_gate_param = double_params[0];

    try
    {
        if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR == iter->first)
        {
            m_noise_params.single_p2 = single_params[1];
            m_noise_params.double_p2 = double_params[1];

            m_noise_params.single_pgate = single_params[2];
            m_noise_params.double_pgate = double_params[2];
        }
    }
    catch (std::exception &e)
    {
        QCERR_AND_THROW(run_fail, "DECOHERENCE_KRAUS_OPERATOR ERROR");
    }

    return;
}

std::string QCloudMachine::post_json(const std::string &sUrl, std::string & sJson)
{
#ifdef USE_CURL
    std::stringstream out;
    curl_easy_setopt(m_post_curl, CURLOPT_URL, sUrl.c_str());
    //curl_easy_setopt(m_post_curl, CURLOPT_VERBOSE, 1);
    curl_easy_setopt(m_post_curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(m_post_curl, CURLOPT_POSTFIELDS, sJson.c_str());
    curl_easy_setopt(m_post_curl, CURLOPT_POSTFIELDSIZE, sJson.size());

    CURLcode res;
    for (size_t i = 0; i < m_cur_reperform_time; i++)
    {
        res = curl_easy_perform(m_post_curl);
        if (CURLE_OK == res)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (CURLE_OK != res)
    {
        std::string error_msg = curl_easy_strerror(res);
        QCERR_AND_THROW(run_fail, error_msg);
    }

    try
    {
        return out.str();
    }
    catch (...)
    {
        if (m_is_logged) std::cout << out.str() << endl;
        QCERR_AND_THROW(run_fail, "catch exception in recv json");
    }
#else
    QCERR_AND_THROW(run_fail, "need support the curl libray");
#endif
}

void QCloudMachine::inquire_result(std::string recv_json_str, std::string url, CloudQMchineType type)
{
    std::string taskid;
    if (parser_submit_json(recv_json_str, taskid))
    {
        bool is_retry_again = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            auto result_json = get_result_json(taskid, url, type);

            is_retry_again = parser_result_json(result_json, taskid);

        } while (is_retry_again);
    }

    return;
}

double QCloudMachine::get_state_fidelity(
    QProg &prog,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    rabbit::document doc;
    doc.parse("{}");

    construct_real_chip_task_json(doc, prog_str, m_token, is_amend, is_mapping, is_optimization,
        (size_t)CloudQMchineType::FIDELITY, getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_MEASURE, shots, (size_t)chip_id, task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);
    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::FIDELITY);

    return m_qst_fidelity;
}


std::vector<QStat> QCloudMachine::get_state_tomography_density(
    QProg &prog,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rabbit::document doc;
    doc.parse("{}");
    construct_real_chip_task_json(doc, prog_str, m_token, is_amend, is_mapping, is_optimization,
        (size_t)CloudQMchineType::QST, getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_MEASURE, shots, (size_t)chip_id, task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::QST);

    return m_qst_result;
}

std::map<std::string, double> QCloudMachine::real_chip_measure(
    QProg &prog,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rabbit::document doc;
    doc.parse("{}");
    construct_real_chip_task_json(doc, prog_str, m_token, is_amend, is_mapping, is_optimization,
        (size_t)CloudQMchineType::REAL_CHIP, getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_MEASURE, shots, (size_t)chip_id, task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::REAL_CHIP);
    return m_measure_result;
}

std::map<std::string, double> QCloudMachine::noise_measure(QProg &prog, int shot, string task_name)
{
    //convert prog to originir
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rabbit::document doc;
    doc.parse("{}");

    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::NOISE_QMACHINE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_MEASURE, task_name);

    doc.insert("shot", (size_t)shot);
    doc.insert("noisemodel", m_noise_params.noise_model);
    doc.insert("singleGate", m_noise_params.single_gate_param);
    doc.insert("doubleGate", m_noise_params.double_gate_param);

    if ("DECOHERENCE_KRAUS_OPERATOR" == m_noise_params.noise_model)
    {
        doc.insert("singleP2", m_noise_params.single_p2);
        doc.insert("doubleP2", m_noise_params.double_p2);
        doc.insert("singlePgate", m_noise_params.single_pgate);
        doc.insert("doublePgate", m_noise_params.double_pgate);
    }

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::NOISE_QMACHINE);
    return m_measure_result;
}

std::map<std::string, double> QCloudMachine::full_amplitude_measure(QProg &prog, int shot, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rabbit::document doc;
    doc.parse("{}");

    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::Full_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_MEASURE, task_name);
    doc.insert("shot", (size_t)shot);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);
    return m_measure_result;
}

std::string QCloudMachine::full_amplitude_measure_commit(QProg &prog, int shot, TaskStatus& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rabbit::document doc;

    doc.parse("{}");
    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::Full_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_MEASURE, task_name);
    doc.insert("shot", (size_t)shot);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    try
    {
        std::string taskid;
        parser_submit_json(recv_json_str, taskid);
        status = TaskStatus::COMPUTING;
        return taskid;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return "";
    }
}

std::string QCloudMachine::full_amplitude_pmeasure_commit(QProg &prog, Qnum qubit_vec, TaskStatus& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rabbit::document doc;

    doc.parse("{}");
    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::Full_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_PMEASURE, task_name);
    doc.insert("qubits", to_string_array(qubit_vec));

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    try
    {
        std::string taskid;
        parser_submit_json(recv_json_str, taskid);
        status = TaskStatus::COMPUTING;
        return taskid;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return "";
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_measure_query(std::string taskid, TaskStatus& status)
{
    try
    {
        auto result_json = get_result_json(taskid, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);

        bool is_retry_again = parser_result_json(result_json, taskid);

        status = m_task_status;
        return is_retry_again ? std::map<std::string, double>() : m_measure_result;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return {};
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_measure_exec(std::string taskid, TaskStatus& status)
{
    try
    {
        bool is_retry_again = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            auto result_json = get_result_json(taskid, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);

            is_retry_again = parser_result_json(result_json, taskid);

        } while (is_retry_again);

        status = TaskStatus::FINISHED;
        return m_measure_result;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return {};
    }
}

std::map<std::string, qcomplex_t> QCloudMachine::full_amplitude_pmeasure_query(std::string taskid, TaskStatus& status)
{
    try
    {
        auto result_json = get_result_json(taskid, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);
        bool is_retry_again = parser_result_json(result_json, taskid);
        status = m_task_status;
        return is_retry_again ? std::map<std::string, qcomplex_t>() : m_pmeasure_result;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return {};
    }
}

std::map<std::string, qcomplex_t> QCloudMachine::full_amplitude_pmeasure_exec(std::string taskid, TaskStatus& status)
{
    try
    {
        bool is_retry_again = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            auto result_json = get_result_json(taskid, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);
            is_retry_again = parser_result_json(result_json, taskid);

        } while (is_retry_again);

        status = TaskStatus::FINISHED;
        return m_pmeasure_result;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return {};
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_pmeasure(QProg &prog, Qnum qubit_vec, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rabbit::document doc;

    doc.parse("{}");
    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::Full_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_PMEASURE, task_name);

    doc.insert("qubits", to_string_array(qubit_vec));

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);
    return m_measure_result;
}

std::map<std::string, qcomplex_t> QCloudMachine::partial_amplitude_pmeasure(QProg &prog, std::vector<std::string> amplitude_vec, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    params_verification(amplitude_vec, getAllocateQubitNum());

    //construct json
    rabbit::document doc;
    doc.parse("{}");

    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::PARTIAL_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_PMEASURE, task_name);
    doc.insert("Amplitude", to_string_array(amplitude_vec));

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::PARTIAL_AMPLITUDE);
    return m_pmeasure_result;
}

qcomplex_t QCloudMachine::single_amplitude_pmeasure(QProg &prog, std::string amplitude, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    params_verification(amplitude, getAllocateQubitNum());

    //construct json
    rabbit::document doc;
    doc.parse("{}");

    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::SINGLE_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_PMEASURE, task_name);

    doc.insert("Amplitude", amplitude);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::SINGLE_AMPLITUDE);
    return m_single_result;
}

double QCloudMachine::get_expectation(QProg prog, const QHamiltonian& hamiltonian, const QVec& qvec, TaskStatus& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    Qnum qubits;
    for (auto qubit : qvec)
    {
        qubits.emplace_back(qubit->get_phy_addr());
    }

    //construct json
    rabbit::document doc;
    doc.parse("{}");

    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::Full_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_EXPECTATION, task_name);

    doc.insert("qubits", to_string_array(qubits));
    doc.insert("hamiltonian", hamiltonian_to_json(hamiltonian));

    std::string post_json_str = doc.str();

    try
    {
        std::string recv_json_str = post_json(m_compute_url, post_json_str);

        inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);

        status = TaskStatus::FINISHED;
        return m_expectation;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return 0.;
    }
}

double QCloudMachine::get_expectation(QProg prog, const QHamiltonian& hamiltonian, const QVec& qvec, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    Qnum qubits;
    for (auto qubit : qvec)
        qubits.emplace_back(qubit->get_phy_addr());

    //construct json
    rabbit::document doc;
    doc.parse("{}");

    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::Full_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_EXPECTATION, task_name);

    doc.insert("qubits", to_string_array(qubits));
    doc.insert("hamiltonian", hamiltonian_to_json(hamiltonian));

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);
    return m_expectation;
}


std::string QCloudMachine::get_expectation_commit(QProg prog, const QHamiltonian& hamiltonian, const QVec& qvec, TaskStatus& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    Qnum qubits;
    for (auto qubit : qvec)
    {
        qubits.emplace_back(qubit->get_phy_addr());
    }

    //construct json
    rabbit::document doc;
    doc.parse("{}");
    construct_cluster_task_json(doc, prog_str, m_token, (size_t)CloudQMchineType::Full_AMPLITUDE,
        getAllocateQubitNum(), getAllocateCMem(),
        (size_t)ClusterTaskType::CLUSTER_EXPECTATION, task_name);

    doc.insert("qubits", to_string_array(qubits));
    doc.insert("hamiltonian", hamiltonian_to_json(hamiltonian));

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    try
    {
        std::string taskid;
        parser_submit_json(recv_json_str, taskid);
        status = TaskStatus::COMPUTING;
        return taskid;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return "";
    }
}

double QCloudMachine::get_expectation_query(std::string taskid, TaskStatus& status)
{
    try
    {
        auto result_json = get_result_json(taskid, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);
        bool is_retry_again = parser_result_json(result_json, taskid);

        status = m_task_status;
        return is_retry_again ? -1 : m_expectation;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return 0.;
    }
}

double QCloudMachine::get_expectation_exec(std::string taskid, TaskStatus& status)
{
    try
    {
        bool is_retry_again = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            auto result_json = get_result_json(taskid, m_inquire_url, CloudQMchineType::Full_AMPLITUDE);
            is_retry_again = parser_result_json(result_json, taskid);

        } while (is_retry_again);

        status = TaskStatus::FINISHED;
        return m_expectation;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return 0.;
    }
}

bool QCloudMachine::parser_submit_json(std::string &recv_json, std::string& taskid)
{
    try
    {
        rabbit::document doc;
        doc.parse(recv_json);

        auto success = doc["success"].as_bool();
        if (!success)
        {
            if (m_is_logged) std::cout << recv_json << std::endl;

            auto message = doc["message"].as_string();
            QCERR_AND_THROW(run_fail, message);
        }
        else
        {
            taskid = doc["obj"]["taskId"].as_string();
            return true;
        }
    }
    catch (const std::exception& e)
    {
        if (m_is_logged) std::cout << recv_json << std::endl;
        QCERR_AND_THROW(run_fail, e.what());
    }
}

bool QCloudMachine::parser_result_json(std::string &recv_json, std::string& taskid)
{
    json_string_transfer_encoding(recv_json);

    rabbit::document recv_doc;
    recv_doc.parse(recv_json.c_str());

    if (!recv_doc["success"].as_bool())
    {
        m_error_info = recv_doc["message"].as_string();
        QCERR_AND_THROW(run_fail, m_error_info);
    }

    try
    {
        auto list = recv_doc["obj"]["qcodeTaskNewVo"]["taskResultList"];
        std::string state = list[0]["taskState"].as_string();
        std::string qtype = list[0]["rQMachineType"].as_string();

        auto status = static_cast<TaskStatus>(atoi(state.c_str()));
        auto backend_type = static_cast<CloudQMchineType>(atoi(qtype.c_str()));
        switch (status)
        {
        case TaskStatus::FINISHED:
        {
            auto result_string = list[0]["taskResult"].as_string();

            rabbit::document result_doc;
            result_doc.parse(result_string.c_str());

            switch (backend_type)
            {
            case CloudQMchineType::REAL_CHIP:
            case CloudQMchineType::NOISE_QMACHINE:
            case CloudQMchineType::Full_AMPLITUDE:
            {
                if (result_doc.has("ResultType") && 
                    result_doc["ResultType"].as_int() == (int)ClusterResultType::EXPECTATION)
                {
                    auto exp_value_str = rabbit_json_extract(result_doc, "value");
                    
                    m_expectation = result_doc[exp_value_str.c_str()].is_double() ?
                        result_doc[exp_value_str.c_str()].as_double() : (double)result_doc[exp_value_str.c_str()].as_int();
                }
                else
                {
                    m_measure_result.clear();
                    std::vector<std::string> key_list;
                    std::vector<double> value_list;

                    auto key_string = rabbit_json_extract(result_doc, "key");
                    auto value_string = rabbit_json_extract(result_doc, "value");

                    for (auto i = 0; i < result_doc[key_string.c_str()].size(); ++i)
                        key_list.emplace_back(result_doc[key_string.c_str()][i].as_string());

                    for (auto i = 0; i < result_doc[value_string.c_str()].size(); ++i)
                    {
                        auto val = result_doc[value_string.c_str()][i].is_double() ?
                            result_doc[value_string.c_str()][i].as_double() : (double)result_doc[value_string.c_str()][i].as_int();
                        value_list.emplace_back(val);
                    }

                    if (key_list.size() != value_list.size())
                        QCERR_AND_THROW(std::runtime_error, "reasult json size incorrect");

                    for (size_t i = 0; i < key_list.size(); i++)
                        m_measure_result.insert(make_pair(key_list[i], value_list[i]));
                }

                break;
            }

            case CloudQMchineType::PARTIAL_AMPLITUDE:
            {
                m_pmeasure_result.clear();

                auto key_string = rabbit_json_extract(result_doc, "key");

                for (auto i = 0; i < result_doc[key_string.c_str()].size(); ++i)
                {
                    auto key = result_doc[key_string.c_str()][i].as_string();
                    auto val_real = result_doc["ValueReal"][i].is_double() ?
                        result_doc["ValueReal"][i].as_double() : (double)result_doc["ValueReal"][i].as_int();
                    auto val_imag = result_doc["ValueImag"][i].is_double() ?
                        result_doc["ValueImag"][i].as_double() : (double)result_doc["ValueImag"][i].as_int();

                    m_pmeasure_result.insert(make_pair(key, qcomplex_t(val_real, val_imag)));
                }

                break;
            }

            case CloudQMchineType::SINGLE_AMPLITUDE:
            {
                auto val_real = result_doc["ValueReal"][0].is_double() ?
                    result_doc["ValueReal"][0].as_double() : (double)result_doc["ValueReal"][0].as_int();
                auto val_imag = result_doc["ValueImag"][0].is_double() ?
                    result_doc["ValueImag"][0].as_double() : (double)result_doc["ValueImag"][0].as_int();

                m_single_result = qcomplex_t(val_real, val_imag);
                break;
            }

            case CloudQMchineType::QST:
            {
                rabbit::document qst_result_doc;
                qst_result_doc.parse(list[0]["qstresult"].as_string());

                m_qst_result.clear();
                int rank = (int)std::sqrt(qst_result_doc.size());

                for (auto i = 0; i < rank; ++i)
                {
                    QStat row_value;
                    for (auto j = 0; j < rank; ++j)
                    {
                        auto qst_result_real_value = qst_result_doc[i*rank + j]["r"];
                        auto qst_result_imag_value = qst_result_doc[i*rank + j]["i"];

                        auto real_val = qst_result_real_value.is_double() ?
                            qst_result_real_value.as_double() : (double)qst_result_real_value.as_int();
                        auto imag_val = qst_result_imag_value.is_double() ?
                            qst_result_imag_value.as_double() : (double)qst_result_imag_value.as_int();

                        row_value.emplace_back(qcomplex_t(real_val, imag_val));
                    }

                    m_qst_result.emplace_back(row_value);
                }

                break;
            }

            case CloudQMchineType::FIDELITY:
            {
                std::string qst_fidelity_str = list[0]["qstfidelity"].as_string();
                m_qst_fidelity = stod(qst_fidelity_str);

                break;
            }

            default: QCERR_AND_THROW(run_fail, "quantum machine type error");
            }

            return false;
        }

        case TaskStatus::FAILED:
        {
            QCERR_AND_THROW(run_fail, "Task run failed");
        }

        case TaskStatus::WAITING:
        case TaskStatus::COMPUTING:
        case TaskStatus::QUEUING: m_task_status = status;

            //The next status only appear in real chip backend
        case TaskStatus::SENT_TO_BUILD_SYSTEM:
        case TaskStatus::BUILD_SYSTEM_RUN: return true;
        case TaskStatus::BUILD_SYSTEM_ERROR: QCERR_AND_THROW(run_fail, "build system error");
        case TaskStatus::SEQUENCE_TOO_LONG: QCERR_AND_THROW(run_fail, "exceeding maximum timing sequence");

        default: return true;
        }
    }
    catch (const std::exception&e)
    {
        if (m_is_logged) std::cout << recv_json << std::endl;
        std::string err_info = "task execute failed : " + std::string(e.what());
        QCERR_AND_THROW(run_fail, err_info);
    }

    return false;
}

std::string QCloudMachine::get_result_json(std::string taskid, std::string url, CloudQMchineType type)
{
    rabbit::document doc;
    doc.parse("{}");

    doc.insert("taskId", taskid);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", (size_t)type);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(url, post_json_str);
    return recv_json_str;
}

std::vector<std::map<std::string, double>> QCloudMachine::full_amplitude_measure_batch(std::vector<QProg>& prog_array, int shot, std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::Full_AMPLITUDE));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_MEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("shot", to_string(shot));
    doc.insert("taskName", task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    inquire_batch_result(recv_json_str, m_batch_inquire_url, CloudQMchineType::Full_AMPLITUDE);
    std::vector<std::map<std::string, double>> result;
    for (const auto& val : m_batch_measure_result)
    {
        result.emplace_back(val.second);
    }
    return result;
}

std::vector<std::map<std::string, double>> QCloudMachine::full_amplitude_pmeasure_batch(std::vector<QProg>& prog_array, Qnum qubits, std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::Full_AMPLITUDE));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_PMEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("qubits", to_string_array(qubits));
    doc.insert("taskName", task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    inquire_batch_result(recv_json_str, m_batch_inquire_url, CloudQMchineType::Full_AMPLITUDE);
    std::vector<std::map<std::string, double>> result;
    for (const auto& val : m_batch_measure_result)
    {
        result.emplace_back(val.second);
    }
    return result;
}

std::vector<std::map<std::string, qcomplex_t>> QCloudMachine::partial_amplitude_pmeasure_batch(
    std::vector<QProg>& prog_array,
    std::vector<std::string> amplitude_vec,
    std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::PARTIAL_AMPLITUDE));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_PMEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("Amplitude", to_string_array(amplitude_vec));
    doc.insert("taskName", task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    inquire_batch_result(recv_json_str, m_batch_inquire_url, CloudQMchineType::PARTIAL_AMPLITUDE);

    std::vector<std::map<std::string, qcomplex_t>> result;
    for (auto val : m_batch_pmeasure_result)
    {
        result.emplace_back(val.second);
    }

    return result;
}

std::vector<qcomplex_t> QCloudMachine::single_amplitude_pmeasure_batch(
    std::vector<QProg>& prog_array,
    std::string amplitude,
    std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::SINGLE_AMPLITUDE));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_PMEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("Amplitude", amplitude);
    doc.insert("taskName", task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    inquire_batch_result(recv_json_str, m_batch_inquire_url, CloudQMchineType::SINGLE_AMPLITUDE);
    std::vector<qcomplex_t> result;
    for (auto val : m_batch_single_result)
    {
        result.emplace_back(val.second);
    }

    return result;
}

std::vector<std::map<std::string, double>> QCloudMachine::noise_measure_batch(
    std::vector<QProg>& prog_array,
    int shot,
    std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::NOISE_QMACHINE));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_MEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("shot", to_string(shot));
    doc.insert("taskName", task_name);

    doc.insert("singleGate", m_noise_params.single_gate_param);
    doc.insert("doubleGate", m_noise_params.double_gate_param);

    if ("DECOHERENCE_KRAUS_OPERATOR" == m_noise_params.noise_model)
    {
        doc.insert("singleP2", m_noise_params.single_p2);
        doc.insert("doubleP2", m_noise_params.double_p2);
        doc.insert("singlePgate", m_noise_params.single_pgate);
        doc.insert("doublePgate", m_noise_params.double_pgate);
    }

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    inquire_batch_result(recv_json_str, m_batch_inquire_url, CloudQMchineType::NOISE_QMACHINE);
    std::vector<std::map<std::string, double>> result;
    for (const auto& val : m_batch_measure_result)
    {
        result.emplace_back(val.second);
    }
    return result;
}

std::vector<std::map<std::string, double>> QCloudMachine::real_chip_measure_batch(
    std::vector<QProg>& prog_array,
    int shot,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::REAL_CHIP));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_MEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("shot", to_string(shot));
    doc.insert("taskName", task_name);
    doc.insert("isAmend", !is_amend);
    doc.insert("mappingFlag", !is_mapping);
    doc.insert("circuitOptimization", !is_optimization);
    doc.insert("chipId", (size_t)chip_id);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    inquire_batch_result(recv_json_str, m_batch_inquire_url, CloudQMchineType::REAL_CHIP);
    std::vector<std::map<std::string, double>> result;
    for (const auto& val : m_batch_measure_result)
    {
        result.emplace_back(val.second);
    }
    return result;
}

bool QCloudMachine::parser_submit_json_batch(std::string &recv_json, std::map<size_t, std::string>& taskid_map)
{
    json_string_transfer_encoding(recv_json);
    rabbit::document doc;

    try
    {
        doc.parse(recv_json);

        auto success = doc["success"].as_bool();
        if (!success)
        {
            m_error_info = doc["message"].as_string();
            QCERR_AND_THROW(run_fail, m_error_info);
        }
        else
        {
            for (auto i = 0; i < doc["obj"]["stepTaskResultList"].size(); ++i)
            {
                auto step_id = doc["obj"]["stepTaskResultList"][i]["step"].as_string();
                auto task_id = doc["obj"]["stepTaskResultList"][i]["taskId"].as_string();

                taskid_map.insert(make_pair(stoi(step_id), task_id));
            }

            return true;
        }

    }
    catch (const exception& e)
    {
        if (m_is_logged) std::cout << recv_json << std::endl;

        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::string QCloudMachine::get_result_json_batch(std::map<size_t, std::string> taskid_map, std::string url, CloudQMchineType type)
{
    rabbit::document doc;
    doc.parse("{}");

    std::string string_array;
    for (auto val : taskid_map)
    {
        string_array.append(val.second);
        string_array.append(";");
    }

    doc.insert("taskIds", string_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)type));

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(url, post_json_str);
    return recv_json_str;
}

bool QCloudMachine::parser_result_json_batch(std::string &recv_json, std::map<size_t, std::string>& taskid_map)
{
    json_string_transfer_encoding(recv_json);

    try
    {
        rabbit::document doc;
        doc.parse(recv_json);

        bool success = doc["success"].as_bool();
        if (!success)
        {
            string message = doc["message"].as_string();
            QCERR_AND_THROW(run_fail, message.c_str());
        }

        for (int i = 0; i < doc["obj"].size(); i++)
        {
            auto step = doc["obj"][i]["step"].as_string();
            auto stat = doc["obj"][i]["taskState"].as_string();

            switch ((TaskStatus)stoi(stat))
            {
            case TaskStatus::FINISHED:
            {
                auto result_step = doc["obj"][i]["step"].as_string();
                auto result_string = doc["obj"][i]["taskResult"].as_string();

                rabbit::document result_doc;
                result_doc.parse(result_string);

                if (!result_doc.has("ResultType"))
                {
                    m_measure_result.clear();

                    auto key_string = rabbit_json_extract(result_doc, "key");
                    auto value_string = rabbit_json_extract(result_doc, "value");

                    for (int j = 0; j < result_doc[value_string.c_str()].size(); j++)
                    {
                        auto key = result_doc[key_string.c_str()][j].as_string();
                        auto val = result_doc[value_string.c_str()][j].is_double() ?
                            result_doc[value_string.c_str()][j].as_double() : (double)result_doc[value_string.c_str()][j].as_int();

                        m_measure_result.insert(make_pair(key, val));
                    }

                    m_batch_measure_result[stoi(result_step)] = m_measure_result;
                    break;
                }

                auto result_type = result_doc["ResultType"].as_int();
                switch ((ReasultType)result_type)
                {
                case ReasultType::PROBABILITY_MAP:
                {
                    m_measure_result.clear();

                    auto key_string = rabbit_json_extract(result_doc, "key");
                    auto value_string = rabbit_json_extract(result_doc, "value");

                    for (int j = 0; j < result_doc[value_string.c_str()].size(); j++)
                    {
                        auto key = result_doc[key_string.c_str()][j].as_string();
                        auto val = result_doc[value_string.c_str()][j].is_double() ?
                            result_doc[value_string.c_str()][j].as_double() : (double)result_doc[value_string.c_str()][j].as_int();

                        m_measure_result.insert(make_pair(key, val));
                    }

                    m_batch_measure_result[stoi(result_step)] = m_measure_result;
                    break;
                }

                case ReasultType::SINGLE_PROBABILITY:
                {
                    auto value_string = rabbit_json_extract(result_doc, "value");

                    m_expectation = result_doc[value_string.c_str()].is_double() ?
                        result_doc[value_string.c_str()].as_double() : (double)result_doc[value_string.c_str()].as_int();
                }

                case ReasultType::MULTI_AMPLITUDE:
                {
                    m_pmeasure_result.clear();

                    auto key_string = rabbit_json_extract(result_doc, "key");

                    if (result_doc[key_string.c_str()].is_array())   //partial amplitude
                    {
                        for (int j = 0; j < result_doc[key_string.c_str()].size(); j++)
                        {
                            auto key = result_doc[key_string.c_str()][j].as_string();
                            auto val_real = result_doc["ValueReal"][j].is_double() ?
                                result_doc["ValueReal"][j].as_double() : (double)result_doc["ValueReal"][j].as_int();
                            auto val_imag = result_doc["ValueImag"][j].is_double() ?
                                result_doc["ValueImag"][j].as_double() : (double)result_doc["ValueImag"][j].as_int();

                            m_pmeasure_result.insert(make_pair(key, qcomplex_t(val_real, val_imag)));
                        }

                        m_batch_pmeasure_result[stoi(result_step)] = m_pmeasure_result;
                    }
                    else  //single amplitude
                    {
                        auto key_string = rabbit_json_extract(result_doc, "key");
                        auto key = result_doc[key_string.c_str()].as_string();
                        auto val_real = result_doc["ValueReal"][0].is_double() ?
                            result_doc["ValueReal"][0].as_double() : (double)result_doc["ValueReal"][0].as_int();
                        auto val_imag = result_doc["ValueImag"][0].is_double() ?
                            result_doc["ValueImag"][0].as_double() : (double)result_doc["ValueImag"][0].as_int();

                        m_batch_single_result.insert(make_pair(stoi(result_step), qcomplex_t(val_real, val_imag)));
                    }

                    break;
                }

                case ReasultType::SINGLE_AMPLTUDE:
                {
                    auto val_real = result_doc["ValueReal"].is_double() ?
                        result_doc["ValueReal"].as_double() : (double)result_doc["ValueReal"].as_int();
                    auto val_imag = result_doc["ValueImag"].is_double() ?
                        result_doc["ValueImag"].as_double() : (double)result_doc["ValueImag"].as_int();

                    m_batch_single_result[stoi(result_step)] = (qcomplex_t(val_real, val_imag));
                    break;
                }

                default: QCERR_AND_THROW(run_fail, "quantum machine type error");
                }
            }break;


            case TaskStatus::WAITING:
            case TaskStatus::COMPUTING:
            case TaskStatus::QUEUING:

                //The next status only appear in real chip backend
            case TaskStatus::SENT_TO_BUILD_SYSTEM:
            case TaskStatus::BUILD_SYSTEM_RUN: return true;

            case TaskStatus::BUILD_SYSTEM_ERROR: QCERR_AND_THROW(run_fail, "build system error");
            case TaskStatus::SEQUENCE_TOO_LONG: QCERR_AND_THROW(run_fail, "exceeding maximum timing sequence");
            case TaskStatus::FAILED: QCERR_AND_THROW(run_fail, "task failed");
            default: return true;
            }
        }
    }
    catch (const std::exception& e)
    {
        if (m_is_logged) std::cout << recv_json << std::endl;

        QCERR_AND_THROW(run_fail, e.what());
    }

    return false;
}

void QCloudMachine::inquire_batch_result(std::string recv_json_str, std::string url, CloudQMchineType type)
{
    std::map<size_t, std::string> taskid_map;
    if (parser_submit_json_batch(recv_json_str, taskid_map))
    {
        bool is_retry_again = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            auto result_json = get_result_json_batch(taskid_map, url, type);

            is_retry_again = parser_result_json_batch(result_json, taskid_map);

        } while (is_retry_again);
    }

    return;
}

std::map<size_t, std::string> QCloudMachine::full_amplitude_measure_batch_commit(
    std::vector<QProg>& prog_array,
    int shot,
    TaskStatus& status,
    std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::Full_AMPLITUDE));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_MEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("shot", to_string(shot));
    doc.insert("taskName", task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    try
    {

        std::map<size_t, std::string> taskid_map;
        parser_submit_json_batch(recv_json_str, taskid_map);

        status = TaskStatus::COMPUTING;
        return taskid_map;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return {};
    }
}

std::map<size_t, std::string> QCloudMachine::full_amplitude_pmeasure_batch_commit(
    std::vector<QProg>& prog_array,
    Qnum qubit_vec,
    TaskStatus& status,
    std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::Full_AMPLITUDE));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_MEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("qubits", to_string_array(qubit_vec));
    doc.insert("taskName", task_name);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    try
    {
        std::map<size_t, std::string> taskid_map;
        parser_submit_json_batch(recv_json_str, taskid_map);

        status = TaskStatus::COMPUTING;
        return taskid_map;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return {};
    }
}
std::map<size_t, std::string> QCloudMachine::real_chip_measure_batch_commit(std::vector<QProg>& prog_array,
    int shot,
    TaskStatus& status,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    //construct json
    rabbit::document doc;
    doc.parse("{}");

    size_t code_len;
    rabbit::array code_array;
    construct_multi_prog_json(this, code_array, code_len, prog_array);

    doc.insert("codeArr", code_array);
    doc.insert("apiKey", m_token);
    doc.insert("QMachineType", to_string((size_t)CloudQMchineType::REAL_CHIP));
    doc.insert("codeLen", to_string(code_len));
    doc.insert("qubitNum", to_string(getAllocateQubit()));
    doc.insert("measureType", to_string(ClusterTaskType::CLUSTER_MEASURE));
    doc.insert("classicalbitNum", to_string(getAllocateCMem()));
    doc.insert("shot", to_string(shot));
    doc.insert("taskName", task_name);
    doc.insert("isAmend", !is_amend);
    doc.insert("mappingFlag", !is_mapping);
    doc.insert("circuitOptimization", !is_optimization);
    doc.insert("chipId", (size_t)chip_id);

    std::string post_json_str = doc.str();
    std::string recv_json_str = post_json(m_batch_compute_url, post_json_str);

    try
    {

        std::map<size_t, std::string> taskid_map;
        parser_submit_json_batch(recv_json_str, taskid_map);

        status = TaskStatus::COMPUTING;
        return taskid_map;
    }
    catch (...)
    {
        status = TaskStatus::FAILED;
        return {};
    }
}

std::map<size_t, std::map<std::string, double>> QCloudMachine::full_amplitude_measure_batch_query(std::map<size_t, std::string> taskid_map)
{
    try
    {
        auto result_json = get_result_json_batch(taskid_map, m_batch_inquire_url, CloudQMchineType::Full_AMPLITUDE);

        bool is_retry_again = parser_result_json_batch(result_json, taskid_map);

        return m_batch_measure_result;
    }
    catch (...)
    {
        return {};
    }
}
std::map<size_t, std::map<std::string, double>> QCloudMachine::full_amplitude_pmeasure_batch_query(std::map<size_t, std::string> taskid_map)
{
    try
    {
        auto result_json = get_result_json_batch(taskid_map, m_batch_inquire_url, CloudQMchineType::Full_AMPLITUDE);

        bool is_retry_again = parser_result_json_batch(result_json, taskid_map);

        return m_batch_measure_result;
    }
    catch (...)
    {
        return {};
    }
}
std::map<size_t, std::map<std::string, double>> QCloudMachine::real_chip_measure_batch_query(std::map<size_t, std::string> taskid_map)
{
    try
    {
        auto result_json = get_result_json_batch(taskid_map, m_batch_inquire_url, CloudQMchineType::REAL_CHIP);

        bool is_retry_again = parser_result_json_batch(result_json, taskid_map);

        return m_batch_measure_result;
    }
    catch (...)
    {
        return {};
    }
}


REGISTER_QUANTUM_MACHINE(QCloudMachine);
