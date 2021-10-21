#include <chrono>
#include <thread>
#include <math.h>
#include <fstream>
#include <algorithm>
#include "Core/Utilities/Tools/base64.hpp"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"

#ifdef USE_CURL
#include <curl/curl.h>
#endif

#define MAX_FULL_AMPLITUDE_QUBIT_NUM 35
#define MAX_PARTIAL_AMPLITUDE_QUBIT_NUM 64

#define  QCLOUD_COMPUTE_API_POSTFIX  "/api/taskApi/submitTask.json"
#define  QCLOUD_INQUIRE_API_POSTFIX  "/api/taskApi/getTaskDetail.json"

#define  REAL_CHIP_COMPUTE_API_POSTFIX "/api/taskApi/submitTask.json"
#define  REAL_CHIP_INQUIRE_API_POSTFIX "/api/taskApi/getTaskDetail.json"

#define DEFAULT_CLUSTER_COMPUTEAPI    "https://qcloud.originqc.com.cn/api/taskApi/submitTask.json"
#define DEFAULT_CLUSTER_INQUIREAPI     "https://qcloud.originqc.com.cn/api/taskApi/getTaskDetail.json"

#define DEFAULT_REAL_CHIP_TASK_COMPUTEAPI     "https://qcloud.originqc.com.cn/api/taskApi/submitTask.json"
#define DEFAULT_REAL_CHIP_TASK_INQUIREAPI      "https://qcloud.originqc.com.cn/api/taskApi/getTaskDetail.json"

USING_QPANDA
using namespace std;
using namespace Base64;
using namespace rapidjson;

static std::map<NOISE_MODEL, string> noise_model_mapping = 
{ {NOISE_MODEL::BITFLIP_KRAUS_OPERATOR,"BITFLIP_KRAUS_OPERATOR"},
  {NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR,"BIT_PHASE_FLIP_OPRATOR"},
  {NOISE_MODEL::DAMPING_KRAUS_OPERATOR,"DAMPING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR,"DECOHERENCE_KRAUS_OPERATOR"},
  {NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR_P1_P2,"DECOHERENCE_KRAUS_OPERATOR_P1_P2"},
  {NOISE_MODEL::DEPHASING_KRAUS_OPERATOR,"DEPHASING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR,"DEPOLARIZING_KRAUS_OPERATOR"},
  {NOISE_MODEL::KRAUS_MATRIX_OPRATOR,"KRAUS_MATRIX_OPRATOR"},
  {NOISE_MODEL::MIXED_UNITARY_OPRATOR,"MIXED_UNITARY_OPRATOR"},
  {NOISE_MODEL::PAULI_KRAUS_MAP,"PAULI_KRAUS_MAP"},
  {NOISE_MODEL::PHASE_DAMPING_OPRATOR,"PHASE_DAMPING_OPRATOR"}
};

static string to_string_array(const Qnum values)
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

static string to_string_array(const std::vector<string> values)
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

static string hamiltonian_to_json(const QHamiltonian& hamiltonian)
{
    //construct json
    Document doc;
    doc.SetObject();
    Document::AllocatorType &alloc = doc.GetAllocator();

    Value hamiltonian_value_array(rapidjson::kObjectType);
    Value hamiltonian_param_array(rapidjson::kArrayType);

    Value pauli_parm_array(rapidjson::kArrayType);
    Value pauli_type_array(rapidjson::kArrayType);

    for (auto i = 0; i < hamiltonian.size(); ++i)
    {
        const auto& item = hamiltonian[i];

        Value temp_pauli_parm_array(rapidjson::kArrayType);
        Value temp_pauli_type_array(rapidjson::kArrayType);

        for (auto val : item.first)
        {
            temp_pauli_parm_array.PushBack((SizeType)val.first, alloc);

            rapidjson::Value string_key(kStringType);
            string_key.SetString(std::string(1, val.second).c_str(), 1, alloc);

            temp_pauli_type_array.PushBack(string_key, alloc);
        }

        pauli_parm_array.PushBack(temp_pauli_parm_array, alloc);
        pauli_type_array.PushBack(temp_pauli_type_array, alloc);

        hamiltonian_param_array.PushBack(item.second, alloc);
    }

    hamiltonian_value_array.AddMember("pauli_type", pauli_type_array, alloc);
    hamiltonian_value_array.AddMember("pauli_parm", pauli_parm_array, alloc);

    doc.AddMember("hamiltonian_value", hamiltonian_value_array, alloc);
    doc.AddMember("hamiltonian_param", hamiltonian_param_array, alloc);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string hamiltonian_str = buffer.GetString();
    return hamiltonian_str;
}

static QHamiltonian json_to_hamiltonian(const std::string& hamiltonian_json)
{
    Document doc;
    if (doc.Parse(hamiltonian_json.c_str()).HasParseError())
    {
        QCERR(hamiltonian_json.c_str());
        throw run_fail("result json parse error");
    }

    try
    {
        QHamiltonian result;

        const rapidjson::Value& hamiltonian_value_array = doc["hamiltonian_value"];
        const rapidjson::Value& hamiltonian_param_array = doc["hamiltonian_param"];

        const rapidjson::Value& pauli_type_array = hamiltonian_value_array["pauli_type"];
        const rapidjson::Value& pauli_parm_array = hamiltonian_value_array["pauli_parm"];

        QPANDA_ASSERT(pauli_type_array.Size() != pauli_parm_array.Size(), "hamiltonian json error");

        for (SizeType i = 0; i < pauli_type_array.Size(); ++i)
        {
            QTerm qterm;

            const rapidjson::Value &pauli_type_value = pauli_type_array[i];
            const rapidjson::Value &pauli_parm_value = pauli_parm_array[i];

            const rapidjson::Value &hamiltonian_parm = hamiltonian_param_array[i];

            QPANDA_ASSERT(pauli_type_value.Size() != pauli_parm_value.Size(), "hamiltonian json error");

            for (SizeType j = 0; j < pauli_type_value.Size(); ++j)
            {
                size_t pauli_parm = pauli_parm_value[j].GetInt();
                string pauli_type = pauli_type_value[j].GetString();
                qterm.insert(std::make_pair(pauli_parm, pauli_type[0]));
            }

            result.emplace_back(std::make_pair(qterm, hamiltonian_parm.GetDouble()));
        }

        return result;
    }
    catch (std::exception& e)
    {
        QCERR(e.what());
        throw run_fail("hamiltonian json error");
    }
}

QCloudMachine::QCloudMachine()
{
#ifdef USE_CURL
    curl_global_init(CURL_GLOBAL_ALL);
#else
    QCERR_AND_THROW(run_fail, "Need support the curl libray");
#endif
}

QCloudMachine::~QCloudMachine()        
{
#ifdef USE_CURL
    curl_global_cleanup();
#else
	QCERR_AND_THROW(run_fail, "Need support the curl libray");
#endif
}

void QCloudMachine::set_qcloud_api(std::string url) 
{ 
    m_compute_url = url + QCLOUD_COMPUTE_API_POSTFIX;
    m_inquire_url = url + QCLOUD_INQUIRE_API_POSTFIX;
}

void QCloudMachine::set_real_chip_api(std::string url)
{
    m_real_chip_task_compute_url = url + REAL_CHIP_COMPUTE_API_POSTFIX;
    m_real_chip_task_inquire_url = url + REAL_CHIP_INQUIRE_API_POSTFIX;
}

void QCloudMachine::init(string token, bool is_logged)
{
    JsonConfigParam config;
    try
    {
        if (!config.load_config(CONFIG_PATH))
        {
            if (m_is_logged) std::cout << "config warning: can not find config file, use default config" << endl;

            m_compute_url = DEFAULT_CLUSTER_COMPUTEAPI;
            m_inquire_url = DEFAULT_CLUSTER_INQUIREAPI;

            m_real_chip_task_compute_url = DEFAULT_REAL_CHIP_TASK_COMPUTEAPI;
            m_real_chip_task_inquire_url = DEFAULT_REAL_CHIP_TASK_INQUIREAPI;
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

                m_real_chip_task_compute_url = DEFAULT_REAL_CHIP_TASK_COMPUTEAPI;
                m_real_chip_task_inquire_url = DEFAULT_REAL_CHIP_TASK_INQUIREAPI;
            }
            else
            {
                set_qcloud_api(QCloudConfig["QCloudAPI"]);
                set_real_chip_api(QCloudConfig["RealChipAPI"]);
            }
        }
    }
    catch (std::exception &e)
    {
        if (m_is_logged) std::cout << "config warning: load config file catch exception, use default config" << endl;

        m_compute_url = DEFAULT_CLUSTER_COMPUTEAPI;
        m_inquire_url = DEFAULT_CLUSTER_INQUIREAPI;

        m_real_chip_task_compute_url = DEFAULT_REAL_CHIP_TASK_COMPUTEAPI;
        m_real_chip_task_inquire_url = DEFAULT_REAL_CHIP_TASK_INQUIREAPI;
    }

    try
    {
        m_token = token;
        m_is_logged = is_logged;
        _start();
    }
    catch (std::exception &e)
    {
        finalize();
        QCERR(e.what());
        throw init_fail(e.what());
    }
}

size_t recvJsonData(void *ptr, size_t size, size_t nmemb, void *stream)
{
    std::string data((const char*)ptr, 0, (size_t)(size * nmemb));
    *((std::stringstream*)stream) << data << std::endl;
    return size * nmemb;
}

void QCloudMachine::set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params)
{
    auto iter = noise_model_mapping.find(model);
    if (noise_model_mapping.end() == iter || single_params.empty() || double_params.empty())
    {
        QCERR("NOISE MODEL ERROR");
        throw run_fail("NOISE MODEL ERROR");
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
        QCERR("DECOHERENCE_KRAUS_OPERATOR ERROR");
        throw run_fail("DECOHERENCE_KRAUS_OPERATOR ERROR");
    }

    return;
}

std::string QCloudMachine::post_json(const std::string &sUrl, std::string & sJson)
{
#ifdef USE_CURL
    std::stringstream out;

    auto pCurl = curl_easy_init();

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers,"Content-Type: application/json;charset=UTF-8");
    headers = curl_slist_append(headers, "Connection: keep-alive");
    headers = curl_slist_append(headers, "Server: nginx/1.16.1");
    headers = curl_slist_append(headers,"Transfer-Encoding: chunked"); 
    curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(pCurl, CURLOPT_TIMEOUT, 10);

    curl_easy_setopt(pCurl, CURLOPT_CONNECTTIMEOUT, 0);

    curl_easy_setopt(pCurl, CURLOPT_URL, sUrl.c_str());

    curl_easy_setopt(pCurl, CURLOPT_HEADER, true);

    curl_easy_setopt(pCurl, CURLOPT_POST, true);
     
    curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYHOST, false);

    curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYPEER, false);

    curl_easy_setopt(pCurl, CURLOPT_READFUNCTION, NULL);

    curl_easy_setopt(pCurl, CURLOPT_NOSIGNAL, 1);

    curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, sJson.c_str());

    curl_easy_setopt(pCurl, CURLOPT_POSTFIELDSIZE, sJson.size());

    curl_easy_setopt(pCurl, CURLOPT_WRITEFUNCTION, recvJsonData);

    //curl_easy_setopt(pCurl, CURLOPT_VERBOSE, 1);

    curl_easy_setopt(pCurl, CURLOPT_WRITEDATA, &out);

    auto res = curl_easy_perform(pCurl);
    if (CURLE_OK != res)
    {
        QCERR(curl_easy_strerror(res));
        throw run_fail("QCloudMachine::post_json");
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(pCurl);

    try
    {
        std::string result =  out.str().substr(out.str().find("{"));;
        return result;
    }
    catch (...)
    {
        if (m_is_logged) std::cout << out.str() << endl;

        QCERR("post json failed");
        throw run_fail("QCloudMachine::post_json");
    }
#else
    QCERR_AND_THROW(run_fail, "need support the curl libray");
#endif
}

void QCloudMachine::inquire_result(std::string recv_json_str, std::string url, CLOUD_QMACHINE_TYPE type)
{
    std::string taskid;
    if (parser_submit_json(recv_json_str, taskid))
    {
        bool retry_inquire = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            auto result_json = get_result_json(taskid, url, type);

            retry_inquire = parser_result_json(result_json, taskid);

        } while (retry_inquire);
    }

    return;
}

double QCloudMachine::get_state_fidelity(
    QProg &prog,
    int shots,
    REAL_CHIP_TYPE chipid,
    bool mapping_flag,
    bool circuit_optimization,
    std::string task_name)
{
    auto qubit_num = getAllocateQubit();
    auto cbit_num = getAllocateCMem();

    QPANDA_ASSERT(qubit_num > 6 || cbit_num > 6, "real chip qubit num or cbit num are not less or equal to 6");
    QPANDA_ASSERT(shots > 10000 || shots < 1000, "real chip shots must be in range [1000,10000]");

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    if (!traver_param.m_can_optimize_measure)
    {
        QCERR("measure must be last");
        throw run_fail("measure must be last");
    }

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "mappingFlag", (size_t)!mapping_flag);
    add_string_value(doc, "circuitOptimization", (size_t)!circuit_optimization);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::FIDELITY);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", (size_t)shots);
    add_string_value(doc, "chipId", (size_t)chipid);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_real_chip_task_compute_url, post_json_str);

    inquire_result(recv_json_str, m_real_chip_task_inquire_url, CLOUD_QMACHINE_TYPE::FIDELITY);

    return m_qst_fidelity;
}


std::vector<QStat> QCloudMachine::get_state_tomography_density(
    QProg &prog,
    int shots,
    REAL_CHIP_TYPE chipid,
    bool mapping_flag,
    bool circuit_optimization,
    std::string task_name)
{
    auto qubit_num = getAllocateQubit();
    auto cbit_num = getAllocateCMem();

    QPANDA_ASSERT(qubit_num > 6 || cbit_num > 6, "real chip qubit num or cbit num are not less or equal to 6");
    QPANDA_ASSERT(shots > 10000 || shots < 1000, "real chip shots must be in range [1000,10000]");

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    if (!traver_param.m_can_optimize_measure)
    {
        QCERR("measure must be last");
        throw run_fail("measure must be last");
    }

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "mappingFlag", (size_t)!mapping_flag);
    add_string_value(doc, "circuitOptimization", (size_t)!circuit_optimization);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::QST);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", (size_t)shots);
    add_string_value(doc, "chipId", (size_t)chipid);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_real_chip_task_compute_url, post_json_str);
    
    inquire_result(recv_json_str, m_real_chip_task_inquire_url, CLOUD_QMACHINE_TYPE::QST);

    return m_qst_result;
}

std::map<std::string, double> QCloudMachine::real_chip_measure(
    QProg &prog,
    int shots,
    REAL_CHIP_TYPE chipid,
    bool mapping_flag,
    bool circuit_optimization,
    std::string task_name)
{
    auto qubit_num = getAllocateQubit();
    auto cbit_num = getAllocateCMem();

    QPANDA_ASSERT(qubit_num > 6 || cbit_num > 6, "real chip qubit num or cbit num are not less or equal to 6");
    QPANDA_ASSERT(shots > 10000 || shots < 1000, "real chip shots must be in range [1000,10000]");

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    QPANDA_ASSERT(!traver_param.m_can_optimize_measure, "measure must be last");

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    //construct configuration json: circuitOptimization, mappingFlag, shot
    Document configuration_doc;
    configuration_doc.SetObject();

    rapidjson::Document::AllocatorType &configuration_alloc = configuration_doc.GetAllocator();

    configuration_doc.AddMember("mappingFlag", (SizeType)!true, configuration_alloc);
    configuration_doc.AddMember("circuitOptimization", (SizeType)!true, configuration_alloc);

    rapidjson::StringBuffer configuration_buffer;
    rapidjson::Writer<rapidjson::StringBuffer> configuration_writer(configuration_buffer);
    configuration_doc.Accept(configuration_writer);

    auto configuration_str = configuration_buffer.GetString();

    //others
    //add_string_value(doc, "apiKey", m_token);
    //add_string_value(doc, "Configuration", configuration_str);

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "mappingFlag", (size_t)!mapping_flag);
    add_string_value(doc, "circuitOptimization", (size_t)!circuit_optimization);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::REAL_CHIP);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", (size_t)shots);
    add_string_value(doc, "taskName", task_name);
    add_string_value(doc, "chipId", (size_t)chipid);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_real_chip_task_compute_url, post_json_str);

    inquire_result(recv_json_str, m_real_chip_task_inquire_url, CLOUD_QMACHINE_TYPE::REAL_CHIP);
    return m_measure_result;
}

std::map<std::string, double> QCloudMachine::noise_measure(QProg &prog, int shot, string task_name)
{
    //convert prog to originir
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::NOISE_QMACHINE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubit());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", (size_t)shot);
    add_string_value(doc, "noisemodel", m_noise_params.noise_model);
    add_string_value(doc, "singleGate", m_noise_params.single_gate_param);
    add_string_value(doc, "doubleGate", m_noise_params.double_gate_param);
    add_string_value(doc, "taskName", task_name);

    if ("DECOHERENCE_KRAUS_OPERATOR" == m_noise_params.noise_model)
    {
        add_string_value(doc, "singleP2", m_noise_params.single_p2);
        add_string_value(doc, "doubleP2", m_noise_params.double_p2);
        add_string_value(doc, "singlePgate", m_noise_params.single_pgate);
        add_string_value(doc, "doublePgate", m_noise_params.double_pgate);
    }

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CLOUD_QMACHINE_TYPE::NOISE_QMACHINE);
    return m_measure_result;
}

std::map<std::string, double> QCloudMachine::full_amplitude_measure(QProg &prog, int shot, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    QPANDA_ASSERT(getAllocateQubit() > MAX_FULL_AMPLITUDE_QUBIT_NUM, "unsupported qubit num, max is 35");

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubit());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", (size_t)shot);
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    return m_measure_result;
}

std::string QCloudMachine::full_amplitude_measure_commit(QProg &prog, int shot, TASK_STATUS& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    QPANDA_ASSERT(getAllocateQubit() > MAX_FULL_AMPLITUDE_QUBIT_NUM, "unsupported qubit num, max is 35");

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubit());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", (size_t)shot);
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    try
    {
        std::string taskid;
        parser_submit_json(recv_json_str, taskid);
        status = TASK_STATUS::COMPUTING;
        return taskid;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return "";
    }
}

std::string QCloudMachine::full_amplitude_pmeasure_commit(QProg &prog, Qnum qubit_vec, TASK_STATUS& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    QPANDA_ASSERT(getAllocateQubit() > MAX_FULL_AMPLITUDE_QUBIT_NUM, "unsupported qubit num, max is 35");

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "qubits", to_string_array(qubit_vec));
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    try
    {
        std::string taskid;
        parser_submit_json(recv_json_str, taskid);
        status = TASK_STATUS::COMPUTING;
        return taskid;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return "";
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_measure_query(std::string taskid, TASK_STATUS& status)
{
    try
    {
        auto result_json = get_result_json(taskid, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

        bool retry_inquire = parser_result_json(result_json, taskid);

        status = m_task_status;
        return retry_inquire ? std::map<std::string, double>() : m_measure_result;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return {};
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_measure_exec(std::string taskid, TASK_STATUS& status)
{
    try
    {
        bool retry_inquire = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            auto result_json = get_result_json(taskid, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

            retry_inquire = parser_result_json(result_json, taskid);

        } while (retry_inquire);

        status = TASK_STATUS::FINISHED;
        return m_measure_result;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return {};
    }
}

std::map<std::string, qcomplex_t> QCloudMachine::full_amplitude_pmeasure_query(std::string taskid, TASK_STATUS& status)
{
    try
    {
        auto result_json = get_result_json(taskid, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

        bool retry_inquire = parser_result_json(result_json, taskid);

        status = m_task_status;
        return retry_inquire ? std::map<std::string, qcomplex_t>() : m_pmeasure_result;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return {};
    }
}

std::map<std::string, qcomplex_t> QCloudMachine::full_amplitude_pmeasure_exec(std::string taskid, TASK_STATUS& status)
{
    try
    {
        bool retry_inquire = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            auto result_json = get_result_json(taskid, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

            retry_inquire = parser_result_json(result_json, taskid);

        } while (retry_inquire);

        status = TASK_STATUS::FINISHED;
        return m_pmeasure_result;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return {};
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_pmeasure(QProg &prog, Qnum qubit_vec, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    QPANDA_ASSERT(getAllocateQubit() > MAX_FULL_AMPLITUDE_QUBIT_NUM, "unsupported qubit num, max is 35");

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "qubits", to_string_array(qubit_vec));
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    return m_measure_result;
}

std::map<std::string, qcomplex_t> QCloudMachine::partial_amplitude_pmeasure(QProg &prog, std::vector<std::string> amplitude_vec, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    QPANDA_ASSERT(getAllocateQubit() > MAX_PARTIAL_AMPLITUDE_QUBIT_NUM, "unsupported qubit num, max is 64");

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::PARTIAL_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "Amplitude", to_string_array(amplitude_vec));
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);
	
    inquire_result(recv_json_str, m_inquire_url, CLOUD_QMACHINE_TYPE::PARTIAL_AMPLITUDE);
    return m_pmeasure_result;
}

qcomplex_t QCloudMachine::single_amplitude_pmeasure(QProg &prog, std::string amplitude, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::SINGLE_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "Amplitude", amplitude);
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    inquire_result(recv_json_str, m_inquire_url, CLOUD_QMACHINE_TYPE::SINGLE_AMPLITUDE);
    return m_single_result;
}

double QCloudMachine::get_expectation(QProg prog, const QHamiltonian& hamiltonian, const QVec& qvec, TASK_STATUS& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    Qnum qubits;
    for (auto qubit : qvec)
    {
        qubits.emplace_back(qubit->get_phy_addr());
    }

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    QPANDA_ASSERT(getAllocateQubit() > MAX_FULL_AMPLITUDE_QUBIT_NUM, "unsupported qubit num, max is 35");

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_EXPECTATION);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "qubits", to_string_array(qubits));
    add_string_value(doc, "hamiltonian", hamiltonian_to_json(hamiltonian));
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();

    try
    {
        std::string recv_json_str = post_json(m_compute_url, post_json_str);

        inquire_result(recv_json_str, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

        status = TASK_STATUS::FINISHED;
        return m_expectation;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return 0.;
    }
}

std::string QCloudMachine::get_expectation_commit(QProg prog, const QHamiltonian& hamiltonian, const QVec& qvec, TASK_STATUS& status, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    Qnum qubits;
    for (auto qubit : qvec)
    {
        qubits.emplace_back(qubit->get_phy_addr());
    }

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    QPANDA_ASSERT(getAllocateQubit() > MAX_FULL_AMPLITUDE_QUBIT_NUM, "unsupported qubit num, max is 35");

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (size_t)CLUSTER_TASK_TYPE::CLUSTER_EXPECTATION);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "qubits", to_string_array(qubits));
    add_string_value(doc, "hamiltonian", hamiltonian_to_json(hamiltonian));
    add_string_value(doc, "taskName", task_name);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(m_compute_url, post_json_str);

    try
    {
        std::string taskid;
        parser_submit_json(recv_json_str, taskid);
        status = TASK_STATUS::COMPUTING;
        return taskid;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return "";
    }
}

double QCloudMachine::get_expectation_query(std::string taskid, TASK_STATUS& status)
{
    try
    {
        auto result_json = get_result_json(taskid, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

        bool retry_inquire = parser_result_json(result_json, taskid);

        status = m_task_status;
        return retry_inquire ? -1 : m_expectation;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return 0.;
    }
}

double QCloudMachine::get_expectation_exec(std::string taskid, TASK_STATUS& status)
{
    try
    {
        bool retry_inquire = false;

        do
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            auto result_json = get_result_json(taskid, m_inquire_url, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

            retry_inquire = parser_result_json(result_json, taskid);

        } while (retry_inquire);

        status = TASK_STATUS::FINISHED;
        return m_expectation;
    }
    catch (...)
    {
        status = TASK_STATUS::FAILED;
        return 0.;
    }
}

bool QCloudMachine::parser_submit_json(std::string &recv_json, std::string& taskid)
{
    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError() || !recv_doc.HasMember("success"))
    {
        if(m_is_logged) std::cout << recv_json << std::endl;

        m_error_info = "server connection failed";
        QCERR("server connection failed");
        throw run_fail("server connection failed");
    }
    else
    {
        const rapidjson::Value &success = recv_doc["success"];
        if (!success.GetBool())
        {
            if (m_is_logged) std::cout << recv_json << std::endl;

            const rapidjson::Value &message = recv_doc["enMessage"];
            std::string error_msg = message.GetString();

            m_error_info = error_msg;
            QCERR(error_msg.c_str());
            throw run_fail(error_msg);
        }
        else
        {
            const rapidjson::Value &Obj = recv_doc["obj"];
            if (!Obj.IsObject() ||
                !Obj.HasMember("taskId") ||
                !Obj.HasMember("id"))
            {
                if (m_is_logged) std::cout << recv_json << std::endl;

                QCERR("json object error");
                throw run_fail("json object error");
            }
            else
            {
                const rapidjson::Value &task_value = Obj["taskId"];
				taskid = task_value.GetString();
				return true;
            }
        }
    }
}

bool QCloudMachine::parser_result_json(std::string &recv_json, std::string& taskid)
{
    //delete "\r\n" from recv json, Transfer-Encoding: chunked
    int pos = 0;
    while ((pos = recv_json.find("\n")) != -1)
    {
        recv_json.erase(pos, 1);
    }

    if (recv_json.empty())
    {
        QCERR(recv_json.c_str());
        throw run_fail("result json is empty");
    }

    Document recv_doc;
	if (recv_doc.Parse(recv_json.c_str()).HasParseError() || !recv_doc.HasMember("success"))
	{
        QCERR(recv_json.c_str());
        throw run_fail("result json parse error or has no member 'success' ");
	}

    const rapidjson::Value &success = recv_doc["success"];
    if (!success.GetBool())
    {
        const rapidjson::Value &message = recv_doc["enMessage"];
        std::string error_msg = message.GetString();

        m_error_info = error_msg;
        QCERR(error_msg.c_str());
        throw run_fail(error_msg);
    }

	try
	{
		const rapidjson::Value &Obj = recv_doc["obj"];
		const rapidjson::Value &Val = Obj["qcodeTaskNewVo"];
		const rapidjson::Value &List = Val["taskResultList"];
		const rapidjson::Value &result = List[0]["taskResult"];

		std::string state = List[0]["taskState"].GetString();
		std::string qtype = List[0]["rQMachineType"].GetString();

        auto status = static_cast<TASK_STATUS>(atoi(state.c_str()));
        auto backend_type = static_cast<CLOUD_QMACHINE_TYPE>(atoi(qtype.c_str()));
		switch (status)
		{
			case TASK_STATUS::FINISHED:
			{
				Document result_doc;
				result_doc.Parse(result.GetString());

				switch (backend_type)
				{
                    case CLOUD_QMACHINE_TYPE::REAL_CHIP:
                    {
                        Value &key = result_doc["key"];
                        Value &value = result_doc["value"];

                        m_measure_result.clear();
                        for (SizeType i = 0; i < key.Size(); ++i)
                        {
                            std::string bin_amplitude = key[i].GetString();
                            m_measure_result.insert(make_pair(bin_amplitude, value[i].GetDouble()));
                        }

                        break;
                    }

                    case CLOUD_QMACHINE_TYPE::NOISE_QMACHINE:
                    case CLOUD_QMACHINE_TYPE::Full_AMPLITUDE:
					{

                        Value &result_type = result_doc["ResultType"];

                        if (result_type.GetInt() == (int)CLUSTER_RESULT_TYPE::EXPECTATION)
                        {
                            Value &value = result_doc["Value"];
                            m_expectation = value.GetDouble();
                        }
                        else
                        {
                            Value &key = result_doc["Key"];
                            Value &value = result_doc["Value"];

                            m_measure_result.clear();
                            for (SizeType i = 0; i < key.Size(); ++i)
                            {
                                std::string bin_amplitude = key[i].GetString();
                                m_measure_result.insert(make_pair(bin_amplitude, value[i].GetDouble()));
                            }
                        }

                        break;
					}

					case CLOUD_QMACHINE_TYPE::PARTIAL_AMPLITUDE:
					{
                        Value &key = result_doc["Key"];
						Value &value_real = result_doc["ValueReal"];
						Value &value_imag = result_doc["ValueImag"];

                        m_pmeasure_result.clear();
						for (SizeType i = 0; i < key.Size(); ++i)
						{
                            std::string bin_amplitude = key[i].GetString();
                            auto amplitude = qcomplex_t(value_real[i].GetDouble(), value_imag[i].GetDouble());
                            m_pmeasure_result.insert(make_pair(bin_amplitude, amplitude));
						}

                        break;
					}

                    case CLOUD_QMACHINE_TYPE::SINGLE_AMPLITUDE:
                    {
                        Value &value_real = result_doc["ValueReal"];
                        Value &value_imag = result_doc["ValueImag"];

                        m_single_result = qcomplex_t(value_real[0].GetDouble(), value_imag[0].GetDouble());
                        break;
                    }

                    case CLOUD_QMACHINE_TYPE::QST:
                    {
                        const rapidjson::Value &qst_result = List[0]["qstresult"];

                        Document qst_result_doc;
                        qst_result_doc.Parse(qst_result.GetString());

                        m_qst_result.clear();
                        int rank = (int)std::sqrt(qst_result_doc.Size());

                        for (auto i = 0; i < rank; ++i)
                        {
                            QStat row_value;
                            for (auto j = 0; j < rank; ++j)
                            {
                                auto real_val = qst_result_doc[i*rank + j]["r"].GetDouble();
                                auto imag_val = qst_result_doc[i*rank + j]["i"].GetDouble();

                                row_value.emplace_back(qcomplex_t(real_val, imag_val));
                            }

                            m_qst_result.emplace_back(row_value);
                        }

                        break;
                    }

                    case CLOUD_QMACHINE_TYPE::FIDELITY:
                    {
                        const rapidjson::Value &qst_fidelity = List[0]["qstfidelity"];

                        std::string qst_fidelity_str = qst_fidelity.GetString();

                        m_qst_fidelity = stod(qst_fidelity_str);

                        break;
                    }

                    default: QCERR("quantum machine type error"); throw run_fail("quantum machine type error"); break;
				}

                return false;
            }

			case TASK_STATUS::FAILED:
            {
                if (CLOUD_QMACHINE_TYPE::REAL_CHIP == backend_type)
                {
                    Document result_doc;
                    result_doc.Parse(result.GetString());

                    Value &value = result_doc["Value"];

                    QCERR(value.GetString());
                    throw run_fail(value.GetString());
                }
                else
                {
                    QCERR("Task status failed");
                    throw run_fail("Task status failed");
                }
            }

			case TASK_STATUS::WAITING:
            case TASK_STATUS::COMPUTING:
            case TASK_STATUS::QUEUING:
                m_task_status = status;

            //The next status only appear in real chip backend
            case TASK_STATUS::SENT_TO_BUILD_SYSTEM:
            case TASK_STATUS::BUILD_SYSTEM_RUN: return true;

            case TASK_STATUS::BUILD_SYSTEM_ERROR:
                QCERR("build system error");
                throw run_fail("build system error");

            case TASK_STATUS::SEQUENCE_TOO_LONG:
                QCERR("exceeding maximum timing sequence");
                throw run_fail("exceeding maximum timing sequence");

            default: return true;
		}
	}
	catch (const std::exception&e)
	{
        if (m_is_logged) std::cout << recv_json << std::endl;
        QCERR("parse result exception error");
        throw run_fail("parse result exception error");
	}

    return false;
}

std::string QCloudMachine::get_result_json(std::string taskid, std::string url, CLOUD_QMACHINE_TYPE type)
{
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "taskId",   taskid);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (size_t)type);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(url, post_json_str);
	return recv_json_str;
}

void QCloudMachine::add_string_value(rapidjson::Document &doc,const string &key, const std::string &value)
{
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    rapidjson::Value string_key(kStringType);
    string_key.SetString(key.c_str(), (rapidjson::SizeType)key.size(), allocator);

    rapidjson::Value string_value(kStringType);
    string_value.SetString(value.c_str(), (rapidjson::SizeType)value.size(), allocator);

    doc.AddMember(string_key, string_value, allocator);
}

void QCloudMachine::add_string_value(rapidjson::Document &doc, const string &key, const size_t int_value)
{
    std::string value = to_string(int_value);
    add_string_value(doc, key, value);
}

void QCloudMachine::add_string_value(rapidjson::Document &doc, const string &key, const double double_value)
{
    std::string value = to_string(double_value);
    add_string_value(doc, key, value);
}

REGISTER_QUANTUM_MACHINE(QCloudMachine);
