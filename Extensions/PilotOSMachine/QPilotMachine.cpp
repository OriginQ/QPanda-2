#include "QPandaConfig.h"

#include "QPilotMachine.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include "ThirdParty/rabbit/rabbit.hpp"
#include "Def.h"
#include "JsonParser.h"
#include "JsonBuilder.h"

#include <csignal>
#include <time.h>
#include <chrono>
#include <string>
#include <thread>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <stdexcept>

#include "ELog.h"
#include <atomic>
#include <regex>
#include "bz2/bzlib.h"
#include "TCPClient.h"
#ifdef USE_CURL
#include <curl/curl.h>
#endif
using namespace rapidjson;
using namespace JsonMsg;
using namespace QPanda;
using namespace PilotQVM;
using namespace std;

#define  CLUSTER_COMPUTE_URL   "/task/run"
#define  REAL_CHIP_COMPUTE_URL "/task/realQuantum/run"
#define  TASK_QUERY_URL  "/task/realQuantum/query"
#define  COMPILE_PROG_QUERY_URL  "/management/query/taskinfo"
#define  LOGIN_PILOT  "/management/login"
#define  NOISE_LEARNING "/task/realQuantum/noise_learning"
#define  NOISE_QCIR "/task/realQuantum/em_compute"
#define  LOGIN_PILOT_API  "/management/pilotosmachinelogin"

#define MAX_POST_SIZE  1048576 /* 1M:1024 * 1024 */

std::string _g_token;
std::string m_server_host;
unsigned short m_server_port;

#if 1
static std::map<NOISE_MODEL, string> noise_model_mapping =
{ 
    {NOISE_MODEL::BITFLIP_KRAUS_OPERATOR,"BITFLIP_KRAUS_OPERATOR"},
    {NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR,"BIT_PHASE_FLIP_OPRATOR"},
    {NOISE_MODEL::DAMPING_KRAUS_OPERATOR,"DAMPING_KRAUS_OPERATOR"},
    {NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR,"DECOHERENCE_KRAUS_OPERATOR"},
    /*{NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR_P1_P2,"DECOHERENCE_KRAUS_OPERATOR_P1_P2"},*/ /**< 集群不支持 */
    {NOISE_MODEL::DEPHASING_KRAUS_OPERATOR,"DEPHASING_KRAUS_OPERATOR"},
    {NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR,"DEPOLARIZING_KRAUS_OPERATOR"},
    /*{NOISE_MODEL::KRAUS_MATRIX_OPRATOR,"KRAUS_MATRIX_OPRATOR"},*//**< 集群不支持 */
    /*{NOISE_MODEL::MIXED_UNITARY_OPRATOR,"MIXED_UNITARY_OPRATOR"},*//**< 集群不支持 */
    /*{NOISE_MODEL::PAULI_KRAUS_MAP,"PAULI_KRAUS_MAP"},*//**< 集群不支持 */
    {NOISE_MODEL::PHASE_DAMPING_OPRATOR,"PHASE_DAMPING_OPRATOR"}
};
#endif

/* response data buffer */
struct ResponseData
{
    char* m_data;
    uint32_t m_data_len;
    uint32_t m_buf_len;

    ResponseData()
        :m_data(nullptr), m_data_len(0), m_buf_len(0)
    {}

    void append(void* tmp_data, const size_t& len)
    {
        const size_t idle_size = m_buf_len - m_data_len;
        if (len > idle_size)
        {
            const size_t need_resize_len = len - idle_size + 4;
            m_data = static_cast<char*>(realloc(m_data, m_buf_len + need_resize_len));
            if (nullptr == m_data) {
                std::cerr << "Error: malloc error." << std::endl;
                return;
            }

            memset(m_data + m_buf_len, 0, need_resize_len);
            m_buf_len += need_resize_len;
        }

        memcpy(m_data + m_data_len, tmp_data, len);
        m_data_len += len;
    }

    ~ResponseData() {
        if (nullptr != m_data){
            free(m_data);
        }
    }
};

/*******************************************************************
*                      static interface
********************************************************************/
/* Process the data returned from the server and copy it to \p arg */
static std::string add_flag_of_PilotOSMachine(const std::string& origin_string)
{
    rapidjson::Document document;
    document.Parse(origin_string.c_str());

    if (document.HasParseError()) {
        std::cerr << "Error parsing JSON: " << document.GetParseError() << std::endl;
        return "";
    }

    rapidjson::Value::StringRefType key("PilotOSMachineFlag");
    rapidjson::Value value(true);
    document.AddMember(key, value, document.GetAllocator());

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);

    return buffer.GetString();
}

static size_t deal_response(void* ptr, size_t n, size_t m, void* arg)
{
    const size_t count = m * n;
    ResponseData* response_data = (ResponseData*)arg;
    response_data->append(ptr, count);

    return count;
}

static size_t _receive_data_cb(void* contents, size_t size, size_t nmemb, void* stream)
{
    if (stream)
    {
        string* str = (string*)stream;
        str->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
    return 0;
}

static QuantumHamiltonianData _json_to_hamiltonian(const std::string& hamiltonian_json)
{
    QuantumHamiltonianData result;
    Document doc;
    if (doc.Parse(hamiltonian_json.c_str()).HasParseError())
    {
        return result;
    }

    try
    {
        const rapidjson::Value& hamiltonian_value_array = doc["hamiltonian_value"];
        const rapidjson::Value& hamiltonian_param_array = doc["hamiltonian_param"];
        const rapidjson::Value& pauli_type_array = hamiltonian_value_array["pauli_type"];
        const rapidjson::Value& pauli_parm_array = hamiltonian_value_array["pauli_parm"];

        if (pauli_type_array.Size() != pauli_parm_array.Size())
        {
            return result;
        }

        for (SizeType i = 0; i < pauli_type_array.Size(); ++i)
        {
            std::map<size_t, char> qterm;
            const rapidjson::Value& pauli_type_value = pauli_type_array[i];
            const rapidjson::Value& pauli_parm_value = pauli_parm_array[i];
            const rapidjson::Value& hamiltonian_parm = hamiltonian_param_array[i];

            if (pauli_type_value.Size() != pauli_parm_value.Size())
            {
                return result;
            }

            for (SizeType j = 0; j < pauli_type_value.Size(); ++j)
            {
                size_t pauli_parm = pauli_parm_value[j].GetInt();
                string pauli_type = pauli_type_value[j].GetString();
                qterm.insert(std::make_pair(pauli_parm, (pauli_type[0])));
            }
            result.emplace_back(std::make_pair(qterm, hamiltonian_parm.GetDouble()));
        }
        return result;
    }
    catch (std::exception& e) {
        throw;
    }
    return result;
}

static ErrorCode _parser_expectation_result(JsonParser& json_parser, double &result)
{
    ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
    if (json_parser.has_member("errCode")) {
        err_code = (ErrorCode)json_parser.get_uint32("errCode");
    }

    if (json_parser.has_member("taskResult"))
    {
        const auto _result_str = json_parser.get_string("taskResult");
        std::regex rx("-\\d+\\.\\d+");
        bool matched = std::regex_match(_result_str.begin(), _result_str.end(), rx);
        if(matched)
        {
            result = std::stod(_result_str);
        }
        else
        {
            result = 0;
            err_code = ErrorCode::UNDEFINED_ERROR;
            PTraceInfo("Invalid value: " + _result_str + " for expectation");
        }
    }
    return err_code;
}

static ErrorCode parser_expectation_result_vec(JsonParser& json_parser, std::vector<double>& result)
{
    ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
    if (json_parser.has_member("errCode")) 
    {
        err_code = (ErrorCode)json_parser.get_uint32("errCode");
    }

    if (json_parser.has_member("taskResult"))
    {
        const auto _result_str = json_parser.get_array("taskResult");
        std::regex rx("-?\\d+\\.\\d+");
        uint32_t size = _result_str.size();
        double res;
        for (uint32_t i = 0; i < size; ++i)
        {
            if (std::regex_match(_result_str[i].begin(), _result_str[i].end(), rx))
            {
                res = std::stod(_result_str[i]);
                result.emplace_back(res);
            }
            else
            {
                err_code = ErrorCode::UNDEFINED_ERROR;
                PTraceInfo("Invalid value: " + _result_str[i] + " for expectation");
            }
        }
    }
    return err_code;
}

static ErrorCode _parser_probability_result(JsonParser& json_parser, std::map<std::string, double>& result)
{
    ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
    if (json_parser.has_member("errCode")) 
    {
        err_code = (ErrorCode)json_parser.get_uint32("errCode");
    }

    if (json_parser.has_member_string("taskResult"))
    {
        const auto _result_str = json_parser.get_string("taskResult");
        JsonParser _json_parser_result;
        if (!_json_parser_result.load_json(_result_str.c_str())) 
        {
            return ErrorCode::JSON_FIELD_ERROR;
        }

        std::vector<std::string> key_vec;
        if (_json_parser_result.has_member(MSG_DICT_KEY)) 
        {
            _json_parser_result.get_array(MSG_DICT_KEY, key_vec);
        }
        else if (_json_parser_result.has_member("Key")) 
        {
            _json_parser_result.get_array("Key", key_vec);
        }

        std::vector<double> val_vec;
        if (_json_parser_result.has_member(MSG_DICT_VALUE)) 
        {
            _json_parser_result.get_array(MSG_DICT_VALUE, val_vec);
        }
        else if (_json_parser_result.has_member("Value")) 
        {
            _json_parser_result.get_array("Value", val_vec);
        }

        for (size_t i = 0; i < key_vec.size(); ++i) 
        {
            result.insert(std::make_pair(key_vec[i], val_vec[i]));
        }
    }
    else if (json_parser.has_member_array("taskResult"))
    {
        auto& doc = json_parser.get_json_obj();

        for (auto& iter : doc["taskResult"].GetArray())
        {
            if (iter.IsString())
            {
                std::string _result_str = iter.GetString();
                JsonParser _json_parser_result;
                if (!_json_parser_result.load_json(_result_str.c_str())) 
                {
                    return ErrorCode::JSON_FIELD_ERROR;
                }

                std::vector<std::string> key_vec;
                if (_json_parser_result.has_member(MSG_DICT_KEY)) 
                {
                    _json_parser_result.get_array(MSG_DICT_KEY, key_vec);
                }
                else if (_json_parser_result.has_member("Key")) 
                {
                    _json_parser_result.get_array("Key", key_vec);
                }

                std::vector<double> val_vec;
                if (_json_parser_result.has_member(MSG_DICT_VALUE)) 
                {
                    _json_parser_result.get_array(MSG_DICT_VALUE, val_vec);
                }
                else if (_json_parser_result.has_member("Value")) 
                {
                    _json_parser_result.get_array("Value", val_vec);
                }

                for (size_t i = 0; i < key_vec.size(); ++i) 
                {
                    result.insert(std::make_pair(key_vec[i], val_vec[i]));
                }
            }         
            else
            {
                PTraceError("Result json error: < In value: taskResult no string >!!");
                return ErrorCode::JSON_FIELD_ERROR;
            }
        }
        //const auto _result_str = json_parser.get_string("taskResult");        
    }
    else
    {
        PTraceError("Result json error: < Not string or array >!!");
        return ErrorCode::JSON_FIELD_ERROR;
    }
    std::cout << "------------------- get time info ---------------------------" << std::endl;
    if (json_parser.has_member("aioExecuteTime")) 
    {
        uint64_t aio_time = json_parser.get_uint64("aioExecuteTime");
        std::cout << "aioExecuteTime: " << aio_time << std::endl;
    }
    if (json_parser.has_member("queueTime")) 
    {
        uint64_t queue_time = json_parser.get_uint64("queueTime");
        std::cout << "queueTime: " << queue_time << std::endl;
    }
    if (json_parser.has_member("compileTime"))
    {
        uint64_t compile_time = json_parser.get_uint64("compileTime");
        std::cout << "compileTime: " << compile_time << std::endl;
    }
    if (json_parser.has_member("totalTime"))
    {
        uint64_t total_time = json_parser.get_uint64("totalTime");
        std::cout << "totalTime: " << total_time << std::endl;
    }
    return err_code;
}

static ErrorCode _parser_probability_result_vec(JsonParser& json_parser, std::vector<std::map<std::string, double>>& result)
{
    ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
    if (json_parser.has_member("errCode")) {
        err_code = (ErrorCode)json_parser.get_uint32("errCode");
        if (static_cast<int>(err_code) != 0)
        {
            PTraceInfo("Task error! errCode: " << static_cast<int>(err_code));
            std::cout << "Task error! errCode: " << static_cast<int>(err_code);
            if (json_parser.has_member("errInfo"))
            {
                std::string err_info = json_parser.get_string("errInfo");
                PTraceInfo("errInfo: " << err_info);
                std::cout << "errInfo: " << err_info;
            }
            std::cout << std::endl;
        }
    }

    if (json_parser.has_member_array("taskResult"))
    {
        const auto _result_str = json_parser.get_array("taskResult");
        uint32_t size = _result_str.size();
        JsonParser _json_parser_result;
        for (uint32_t i = 0; i < size; i++)
        {
            if (!_json_parser_result.load_json(_result_str[i])) {
                return ErrorCode::JSON_FIELD_ERROR;
            }

            std::vector<std::string> key_vec;
            if (_json_parser_result.has_member(MSG_DICT_KEY)) {
                _json_parser_result.get_array(MSG_DICT_KEY, key_vec);
            }
            else if (_json_parser_result.has_member("Key")) {
                _json_parser_result.get_array("Key", key_vec);
            }

            std::vector<double> val_vec;
            if (_json_parser_result.has_member(MSG_DICT_VALUE)) {
                _json_parser_result.get_array(MSG_DICT_VALUE, val_vec);
            }
            else if (_json_parser_result.has_member("Value")) {
                _json_parser_result.get_array("Value", val_vec);
            }
            std::map<std::string, double>temp_res;

            for (size_t i = 0; i < key_vec.size(); ++i) {
                temp_res.insert(std::make_pair(key_vec[i], val_vec[i]));                
            }
            result.push_back(temp_res);
        }
    }
    return err_code;
}

static ErrorCode _parser_prob_count_result_vec(JsonParser& json_parser, std::vector<std::map<std::string, size_t>>& result)
{
    ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
    if (json_parser.has_member("errCode")) {
        err_code = (ErrorCode)json_parser.get_uint32("errCode");
    }

    if (json_parser.has_member_array("probCount"))
    {
        const auto _result_str = json_parser.get_array("probCount");
        uint32_t size = _result_str.size();
        JsonParser _json_parser_result;
        for (uint32_t i = 0; i < size; i++)
        {
            if (!_json_parser_result.load_json(_result_str[i])) {
                return ErrorCode::JSON_FIELD_ERROR;
            }

            std::vector<std::string> key_vec;
            if (_json_parser_result.has_member(MSG_DICT_KEY)) {
                _json_parser_result.get_array(MSG_DICT_KEY, key_vec);
            }
            else if (_json_parser_result.has_member("Key")) {
                _json_parser_result.get_array("Key", key_vec);
            }

            std::vector<size_t> val_vec;
            if (_json_parser_result.has_member(MSG_DICT_VALUE)) {
                _json_parser_result.get_array(MSG_DICT_VALUE, val_vec);
            }
            else if (_json_parser_result.has_member("Value")) {
                _json_parser_result.get_array("Value", val_vec);
            }
            std::map<std::string, size_t>temp_res;

            for (size_t i = 0; i < key_vec.size(); ++i) {
                temp_res.insert(std::make_pair(key_vec[i], val_vec[i]));
            }
            result.push_back(temp_res);
        }
    }
    return err_code;
}

static ErrorCode _parser_task_result_json(JsonParser& json_parser, std::string& result)
{
    ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
    try{
        result = object_to_string(json_parser.get_json_obj());
    }
    catch (const std::exception& e)
    {
        PTraceError("Catch json exception" << e.what());
        err_code = ErrorCode::JSON_FIELD_ERROR;
    }
    return err_code;
}

/*******************************************************************
*                      class QPilotMachine
********************************************************************/
namespace PilotQVM {
    class QPilotMachineImp
    {
        enum ClusterResultType
        {
            PROBABILITY = 1,
            AMPLITUDE = 2,
            EXPECTATION = 4,
        };
        friend class QPilotMachine;

    public:
        QPilotMachineImp()
        {
#ifdef USE_CURL
            curl_global_init(CURL_GLOBAL_ALL);
#endif
        }

        virtual ~QPilotMachineImp()
        {
#ifdef USE_CURL
            curl_global_cleanup();
#endif            
            m_tcp_thread.~thread();
        }

        std::string& get_token_str()
        {
            return m_token;
        }

    protected:
        bool init(const std::string& pilot_url,
            bool log_cout = false)
        {
            if (!m_b_init_ok)
            {
                m_log_cout = log_cout;
                m_pilot_url = pilot_url;
                PTraceInfo("pilot addr:" + m_pilot_url);

                m_server_host = m_pilot_url.substr(m_pilot_url.rfind(":") + 1, m_pilot_url.size() - 1);
                if (m_server_host.empty())
                {
                    PTraceError("pilotosmachine server error:invalid port");
                    return false;
                }
                m_server_port = (unsigned short)std::stoul(m_server_host);
                m_server_host = m_pilot_url.substr(m_pilot_url.find("//") + 2, m_pilot_url.rfind(":") - m_pilot_url.find("//") - 2);
                if (m_server_host.empty())
                {
                    PTraceError("pilotosmachine server error:invalid ip");
                    return false;
                }
                PTraceInfo("pilot tcp addr:" + m_server_host + " " + std::to_string(m_server_port));

                m_b_init_ok = true;
            }

            return m_b_init_ok;
        }

        ErrorCode parser_probability_result_vec(JsonParser& json_parser, std::vector<std::map<std::string, double>>& result)
        {
            return _parser_probability_result_vec(json_parser, result);
        }

        ErrorCode _parser_expectation_result_vec(JsonParser& json_parser, std::vector<double>& result)
        {
            return parser_expectation_result_vec(json_parser, result);
        }


#ifdef USE_CURL
        ErrorCode login_pilot_execute(const std::string& url, const std::string& json_task_msg)
        {
            PTraceInfo("login_pilot execut start...");
            std::string resp;
            for (size_t i = 0; (!curl_post(url, json_task_msg, resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + resp + ", url:" + url);
                if (++i > 10) {
                    return ErrorCode::JSON_FIELD_ERROR;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
            JsonParser json_parser_login_result;
            json_parser_login_result.load_json(resp);
            if (json_parser_login_result.has_member("errCode"))
            {
                auto errCode = json_parser_login_result.get_int32("errCode");
                if (errCode == 0) {

                    if (!json_parser_login_result.get_string("token").empty())
                    {
                        m_token = json_parser_login_result.get_string("token");
                    }
                    PTraceInfo("After login your token is: " + m_token);
                    return ErrorCode::NO_ERROR_FOUND;
                }
                else
                {
                    PTraceError("Login failed! response: " << resp);
                    return (ErrorCode)errCode;
                }
            }
            else
            {
                return ErrorCode::UNDEFINED_ERROR;
            }
        }
        
        bool query_task_state_execute(const std::string& url, const std::string& json_task_msg, PilotTaskQueryResult& result)
        {
            PTraceInfo("query_task_state_execute start...");
            std::string  str_resp;

            JsonParser json_parser_json_task_msg;
            json_parser_json_task_msg.load_json(json_task_msg);
            if (!json_parser_json_task_msg.has_member("taskId") || json_parser_json_task_msg.get_string("taskId").empty()) {
                return false;
            }
            std::string task_id = json_parser_json_task_msg.get_string("taskId");
            rabbit::object _obj;
            _obj.insert("taskId", task_id);
            if (!query_task_state_execute(url, _obj.str(), str_resp)) {
                return false;
            }

            /* Resolve the response information of the submitted task */
            PTraceInfo("on json parser:" + str_resp);
            JsonParser json_parser_task_state;
            json_parser_task_state.load_json(str_resp);
            if ((!json_parser_task_state.has_member("taskId")) || (!json_parser_task_state.has_member("taskState"))
                || (!json_parser_task_state.has_member("taskResult")) || (!json_parser_task_state.has_member("errCode"))
                || (!json_parser_task_state.has_member("errInfo")))
            {
                return false;
            }
            if (json_parser_task_state.has_member_array("taskResult")) {
                result.m_result_vec = json_parser_task_state.get_array("taskResult");
            }
            if (json_parser_task_state.has_member_string("taskResult")) {
                result.m_result = json_parser_task_state.get_string("taskResult");
            }
            if (json_parser_task_state.has_member_string("qSTResult")) {
                result.m_qst_density = json_parser_task_state.get_string("qSTResult");
            }
            if (json_parser_task_state.has_member_string("qSTFidelity")) {
                result.m_qst_fidelity = json_parser_task_state.get_string("qSTFidelity");
            }
            uint32_t size = result.m_result_vec.size();
            result.m_taskId = json_parser_task_state.get_string("taskId");
            result.m_state = json_parser_task_state.get_string("taskState");
            result.m_errCode = json_parser_task_state.get_int32("errCode");
            result.m_errInfo = json_parser_task_state.get_string("errInfo");
            result.m_resultJson = str_resp;
            if (result.m_state == "3") {
                std::cout << "------------------- get time info ---------------------------" << std::endl;
                if (json_parser_task_state.has_member("aioExecuteTime")) {
                    uint64_t aio_time = json_parser_task_state.get_uint64("aioExecuteTime");
                    std::cout << "aioExecuteTime: " << aio_time << std::endl;
                }
                if (json_parser_task_state.has_member("queueTime")) {
                    uint64_t queue_time = json_parser_task_state.get_uint64("queueTime");
                    std::cout << "queueTime: " << queue_time << std::endl;
                }
                if (json_parser_task_state.has_member("compileTime")) {
                    uint64_t compile_time = json_parser_task_state.get_uint64("compileTime");
                    std::cout << "compileTime: " << compile_time << std::endl;
                }
                if (json_parser_task_state.has_member("totalTime")) {
                    uint64_t total_time = json_parser_task_state.get_uint64("totalTime");
                    std::cout << "totalTime: " << total_time << std::endl;
                }
            }
            PTraceInfo("task state: " + result.m_state);
            //PTraceInfo("task result: " + result.result);
            return true;
        }

        bool query_task_state_execute(const std::string& url, const std::string& json_task_msg, std::string& resp)
        {
            PTraceInfo("query_task_state_execute start...");

            for (size_t i = 0; (!curl_post(url, json_task_msg, resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + resp + ", url:" + url);
                if (++i > 10) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
            return true;
        }

        bool query_compile_prog_execute(const std::string& url, const std::string& json_task_msg, bool & without_compensate, std::string& resp_data)
        {
            PTraceInfo("query_compile_prog_execut start...");

            for (size_t i = 0; (!curl_post(url, json_task_msg, resp_data)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + resp_data + ", url:" + url);
                if (++i > 10) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }

            JsonParser _json_parser;
            _json_parser.load_json(resp_data);
            if (_json_parser.has_member("errCode"))
            {
                auto errCode = _json_parser.get_string("errCode");
                if (errCode != "0")
                {
                    auto errInfo = _json_parser.get_string("errInfo");
                    std::cout << "errCode: " << errCode << std::endl << "errInfo: " << errInfo << std::endl;
                    PTraceError("Error: Task execute failed!! errInfo: " << errInfo);
                    return false;
                }
            }
            if (_json_parser.has_member("compile_output_prog") && _json_parser.has_member("without_compensate_prog")) {
                auto without_compensate_prog = _json_parser.get_string("without_compensate_prog");
                auto compile_output_prog = _json_parser.get_string("compile_output_prog");
                auto with_comp_size = without_compensate_prog.size();
                auto comp_prog_size = compile_output_prog.size();
                if (with_comp_size < 1 || comp_prog_size < 1)
                {
                    PTraceError("taskinfo does not have without_compensate_prog or compile_output_prog!!!");
                    std::cout << "taskinfo does not have without_compensate_prog or compile_output_prog!!!" << std::endl;
                    return false;
                }
                without_compensate ? resp_data = _json_parser.get_string("without_compensate_prog") : _json_parser.get_string("compile_output_prog");
            }

            else
            {
                PTraceError("Error: pilotos machine get query result msg fail");
                return false;
            }
            return true;
        }

        bool execute_query_task_state(const std::string& task_id, PilotTaskQueryResult& result)
        {
            PTraceInfo("execute_query_task_state start...");
            PTraceInfo("Task id : " + task_id);
            rabbit::object _obj;
            if (!task_id.empty()) {
                _obj.insert("taskId", task_id);
            }
            else {
                PTraceInfo("taskId is empty");
                return false;
            }
            if (!m_token.empty()) {
                _obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
                return false;
            }
            const std::string req = _obj.str();
            const std::string _url = m_pilot_url + TASK_QUERY_URL;

            return query_task_state_execute(_url, req, result);
        }

        bool execute_query_compile_prog(const std::string task_id, std::string& compile_prog, bool& without_compensate)
        {
            PTraceInfo("execute_query_task_state start...");
            PTraceInfo("Task id : " + task_id);
            rabbit::object _obj;
            _obj.insert("taskid", task_id);
            if (!m_token.empty()) {
                _obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }
            const std::string req = _obj.str();
            const std::string _url = m_pilot_url + COMPILE_PROG_QUERY_URL;

            return query_compile_prog_execute(_url, req, without_compensate, compile_prog);
        }

        ErrorCode execute_login_pilot_api(const std::string&api_key)
        {
            PTraceInfo("execute_login_pilot_with_api start...");
            rabbit::object _obj;
            _obj.insert("apiKey", api_key);
            //_obj.insert("PilotOSMachineFlag", true);
            const std::string req = _obj.str();
            const std::string _url = m_pilot_url + LOGIN_PILOT_API;
            return login_pilot_execute(_url, req);
        }

        ErrorCode execute_login_pilot(const std::string&username, const std::string&pwd)
        {
            PTraceInfo("execute_login_pilot start...");
            rabbit::object _obj;
            _obj.insert("username", username);
            _obj.insert("password", pwd);
            //_obj.insert("PilotOSMachineFlag", true);
            const std::string req = _obj.str();
            const std::string _url = m_pilot_url + LOGIN_PILOT;
            return login_pilot_execute(_url, req);
        }

        /* Synchronous execution, blocking calls */
        template<class ParserFunc, class... Args>
        ErrorCode synchronous_execute(const std::string& url, const std::string& json_task_msg,
            ParserFunc&& parser_func, Args && ... args)
        {
            PTraceInfo("synchronous_execute start...");
            std::string  str_resp;
            ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
            PTraceInfo("req_str: " + json_task_msg);
            for (size_t i = 0; (!curl_post(url, json_task_msg, str_resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + str_resp);
                if (++i > 10) {
                    return ErrorCode::ERR_TCP_INIT_FATLT;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }

            try
            {
                /* Resolve the response information of the submitted task */
                PTraceInfo("on json parser:" + str_resp);
                JsonParser json_parser_task_id;
                json_parser_task_id.load_json(str_resp);
                err_code = static_cast<ErrorCode>(json_parser_task_id.get_uint32("errCode"));
                if (!json_parser_task_id.has_member("taskId") || static_cast<uint32_t>(err_code) != 0)
                {
                    std::string errInfo = json_parser_task_id.get_string("errInfo");
                    PTraceError("Task failed, errCode: " << static_cast<uint32_t>(err_code) << ", errInfo: " << errInfo);
                    std::cout << "Task failed, errCode: " << static_cast<uint32_t>(err_code) << ", errInfo: " << errInfo << std::endl;
                    return err_code;
                }
                std::string task_id = json_parser_task_id.get_string("taskId");
                PTraceInfo("task id:" << task_id);
                TCPClient tcp_client;

                tcp_client.init(m_server_host.c_str(), m_server_port + 1, task_id);
                tcp_client.send_data(task_id, TCPMsg::TcpMsgType::TASK_ID_MSG);

                tcp_client.run_heart_thread();

                std::string recv_msg;
                const bool b_recv_result_ok = tcp_client.wait_recv_task_result(recv_msg, task_id);
                tcp_client.stop_heart_thread();
                do {
                    if (b_recv_result_ok)
                    {
                        PTraceInfo("recved msg: " << recv_msg << " task id: " << task_id);
                        QPilotMachineImp::handle_recv_msg(recv_msg);

                        /* Blocking waiting for calculation result */
                        const std::string str_reult_json = get_result_blocking(task_id);
                        std::string errstring = "error";
                        if (str_reult_json.size()==5) {
                            break;
                        }
                        PTraceInfo("Got task_result for task " << task_id << ", task_result:" << str_reult_json);

                        /* Analytical calculation results */
                        JsonParser json_parser_result;
                        json_parser_result.load_json(str_reult_json);
                        err_code = parser_func(json_parser_result, std::forward<Args>(args)...);
                        return err_code;
                    }
                    else {
                        break;
                    }
                } while (false);
                size_t i = 0;
                PTraceInfo("task_State 1 is System received task.\ntask_State 2 is system calculating task.\ntask_State 5 is task in queue.");
                while (true)
                {
                    std::string queryUrl = m_pilot_url + TASK_QUERY_URL;
                    std::string str_query = str_resp;
                    std::string query_result;
                    query_task_state_execute(queryUrl, str_query, query_result);
                    i++;
                    PTraceInfo("On " << i << " times queryResult is :" << query_result);
                    JsonParser json_parser_result;
                    json_parser_result.load_json(query_result);
                    std::string taskState = json_parser_result.get_string("taskState");
                    size_t errCode = json_parser_result.get_int32("errCode");
                    std::string errInfo = json_parser_result.get_string("errInfo");
                    //std::string taskResult = json_parser_result.get_string("taskResult");

                    if (atoi(taskState.c_str()) == 1 || atoi(taskState.c_str()) == 2 || atoi(taskState.c_str()) == 5)
                    {
                        //std::cout << "Now task_State is: " << taskState << ". Please wait a while." << std::endl;
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        continue;
                    }
                    else if (errCode != 0)
                    {
                        PTraceInfo("Got task errInfo for task " << task_id << ", task_result:" << errInfo);
                        err_code = static_cast<ErrorCode>(errCode);
                        break;
                    }
                    else /*if (taskResult.size() > 1)*/
                    {
                        err_code = parser_func(json_parser_result, std::forward<Args>(args)...);
                        break;
                    }
                }
                //tcp_client.wait_for_close();
            }
            catch (const std::exception& e)
            {
                PTraceError("pilotos machine synchronous_execute fail: " << e.what());
                throw;
            }
            return err_code;
        }

        template<class ParserFunc, class... Args>
        ErrorCode synchronous_execute_vec(const std::string& url, const std::string& json_task_msg,
            ParserFunc&& parser_func, Args && ... args)
        {
            PTraceInfo("synchsronous_execute_vec start...");
            std::string  str_resp;
            ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
            PTraceInfo("req_str: " + json_task_msg);
            for (size_t i = 0; (!curl_post(url, json_task_msg, str_resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + str_resp);
                if (++i > 10) {
                    return ErrorCode::ERR_TCP_INIT_FATLT;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }

            try
            {
                /* Resolve the response information of the submitted task */
                PTraceInfo("on json parser:" + str_resp);
                JsonParser json_parser_task_id;
                json_parser_task_id.load_json(str_resp);
                err_code = static_cast<ErrorCode>(json_parser_task_id.get_uint32("errCode"));
                if (!json_parser_task_id.has_member("taskId") || static_cast<uint32_t>(err_code) != 0)
                {
                    std::string errInfo = json_parser_task_id.get_string("errInfo");
                    PTraceError("Task failed, errCode: " << static_cast<uint32_t>(err_code) << "errInfo: " << errInfo);
                    std::cout << "Task failed, errCode: " << static_cast<uint32_t>(err_code) << "errInfo: " << errInfo << std::endl;
                    return err_code;
                }
                std::string task_id = json_parser_task_id.get_string("taskId");
                PTraceInfo("task id:" << task_id);
                TCPClient tcp_client;

                tcp_client.init(m_server_host.c_str(), m_server_port + 1, task_id);
                tcp_client.send_data(task_id, TCPMsg::TcpMsgType::TASK_ID_MSG);

                tcp_client.run_heart_thread();

                std::string recv_msg;
                const bool b_recv_result_ok = tcp_client.wait_recv_task_result(recv_msg, task_id);
                tcp_client.stop_heart_thread();
                do {
                    if (b_recv_result_ok)
                    {
                        PTraceInfo("recved msg: " << recv_msg << " task id: " << task_id);
                        QPilotMachineImp::handle_recv_msg(recv_msg);

                        /* Blocking waiting for calculation result */
                        const std::string str_reult_json = get_result_blocking(task_id);
                        std::string errstring = "error";
                        if (str_reult_json.size() == 5) {
                            break;
                        }
                        PTraceInfo("Got task_result for task " << task_id << ", task_result:" << str_reult_json);

                        /* Analytical calculation results */
                        JsonParser json_parser_result;
                        json_parser_result.load_json(str_reult_json);
                        err_code = parser_func(json_parser_result, std::forward<Args>(args)...);
                        return err_code;
                    }
                    else {
                        break;
                    }
                } while (false);
                size_t i = 0;
                PTraceInfo("task_State 1 is System received task.\ntask_State 2 is system calculating task.\ntask_State 5 is task in queue.");
                while (true)
                {
                    std::string queryUrl = m_pilot_url + TASK_QUERY_URL;
                    std::string str_query = str_resp;
                    std::string query_result;
                    query_task_state_execute(queryUrl, str_query, query_result);
                    i++;
                    PTraceInfo("On " << i << " times queryResult is :" << query_result);
                    JsonParser json_parser_result;
                    json_parser_result.load_json(query_result);
                    std::string taskState = json_parser_result.get_string("taskState");
                    size_t errCode = json_parser_result.get_int32("errCode");
                    std::string errInfo = json_parser_result.get_string("errInfo");
                    //std::vector<std::string> taskResult = json_parser_result.get_array("taskResult");

                    if (atoi(taskState.c_str()) == 1 || atoi(taskState.c_str()) == 2 || atoi(taskState.c_str()) == 5)
                    {
                        //std::cout << "Now task_State is: " << taskState << ". Please wait a while." << std::endl;
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        continue;
                    }
                    else if (errCode != 0)
                    {
                        PTraceInfo("Got task errInfo for task " << task_id << ", task_result:" << errInfo);
                        err_code = static_cast<ErrorCode>(errCode);
                        break;
                    }
                    else /*if (taskResult.size() > 1)*/
                    {
                        err_code = parser_func(json_parser_result, std::forward<Args>(args)...);
                        break;
                    }
                }
                //tcp_client.wait_for_close();
            }
            catch (const std::exception& e)
            {
                PTraceError("pilotos machine synchronous_execute fail: " << e.what());
                throw;
            }
            return err_code;
        }

        ErrorCode send_request(const std::string& url, const std::string& json_task_msg, std::string& str_resp)
        {
            PTraceInfo("send_request start...");
            PTraceInfo("req_str: " + json_task_msg);
            for (size_t i = 0; (!curl_post(url, json_task_msg, str_resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + str_resp);
                if (++i > 10) {
                    return ErrorCode::ERR_TCP_INIT_FATLT;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        /*template<class ParserFunc, class... Args>
        ErrorCode synchronous_execute1(const std::string& url, const std::string& json_task_msg,
            ParserFunc&& parser_func, Args && ... args)
        {
            PTraceInfo("synchronous_execute1 start...");
            std::string  str_resp;
            ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
            PTraceInfo("req_str: " + json_task_msg);
            for (size_t i = 0; (!curl_post(url, json_task_msg, str_resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + str_resp);
                if (++i > 10) {
                    return ErrorCode::ERR_TCP_INIT_FATLT;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }

            try
            {
                PTraceInfo("on json parser:" + str_resp);
                JsonParser json_parser_task_id;
                json_parser_task_id.load_json(str_resp);
                err_code = static_cast<ErrorCode>(json_parser_task_id.get_uint32("errCode"));
                if (!json_parser_task_id.has_member_string("taskId")
                    || static_cast<uint32_t>(err_code) != 0)
                {
                    std::string errInfo = json_parser_task_id.get_string("errInfo");
                    PTraceError("Task failed, errCode: " << static_cast<uint32_t>(err_code) << "errInfo: " << errInfo);
                    std::cout << "Task failed, errCode: " << static_cast<uint32_t>(err_code) << "errInfo: " << errInfo << std::endl;
                    return err_code;
                }
                std::string task_id= json_parser_task_id.get_string("taskId");
                PTraceInfo("task id:" << task_id);
                TCPClient tcp_client;

                tcp_client.init(m_server_host.c_str(), m_server_port + 1, task_id);
                tcp_client.send_data(task_id, TCPMsg::TcpMsgType::TASK_ID_MSG);

                tcp_client.run_heart_thread();

                std::string recv_msg;
                const bool b_recv_result_ok = tcp_client.wait_recv_task_result(recv_msg, task_id);
                tcp_client.stop_heart_thread();
                if (b_recv_result_ok)
                {
                    PTraceInfo("recved msg: " << recv_msg << " task id: " << task_id);
                }
                if (b_recv_result_ok)
                {
                    QPilotMachineImp::handle_recv_msg(recv_msg);

                    const auto str_reult_json = get_result_blocking(task_id);
                    PTraceInfo("Got task_result for task " << task_id << ", task_result:" << str_reult_json);

                    JsonParser json_parser_result;
                    json_parser_result.load_json(str_reult_json);
                    err_code = parser_func(json_parser_result, std::forward<Args>(args)...);
                }
                else
                {
                    size_t i = 0;
                    PTraceInfo("task_State 1 is System received task.\ntask_State 2 is system calculating task.\ntask_State 5 is task in queue.");
                    while (true)
                    {
                        std::string queryUrl = m_pilot_url + TASK_QUERY_URL;
                        std::string str_query = str_resp;
                        std::string query_result;
                        query_task_state_execute(queryUrl, str_query, query_result);
                        i++;
                        PTraceInfo("On " << i << " times queryResult is :" << query_result);
                        JsonParser json_parser_result;
                        json_parser_result.load_json(query_result);
                        std::string taskState = json_parser_result.get_string("taskState");
                        size_t errCode = json_parser_result.get_int32("errCode");
                        std::string errInfo = json_parser_result.get_string("errInfo");
                        //std::vector<std::string> taskResult = json_parser_result.get_array("taskResult");

                        if (atoi(taskState.c_str()) == 1 || atoi(taskState.c_str()) == 2 || atoi(taskState.c_str()) == 5)
                        {
                            //std::cout << "Now task_State is: " << taskState << ". Please wait a while." << std::endl;
                            std::this_thread::sleep_for(std::chrono::seconds(1));
                            continue;
                        }
                        else if (errCode != 0)
                        {
                            PTraceInfo("Got task errInfo for task " << task_id << ", task_result:" << errInfo);
                            err_code = static_cast<ErrorCode>(errCode);
                            break;
                        }
                        else 
                        {
                            err_code = parser_func(json_parser_result, std::forward<Args>(args)...);
                            break;
                        }
                    }
                }
            }
            catch (const std::exception& e)
            {
                PTraceError("pilotos machine synchronous_execute fail: " << e.what());
                throw;
            }
            return err_code;
        }*/

        /* Asynchronous execution, non blocking call */
        template<class ParserFunc, class CbFunc, class ResultTy>
        ErrorCode asynchronous_execute_vec(const std::string& url, const std::string& json_task_msg,
            CbFunc&& cb_func, ParserFunc&& parser_func, ResultTy&& _)
        {
            PTraceInfo("asynchronous_execute_vec start...");
            std::string  str_resp;
            if (!curl_post(url, json_task_msg, str_resp))
            {
                PTraceInfo("pilotos machine curl post fail:" + str_resp);
                return ErrorCode::JSON_FIELD_ERROR;
            }

            std::string task_id;
            try
            {
                PTraceInfo("on json parser:" + str_resp);
                JsonParser json_parser_resp;
                json_parser_resp.load_json(str_resp);
                if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                    return ErrorCode::JSON_FIELD_ERROR;
                }
                else {
                    task_id = json_parser_resp.get_string("taskId");
                }
                PTraceInfo("task id:" << task_id);
            }
            catch (const std::exception& e)
            {
                PTraceInfo(std::string("pilotos machine synchronous_execute fail:") + e.what());
                throw;
            }

            if (cb_func)
            {
                auto cb_func_wrapper = [&, cb_func](std::shared_ptr<JsonParser> p_json_parser) {
                    ResultTy _result;
                    auto err_code = parser_func(*p_json_parser, _result);
                    cb_func(err_code, _result);
                };
                m_cb_func_map.emplace(task_id, cb_func_wrapper);
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        /* Asynchronous execution, non blocking call */
        template<class ParserFunc, class CbFunc, class ResultTy>
        ErrorCode asynchronous_execute(const std::string& url, const std::string& json_task_msg,
            CbFunc&& cb_func, ParserFunc&& parser_func, ResultTy&& _)
        {
            PTraceInfo("asynchronous_execute start...");
            std::string  str_resp;
            if (!curl_post(url, json_task_msg, str_resp))
            {
                PTraceInfo("pilotos machine curl post fail:" + str_resp);
                return ErrorCode::JSON_FIELD_ERROR;
            }

            std::string task_id;
            try
            {
                PTraceInfo("on json parser:" + str_resp);
                JsonParser json_parser_resp;
                json_parser_resp.load_json(str_resp);
                if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                    return ErrorCode::JSON_FIELD_ERROR;
                }
                else {
                    task_id = json_parser_resp.get_string("taskId");
                }
                PTraceInfo("task id:" << task_id);
            }
            catch (const std::exception& e)
            {
                PTraceInfo(std::string("pilotos machine synchronous_execute fail:") + e.what());
                throw;
            }

            if (cb_func)
            {
                auto cb_func_wrapper = [&, cb_func](std::shared_ptr<JsonParser> p_json_parser) {
                    ResultTy _result;
                    auto err_code = parser_func(*p_json_parser, _result);
                    cb_func(err_code, _result);
                };
                m_cb_func_map.emplace(task_id, cb_func_wrapper);
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        ErrorCode asynchronous_execute(const std::string& url, const std::string& json_task_msg, std::string& task_id)
        {
            std::string str_resp;
            PTraceInfo("asynchronous_execute start...");
            if (!curl_post(url, json_task_msg, str_resp))
            {
                PTraceInfo("pilotos machine curl post fail:" + str_resp);
                return ErrorCode::JSON_FIELD_ERROR;
            }
            PTraceInfo("str_resp : " + str_resp);
            JsonParser json_parser_resp;
            json_parser_resp.load_json(str_resp);
            if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                return ErrorCode::JSON_FIELD_ERROR;
            }
            else {
                task_id = json_parser_resp.get_string("taskId");
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        ErrorCode synchronous_noise_learning(const std::string& url, const std::string& json_task_msg, std::string& task_id)
        {
            PTraceInfo("synchronous_noise_learning start...");
            std::string  str_resp;
            ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
            PTraceInfo("req_str: " + json_task_msg);
            for (size_t i = 0; (!curl_post(url, json_task_msg, str_resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + str_resp);
                if (++i > 10) {
                    task_id = "task execute error";
                    return ErrorCode::ERR_TCP_INIT_FATLT;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
            PTraceInfo("str_resp: " + str_resp);
            JsonParser json_parser_resp;
            json_parser_resp.load_json(str_resp);
            if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                task_id = "task execute error";
                return ErrorCode::JSON_FIELD_ERROR;
            }
            else {
                task_id = json_parser_resp.get_string("taskId");
            }
            return err_code;
        }

        ErrorCode asynchronous_noise_learning(const std::string& url, const std::string& json_task_msg, std::string& task_id)
        {
            std::string str_resp;
            PTraceInfo("asynchronous_noise_learning start...");
            if (!curl_post(url, json_task_msg, str_resp))
            {
                PTraceInfo("pilotos machine curl post fail:" + str_resp);
                task_id = "task execute error";
                return ErrorCode::JSON_FIELD_ERROR;
            }
            PTraceInfo("str_resp : " + str_resp);
            JsonParser json_parser_resp;
            json_parser_resp.load_json(str_resp);
            if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                task_id = "task execute error";
                return ErrorCode::JSON_FIELD_ERROR;
            }
            else {
                task_id = json_parser_resp.get_string("taskId");
            }

            return ErrorCode::NO_ERROR_FOUND;
        }

        ErrorCode synchronous_em_compute(const std::string& url, const std::string& json_task_msg, std::string& task_id)
        {
            PTraceInfo("synchronous_em_compute start...");
            std::string  str_resp;
            ErrorCode err_code = ErrorCode::NO_ERROR_FOUND;
            PTraceInfo("req_str: " + json_task_msg);
            for (size_t i = 0; (!curl_post(url, json_task_msg, str_resp)); )
            {
                PTraceError("Error: pilotos machine curl post fail:" + str_resp);
                if (++i > 10) {
                    task_id = "task execute error";
                    return ErrorCode::ERR_TCP_INIT_FATLT;
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
            PTraceInfo("str_resp: " + str_resp);

            JsonParser json_parser_resp;
            json_parser_resp.load_json(str_resp);
            if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                task_id = "task execute error";
                return ErrorCode::JSON_FIELD_ERROR;
            }
            else {
                task_id = json_parser_resp.get_string("taskId");
            }

            return err_code;
        }

        ErrorCode asynchronous_em_compute(const std::string& url, const std::string& json_task_msg, std::string& task_id)
        {
            std::string str_resp;
            PTraceInfo("asynchronous_em_compute start...");
            if (!curl_post(url, json_task_msg, str_resp))
            {
                PTraceInfo("pilotos machine curl post fail:" + str_resp);
                task_id = "task execute error";
                return ErrorCode::JSON_FIELD_ERROR;
            }
            PTraceInfo("str_resp : " + str_resp);
            JsonParser json_parser_resp;
            json_parser_resp.load_json(str_resp);
            if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                task_id = "task execute error";
                return ErrorCode::JSON_FIELD_ERROR;
            }
            else {
                task_id = json_parser_resp.get_string("taskId");
            }

            return ErrorCode::NO_ERROR_FOUND;
        }

        ErrorCode asynchronous_execute_vec(const std::string& url, const std::string& json_task_msg, std::string& task_id)
        {
            std::string str_resp;
            PTraceInfo("asynchronous_execute_vec start...");
            PTraceInfo("json_task_msg : " + json_task_msg);
            if (!curl_post(url, json_task_msg, str_resp))
            {
                PTraceInfo("pilotos machine curl post fail:" + str_resp);
                return ErrorCode::JSON_FIELD_ERROR;
            }
            PTraceInfo("str_resp : " + str_resp);
            JsonParser json_parser_resp;
            json_parser_resp.load_json(str_resp);
            if (!json_parser_resp.has_member("taskId") || json_parser_resp.get_string("taskId").empty()) {
                return ErrorCode::JSON_FIELD_ERROR;
            }
            else {
                task_id = json_parser_resp.get_string("taskId");
            }

            return ErrorCode::NO_ERROR_FOUND;
        }
        
        ErrorCode execute_expectation_task(const CalcConfig &config, const std::vector<uint32_t> &qubits, std::vector<double>& result)
        {
            PTraceInfo("execute_expectation_task start...");
            const auto json_task_msg = build_chip_expectation_task_json_msg(config, qubits);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
            return synchronous_execute(_url, json_task_msg, parser_expectation_result_vec, result);
        }

        //ErrorCode execute_expectation_task_vec(const CalcConfig& config, const std::vector<uint32_t>& qubits, std::vector<double>& result)
        //{
        //    PTraceInfo("execute_expectation_task start...");
        //    const auto json_task_msg = build_chip_expectation_task_json_msg_vec(config, qubits);
        //    const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
        //    return synchronous_execute_vec(_url, json_task_msg, parser_expectation_result_vec, result);
        //}

        std::string async_execute_expectation_task(const CalcConfig& config, const std::vector<uint32_t> &qubits, std::vector<double>& result)
        {
            PTraceInfo("async_execute_expectation_task start...");
            const auto json_task_msg = build_chip_expectation_task_json_msg(config, qubits);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;

            std::string task_id;
            asynchronous_execute(_url, json_task_msg, task_id);
            return task_id;
        }

        /*std::string async_execute_expectation_task_vec(const CalcConfig& config, const std::vector<uint32_t>& qubits, std::vector<double>& result)
        {
            PTraceInfo("async_execute_expectation_task_vec start...");
            const auto json_task_msg = build_chip_expectation_task_json_msg_vec(config, qubits);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;

            std::string task_id;
            asynchronous_execute_vec(_url, json_task_msg, task_id);
            return task_id;
        }*/

        ErrorCode execute_measure_task(const CalcConfig &config, std::map<std::string, double> &result)
        {
            PTraceInfo("execute_measure_task start...");
            const auto json_task_msg = build_chip_measure_task_json_msg(config);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
            return synchronous_execute(_url, json_task_msg, _parser_probability_result, result);
        }

        ErrorCode execute_measure_task_vec(const CalcConfig &config, std::vector<std::map<std::string, double>> &result_vec)
        {
            PTraceInfo("execute_measure_task_vec start...");
            const auto json_task_msg = build_chip_measure_task_json_msg_vec(config);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
            return synchronous_execute_vec(_url, json_task_msg, _parser_probability_result_vec, result_vec);
        }

        ErrorCode execute_measure_task_vec(const CalcConfig &config, std::vector<std::map<std::string, size_t>> &result_vec)
        {
            PTraceInfo("execute_measure_task_prob_count start...");
            const auto json_task_msg = build_chip_measure_task_json_msg_vec(config);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
            return synchronous_execute_vec(_url, json_task_msg, _parser_prob_count_result_vec, result_vec);
        }

        ErrorCode execute_measure_task_vec(const CalcConfig &config, std::string &result_json)
        {
            PTraceInfo("execute_measure_task start...");
            const auto json_task_msg = build_chip_measure_task_json_msg_vec(config);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
            return synchronous_execute_vec(_url, json_task_msg, _parser_task_result_json, result_json);
        }

        //ErrorCode execute_measure_task_vec(const CalcConfig& config, std::map<std::string, double>& result_vec)
        //{
        //    PTraceInfo("execute_measure_task start...");
        //    const auto json_task_msg = build_chip_measure_task_json_msg(config);
        //    const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
        //    return synchronous_execute_vec(_url, json_task_msg, _parser_probability_result_vec, result_vec);
        //}

        //ErrorCode asynch_execute_measure_task(const NoiseLearning& noise_learning, std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func)
        //{
        //    PTraceInfo("execute_measure_task start...");
        //    const auto json_task_msg = build_chip_measure_task_json_msg(config);
        //    const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
        //    return asynchronous_execute(_url, json_task_msg, cb_func, _parser_probability_result, std::map<std::string, double>());
        //}

        //std::string async_execute_measure_task(const CalcConfig &config, std::map<std::string, double> &result)
        //{
        //    PTraceInfo("async_execute_measure_task start...");
        //    const auto json_task_msg = build_chip_measure_task_json_msg(config);
        //    const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
        //
        //    std::string task_id;
        //    asynchronous_execute(_url, json_task_msg, task_id);
        //    PTraceInfo("Task id: " + task_id);
        //    return task_id;
        //}

        std::string async_execute_measure_task(const CalcConfig& config)
        {
            PTraceInfo("async_execute_measure_task start...");
            const auto json_task_msg = build_chip_measure_task_json_msg(config);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;

            std::string task_id;
            asynchronous_execute(_url, json_task_msg, task_id);
            PTraceInfo("Task id: " + task_id);
            return task_id;
        }

        ErrorCode async_execute_measure_task_vec(const CalcConfig &config, std::string& task_id)
        {
            PTraceInfo("async_execute_measure_task_vec start...");
            const auto json_task_msg = build_chip_measure_task_json_msg_vec(config);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
            PTraceInfo("msg: " + json_task_msg);
            return asynchronous_execute_vec(_url, json_task_msg, task_id);
        }

        ErrorCode execute_measure_task(const CalcConfig &config, std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func)
        {
            PTraceInfo("execute_measure_task start...");
            const auto json_task_msg = build_chip_measure_task_json_msg(config);
            const std::string _url = m_pilot_url + REAL_CHIP_COMPUTE_URL;
            return asynchronous_execute(_url, json_task_msg, cb_func, _parser_probability_result, std::map<std::string, double>());
        }

        ErrorCode execute_noise_learning_task(const std::string& parameter_json, std::string& task_id)
        {
            PTraceInfo("execute_noise_learning_task start...");
            try
            {
                const auto json_task_msg = build_chip_noise_learning_task_json_msg(parameter_json);
                const std::string _url = m_pilot_url + NOISE_LEARNING;
                return synchronous_noise_learning(_url, json_task_msg, task_id);
            }
            catch (const std::exception& e)
            {
                PTraceError(e.what());
                return ErrorCode::ERR_PARAMETER;
            }
            
        }

        ErrorCode async_execute_em_compute_task(const std::string& parameter_json, std::string& task_id)
        {
            PTraceInfo("async_execute_em_compute_task start...");
            const auto json_task_msg = build_chip_em_compute_task_json_msg(parameter_json);
            const std::string _url = m_pilot_url + NOISE_QCIR;
            PTraceInfo("msg: " + json_task_msg);
            //return asynchronous_em_compute(_url, json_task_msg, task_id);
            return asynchronous_execute_vec(_url, json_task_msg, task_id);
        }

        ErrorCode execute_em_compute_task(const std::string& parameter_json, std::string& task_id, std::vector<double>& result)
        {
            PTraceInfo("execute_em_compute_task start...");
            try 
            {
                const auto json_task_msg = build_chip_em_compute_task_json_msg(parameter_json);
                const std::string _url = m_pilot_url + NOISE_QCIR;
                return synchronous_execute(_url, json_task_msg, parser_expectation_result_vec, result);
                //return synchronous_em_compute(_url, json_task_msg, task_id);
            }
            catch (const std::exception& e)
            {
                PTraceError(e.what());
                return ErrorCode::ERR_PARAMETER;
            }
        }

        ErrorCode execute_full_amplitude_measure_task(const std::string& prog_str,
            std::map<std::string, double>& result,
            const uint32_t& cluster_id,
            const uint32_t& shots)
        {
            PTraceInfo("execute_full_amplitude_measure_task start...");
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            const auto json_task_msg = build_full_amplitude_measure_task_json_msg(prog_str, cluster_id, shots);
            return synchronous_execute(_url, json_task_msg, _parser_probability_result, result);
        }

        ErrorCode execute_full_amplitude_measure_task(const std::string& prog_str,
            std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func,
            const uint32_t& cluster_id,
            const uint32_t& shots)
        {
            PTraceInfo("execute_full_amplitude_measure_task start...");
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            const auto json_task_msg = build_full_amplitude_measure_task_json_msg(prog_str, cluster_id, shots);
            return asynchronous_execute(_url, json_task_msg, cb_func, _parser_probability_result, std::map<std::string, double>());
        }

        ErrorCode execute_full_amplitude_pmeasure_task(const std::string& prog_str,
            const std::vector<uint32_t>& qubit_vec,
            std::map<std::string, double>& result,
            const uint32_t& cluster_id)
        {
            PTraceInfo("execute_full_amplitude_pmeasure_task start...");
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            const std::string json_task_msg = build_full_amplitude_pmeasure_task_json_msg(prog_str, qubit_vec, cluster_id);
            return synchronous_execute(_url, json_task_msg, _parser_probability_result, result);
        }

        ErrorCode execute_full_amplitude_pmeasure_task(const std::string& prog_str,
            const std::vector<uint32_t>& qubit_vec,
            std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func,
            const uint32_t& cluster_id)
        {
            PTraceInfo("execute_full_amplitude_pmeasure_task start...");
            const std::string json_task_msg = build_full_amplitude_pmeasure_task_json_msg(prog_str,
                qubit_vec, cluster_id);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            return asynchronous_execute(_url, json_task_msg, cb_func,
                _parser_probability_result, std::map<std::string, double>());
        }

        ErrorCode execute_noise_measure_task(const std::string& prog_str,
            const PilotNoiseParams& noise_params,
            std::map<std::string, double>& result,
            const uint32_t& cluster_id,
            const uint32_t& shots)
        {
            PTraceInfo("execute_noise_measure_task start...");
            const std::string json_task_msg = build_noise_measure_task_json_msg(prog_str,
                noise_params, cluster_id, shots);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            return synchronous_execute(_url, json_task_msg, _parser_probability_result, result);
        }

        ErrorCode execute_noise_measure_task(const std::string& prog_str,
            const PilotNoiseParams& noise_params,
            std::function<void(ErrorCode, std::map<std::string, double>&)> cb_func,
            const uint32_t& cluster_id,
            const uint32_t& shots)
        {
            PTraceInfo("execute_noise_measure_task start...");
            const std::string json_task_msg = build_noise_measure_task_json_msg(prog_str,
                noise_params, cluster_id, shots);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            return asynchronous_execute(_url, json_task_msg, cb_func,
                _parser_probability_result, std::map<std::string, double>());
        }

        ErrorCode execute_partial_amplitude_task(const std::string& prog_str,
            const std::vector<std::string>& target_amplitude_vec,
            std::map<std::string, Complex_>& result,
            const uint32_t& cluster_id)
        {
            PTraceInfo("execute_partial_amplitude_task start...");
            const std::string json_task_msg = build_partial_amplitude_task_json_msg(prog_str,
                target_amplitude_vec, cluster_id);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            return synchronous_execute(_url, json_task_msg,
                std::bind(&QPilotMachineImp::parser_amplitude_result, this, std::placeholders::_1, std::placeholders::_2), result);
        }

        ErrorCode execute_partial_amplitude_task(const std::string& prog_str,
            const std::vector<std::string>& target_amplitude_vec,
            std::function<void(ErrorCode, const std::map<std::string, Complex_>&)> cb_func,
            const uint32_t& cluster_id)
        {
            PTraceInfo("execute_partial_amplitude_task start...");
            const std::string json_task_msg = build_partial_amplitude_task_json_msg(prog_str,
                target_amplitude_vec, cluster_id);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            return asynchronous_execute(_url, json_task_msg, cb_func,
                std::bind(&QPilotMachineImp::parser_amplitude_result, this, std::placeholders::_1, std::placeholders::_2),
                std::map<std::string, Complex_>());
        }

        ErrorCode execute_single_amplitude_task(const std::string& prog_str,
            const std::string& target_amplitude,
            Complex_& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND)
        {
            PTraceInfo("execute_single_amplitude_task start...");
            const std::string json_task_msg = build_single_amplitude_task_json_msg(prog_str, target_amplitude, cluster_id);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            return synchronous_execute(_url, json_task_msg,
                std::bind(&QPilotMachineImp::parser_single_amplitude_result, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                target_amplitude, result);
        }

        ErrorCode execute_single_amplitude_task(const std::string& prog_str,
            const std::string& target_amplitude,
            std::function<void(ErrorCode, const Complex_&)> cb_func,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND)
        {
            PTraceInfo("execute_single_amplitude_task start...");
            const std::string json_task_msg = build_single_amplitude_task_json_msg(prog_str, target_amplitude, cluster_id);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            auto parser_func =
                std::bind(&QPilotMachineImp::parser_single_amplitude_result, this, std::placeholders::_1, target_amplitude, std::placeholders::_2);
            return asynchronous_execute(_url, json_task_msg, cb_func, parser_func, Complex_());
        }

        ErrorCode execute_full_amplitude_expectation(const std::string& prog_str,
            const QuantumHamiltonianData& hamiltonian,
            const std::vector<uint32_t>& qubit_vec,
            double& result,
            const uint32_t& cluster_id)
        {
            PTraceInfo("execute_full_amplitude_expectation start...");
            const std::string json_task_msg = build_full_amplitude_expectation_task_json_msg(prog_str, hamiltonian, qubit_vec, cluster_id);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            PTraceInfo("url:" << _url << ", task_json_str:" << json_task_msg);
            return synchronous_execute(_url, json_task_msg,
                std::bind(&QPilotMachineImp::parser_expectation_result, this, std::placeholders::_1, std::placeholders::_2), result);
        }

        ErrorCode execute_full_amplitude_expectation(const std::string& prog_str,
            const QuantumHamiltonianData& hamiltonian,
            const std::vector<uint32_t>& qubit_vec,
            std::function<void(ErrorCode, double)> cb_func,
            const uint32_t& cluster_id)
        {
            PTraceInfo("execute_full_amplitude_expectation start...");
            const std::string json_task_msg = build_full_amplitude_expectation_task_json_msg(prog_str, hamiltonian, qubit_vec, cluster_id);
            const std::string _url = m_pilot_url + CLUSTER_COMPUTE_URL;
            return asynchronous_execute(_url, json_task_msg, cb_func,
                std::bind(&QPilotMachineImp::parser_expectation_result, this, std::placeholders::_1, std::placeholders::_2),
                double());
        }
#endif
    private:
        ErrorCode parser_single_amplitude_result(JsonParser& json_parser,
            const std::string& target_amplitude,
            Complex_& target_amplitude_result)
        {
            std::map<std::string, Complex_> total_result;
            const auto ret = parser_amplitude_result(json_parser, total_result);
            auto found_iter = total_result.find(target_amplitude);
            if (found_iter != total_result.end()) {
                target_amplitude_result = found_iter->second;
            }
            return ret;
        }

        ErrorCode parser_amplitude_result(JsonParser& json_parser,
            std::map<std::string, Complex_>& result)
        {
            try
            {
                /*rabbit::document _doc;
                _doc.parse(str_result_json);*/
                auto& _doc = json_parser.get_json_obj();
                const ErrorCode err_code = static_cast<ErrorCode>(_doc["errCode"].GetInt());
				if (err_code != ErrorCode::NO_ERROR_FOUND){
					return err_code;
				}

                rabbit::document  result_doc;
                if (_doc["taskResult"].IsArray()) 
                {
                    for (auto& iter : _doc["taskResult"].GetArray())
                    {
                        result_doc.parse(iter.GetString());
                        for (SizeType i = 0; i < result_doc["Key"].size(); ++i)
                        {
                            std::string bin_amplitude = result_doc["Key"][i].as_string();
                            auto real = result_doc["ValueReal"][i].as_double();
                            auto imag = result_doc["ValueImag"][i].as_double();
                            result.emplace(bin_amplitude, Complex_(real, imag));
                        }
                    }
                    
                }
                else if (_doc["taskResult"].IsString()) {
                    result_doc.parse(_doc["taskResult"].GetString());

                    for (SizeType i = 0; i < result_doc["Key"].size(); ++i)
                    {
                        std::string bin_amplitude = result_doc["Key"][i].as_string();
                        auto real = result_doc["ValueReal"][i].as_double();
                        auto imag = result_doc["ValueImag"][i].as_double();
                        result.emplace(bin_amplitude, Complex_(real, imag));
                    }
                }
            }
            catch (const std::exception& e)
            {
                PTraceInfo("pilotos machine parser_amplitude_result fail:" + (std::string)e.what());
                throw;
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        ErrorCode parser_qst_matrix_result(JsonParser& json_parser, std::vector<QStat>& result)
        {
            try
            {
                auto& _doc = json_parser.get_json_obj();
                const ErrorCode err_code = static_cast<ErrorCode>(_doc["errCode"].GetInt());
                if (err_code != ErrorCode::NO_ERROR_FOUND) {
                    return err_code;
                }

                rabbit::document  result_doc;
                result_doc.parse(_doc["qSTResult"].GetString());
                int rank = (int)std::sqrt(result_doc.size());
                for (auto i = 0; i < rank; ++i)
                {
                    QStat row_value;
                    for (auto j = 0; j < rank; ++j)
                    {
                        auto real = result_doc[i * rank + j]["r"].as_double();
                        auto imag = result_doc[i * rank + j]["i"].as_double();
                        row_value.emplace_back(Complex_(real, imag));
                    }
                    result.emplace_back(row_value);
                }
            }
            catch (const std::exception& e)
            {
                PTraceInfo("pilotos machine parser_qst_matrix_result fail:" + (std::string)e.what());
                throw;
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        ErrorCode parser_qst_fidelity_result(JsonParser& json_parser,
            std::vector<QStat>& qst_mat, double& qst_fidelity)
        {
            try
            {
                auto& _doc = json_parser.get_json_obj();
                const ErrorCode err_code = static_cast<ErrorCode>(_doc["errCode"].GetInt());
                if (err_code != ErrorCode::NO_ERROR_FOUND) {
                    return err_code;
                }

                rabbit::document  result_doc;
                result_doc.parse(_doc["qSTResult"].GetString());
                std::string qst_fidelity_str = result_doc["qSTFidelity"].as_string();
                qst_fidelity = stod(qst_fidelity_str);

                rabbit::document _qst_mat_doc;
                _qst_mat_doc.parse(_doc["qSTResult"].GetString());
                int rank = (int)std::sqrt(_qst_mat_doc.size());
                for (auto i = 0; i < rank; ++i)
                {
                    QStat row_value;
                    for (auto j = 0; j < rank; ++j)
                    {
                        auto real = _qst_mat_doc[i * rank + j]["r"].as_double();
                        auto imag = _qst_mat_doc[i * rank + j]["i"].as_double();
                        row_value.emplace_back(Complex_(real, imag));
                    }
                    qst_mat.emplace_back(row_value);
                }
            }
            catch (const std::exception& e)
            {
                PTraceInfo("pilotos machine parser_qst_fidelity_result fail:" + (std::string)e.what());
                throw;
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        ErrorCode parser_expectation_result(JsonParser& json_parser, double& expectation_val)
        {
            PTraceInfo("on parser_expectation_result.");
            try
            {
                auto& _doc = json_parser.get_json_obj();
                const ErrorCode err_code = static_cast<ErrorCode>(_doc["errCode"].GetInt());
                if (err_code != ErrorCode::NO_ERROR_FOUND) {
                    return err_code;
                }

                rabbit::document result_doc;
                result_doc.parse(_doc["taskResult"].GetString());
                auto result_type = static_cast<ClusterResultType>(result_doc["ResultType"].as_int());
                if (result_type != ClusterResultType::EXPECTATION)
                {
                    PTraceInfo("pilotos machine parser_expectation_result parser error");
                    return ErrorCode::JSON_FIELD_ERROR;
                }
                expectation_val = result_doc["Value"].as_double();
            }
            catch (const std::exception& e)
            {
                PTraceInfo("pilotos machine parser_expectation_result fail:" + (std::string)e.what());
                throw;
            }
            return ErrorCode::NO_ERROR_FOUND;
        }

        void handle_recv_msg(std::string &msg)
        {
            try
            {
                PTraceInfo("recv msg:"+msg);
                std::lock_guard<std::mutex> _guard(m_func_info_mutex);

                auto p_json_parser = std::make_shared<JsonParser>();
                if (!p_json_parser->load_json(msg.c_str())) {
                    PTraceInfo("pilotos machine handle_recv_msg parser error");
                    return;
                }

                std::string task_id;
                if (p_json_parser->has_member("taskId")) {
                    task_id = p_json_parser->get_string("taskId");
                }
                else {
                    PTraceInfo("pilotos machine handle_recv_msg parser error:can't find taskId");
                    return;
                }

                auto _found_iter = m_cb_func_map.find(task_id);
                if (_found_iter != m_cb_func_map.end()) {
                    _found_iter->second(p_json_parser);
                    m_cb_func_map.erase(_found_iter);
                }
                else
                {
                    PTraceInfo("find task:" + task_id);
					m_mutex.lock();
					m_task_info_map.emplace(task_id, msg);
					m_mutex.unlock();
                    m_cv.notify_all();
                }
            }
            catch (const std::exception& e) {
                PTraceInfo("pilotos machine handle_recv_msg fail:" + (std::string)e.what());
                throw;
            }
        }

        std::string get_result_blocking(const std::string& task_id)
        {
			size_t find_try_time = 0;
			size_t find_try_limit = 3;
			PTraceInfo("On get_result_blocking for task: " + task_id);
			while (find_try_time < find_try_limit)
			{
				m_mutex.lock();
				if (m_task_info_map.find(task_id) != m_task_info_map.end()) {
                    std::string result_msg = m_task_info_map.at(task_id);
                    m_task_info_map.erase(task_id);
					m_mutex.unlock();
                    PTraceInfo("Got msg for task "+ task_id +":" + result_msg);
                    return result_msg;
				}
				m_mutex.unlock();
                std::this_thread::sleep_for(3s);
				if ((find_try_time++) >= find_try_limit)
				{
                    PTraceInfo("Warn: time-out on get-result-blocking");
                    return "error";
					//throw std::runtime_error("get result error.");
				}				
			}            
        }
#ifdef USE_CURL
        bool curl_post(const std::string& str_url, const std::string& str_req, std::string& str_resp)
        {
            std::string send_str = add_flag_of_PilotOSMachine(str_req);
            if (send_str.size() == 0)
            {
                PTraceInfo("add PilotOSMachine flag failed!");
                return false;
            }
            char* req_data = nullptr;
            uint32_t req_data_len = 0;
            std::unique_ptr<char> compress_buf;
            if (send_str.length() > MAX_POST_SIZE)
            {
                /* Do compress */
#define ENABLE_COMPRESS 1
#if ENABLE_COMPRESS
                auto start = chrono::system_clock::now();
                const uint32_t compress_buf_size = 1024 * 1024 * 64;
                compress_buf.reset(new char[compress_buf_size]);
                //std::unique_ptr<char> compress_buf(new char[compress_buf_size]);
                memset(compress_buf.get(), 0, compress_buf_size);

                unsigned int compress_output_len = compress_buf_size;
                const auto compress_ret = BZ2_bzBuffToBuffCompress(compress_buf.get(), &compress_output_len,
                    const_cast<char*>(send_str.c_str()), send_str.length(), 9, 0, 0);
                if (compress_ret == BZ_OK) {
                    PTraceInfo("bz2 compress successed, data-len:" << send_str.length() << "->" << compress_output_len);
                    req_data = compress_buf.get();
                    req_data_len = compress_output_len;
                }
                else
                {
                    PTraceWarn("Warn: bz2 compress failed: " << compress_ret);
                    //strncpy(compress_buf.get(), send_str.c_str(), send_str.length());
                    //compress_output_len = send_str.size();
                    req_data = const_cast<char*>(send_str.c_str());
                    req_data_len = send_str.size();
                }

                auto end = chrono::system_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                PTraceInfo("The bz2 compress takes "
                    << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
                    << " s");
#else
#endif // ENABLE_COMPRESS
            }
            else {
                req_data = const_cast<char*>(send_str.c_str());
                req_data_len = send_str.size();
            }

			str_resp.clear();
			bool ret = true;
            struct curl_slist* headers = NULL;
            auto curl_handle = curl_easy_init();
            headers = curl_slist_append(headers, "Content-Type: application/json;charset=UTF-8");
            curl_slist_append(headers, "Server: nginx/1.16.1");
            curl_slist_append(headers, "Connection: keep-alive");
            curl_slist_append(headers, "Transfer-Encoding: chunked");
            curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, 30);
            curl_easy_setopt(curl_handle, CURLOPT_CONNECTTIMEOUT, 0);
            curl_easy_setopt(curl_handle, CURLOPT_URL, str_url.c_str());
            curl_easy_setopt(curl_handle, CURLOPT_HEADER, false);
            curl_easy_setopt(curl_handle, CURLOPT_POST, true);
            curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYHOST, false);
            curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYPEER, false);
            curl_easy_setopt(curl_handle, CURLOPT_READFUNCTION, NULL);
            curl_easy_setopt(curl_handle, CURLOPT_NOSIGNAL, 1);
            curl_easy_setopt(curl_handle, CURLOPT_POSTFIELDS, req_data);
            curl_easy_setopt(curl_handle, CURLOPT_POSTFIELDSIZE, req_data_len);

            //curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, _receive_data_cb);
            ////curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 1);
            //curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void*)&str_resp);

            curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, deal_response);

            ResponseData response_data_buf;
            curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &response_data_buf);

            auto res = curl_easy_perform(curl_handle);
            if (CURLE_OK != res) {
                ret = false;
                str_resp = std::string(curl_easy_strerror(res));
            }
            else
            {
                try
                {
                    if ((response_data_buf.m_data_len > 3)
                        && ('B' == response_data_buf.m_data[0])
                        && ('Z' == response_data_buf.m_data[1])
                        && ('h' == response_data_buf.m_data[2]))
                    {
                        /* Do decompress */
                        const uint32_t decompress_buf_size = 1024 * 1024 * 128;
                        unsigned int decompress_dest_len = decompress_buf_size;
                        std::unique_ptr<char> decompress_buf(new char[decompress_buf_size]);
                        memset(decompress_buf.get(), 0, decompress_buf_size);
                        const auto _decompress_ret = BZ2_bzBuffToBuffDecompress(decompress_buf.get(), &decompress_dest_len,
                            response_data_buf.m_data, response_data_buf.m_data_len, 0, 0);
                        if (_decompress_ret == BZ_OK) {
                            PTraceInfo("bz2 decompress successed, data-len:" << response_data_buf.m_data_len << "->" << decompress_dest_len);
                            str_resp = std::string(decompress_buf.get(), decompress_dest_len);
                        }
                        else
                        {
                            PTraceError("Error: bz2 decompress failed: " << _decompress_ret);
                            str_resp = std::string(response_data_buf.m_data, response_data_buf.m_data_len);
                        }
                    }
                    else{
                        str_resp = std::string(response_data_buf.m_data, response_data_buf.m_data_len);
                    }

                    JsonParser json_parser_resp;
                    json_parser_resp.load_json(str_resp);
                }
                catch (const std::exception& e)
                {
                    PTraceError("Error: pilotos machine curl_post fail: " << (std::string)e.what() << ", resp:" << str_resp);
                    return false;
                }
            }
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl_handle);
            return ret;
        }
#endif
        void build_meta_json(rabbit::object &obj, const CalcConfig &config, QMType type, bool is_chip = false)
        {
            rabbit::array arr_ir;
            for (auto &ele : config.ir_vec) {
                arr_ir.push_back(ele);
            }
            if (config.ir.size() != 0) {
                obj.insert("QProg", config.ir);
            }
            if (config.ir_vec.size() != 0) {
                obj.insert("QProg", arr_ir);
            }
            obj.insert("QMachineType", std::to_string(static_cast<int>(type)));
            if (!m_token.empty()) {
                obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }
            if(is_chip) {
                obj.insert("ChipID", config.backend_id);
            }
            else {
                obj.insert("QProgLength", config.ir.size());
                obj.insert("ClusterID", config.backend_id);
            }
        }

        void build_meta_json(rabbit::array &arr_ir, rabbit::object &obj, const CalcConfig &config, QMType qm_type, bool is_chip = false)
        {
            obj.insert("PilotOSMachineFlag", true);
            for (auto &ele : config.ir_vec) {
                arr_ir.push_back(ele);
            }

            if (config.ir.size() != 0) {
                arr_ir.push_back(config.ir);
            }

            if (config.task_describe.size() != 0) {
                obj.insert("taskDescribe", config.task_describe);
            }

            if (arr_ir.size() != 0) {
                obj.insert("QProg", arr_ir);
            }
            obj.insert("QMachineType", std::to_string(static_cast<int>(qm_type)));
            obj.insert("TaskType", static_cast<int>(config.task_type));
            if (!m_token.empty()) {
                obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }
            if (is_chip) {
                obj.insert("ChipID", config.backend_id);
            }
            else {
                obj.insert("QProgLength", config.ir.size());
                obj.insert("ClusterID", config.backend_id);
            }
        }

        void build_subconfig_json(rabbit::object &obj, const CalcConfig &config, QMType type, bool is_chip = false)
        {
            /* TODO: insert configuration! */
        }

        std::string build_chip_expectation_task_json_msg(const CalcConfig &config, const std::vector<uint32_t> &qubits)
        {
            rabbit::object _obj;
            rabbit::array arr_ir;
            build_meta_json(arr_ir, _obj, config, QMType::REAL_CHIP, true);
            rabbit::array arr_qubits;
            for(auto &ele : qubits)
            {
                arr_qubits.push_back(ele);
            }

            rabbit::array arr_specified_block;
            for(auto &ele : config.specified_block)
            {
                arr_specified_block.push_back(ele);
            }

            rabbit::object _cfg_obj;
			_cfg_obj.insert("shot", config.shot);
			_cfg_obj.insert("amendFlag", config.is_amend);
            _cfg_obj.insert("mappingFlag", config.is_mapping);
            _cfg_obj.insert("circuitOptimization", config.is_optimization);
            _cfg_obj.insert("hamiltonian", config.hamiltonian);
            _cfg_obj.insert("qubits", arr_qubits);
            _cfg_obj.insert("specified_block", arr_specified_block);
			_obj.insert("Configuration", _cfg_obj);
			_obj.insert("callbackAddr", "TCP_CB");
            _obj.insert("token", m_token);

            return _obj.str();
        }

        std::string build_chip_qst_task_json_msg(const CalcConfig& config)
        {
            rabbit::object _obj;
            rabbit::array arr_ir;
            build_meta_json(arr_ir, _obj, config, QMType::REAL_CHIP, true);
            rabbit::array arr_specified_block;
            for(auto &ele : config.specified_block)
            {
                arr_specified_block.push_back(ele);
            }

            rabbit::object _cfg_obj;
			_cfg_obj.insert("shot", config.shot);
			_cfg_obj.insert("amendFlag", config.is_amend);
            _cfg_obj.insert("mappingFlag", config.is_mapping);
            _cfg_obj.insert("circuitOptimization", config.is_optimization);
            _cfg_obj.insert("specified_block", arr_specified_block);
			_obj.insert("Configuration", _cfg_obj);
			_obj.insert("callbackAddr", "TCP_CB");
            _obj.insert("token", m_token);

            return _obj.str();
        }

        std::string build_chip_expectation_task_json_msg_vec(const CalcConfig& config, const std::vector<uint32_t>& qubits)
        {
            rabbit::object _obj;
            rabbit::array arr_ir;
            build_meta_json(arr_ir, _obj, config, QMType::REAL_CHIP, true);

            rabbit::array arr_qubits;
            for (auto& ele : qubits) {
                arr_qubits.push_back(ele);
            }

            rabbit::array arr_specified_block;
            for (auto& ele : config.specified_block) {
                arr_specified_block.push_back(ele);
            }

            rabbit::object _cfg_obj;
            _cfg_obj.insert("shot", config.shot);
            _cfg_obj.insert("amendFlag", config.is_amend);
            _cfg_obj.insert("mappingFlag", config.is_mapping);
            _cfg_obj.insert("circuitOptimization", config.is_optimization);
            _cfg_obj.insert("hamiltonian", config.hamiltonian);
            _cfg_obj.insert("qubits", arr_qubits);
            _cfg_obj.insert("specified_block", arr_specified_block);
            _obj.insert("Configuration", _cfg_obj);
            _obj.insert("callbackAddr", "TCP_CB");
            _obj.insert("token", m_token);

            return _obj.str();
        }

        std::string build_chip_measure_task_json_msg_vec(const CalcConfig& config)
        {
            rabbit::object _obj;
            rabbit::array arr_ir;
            build_meta_json(arr_ir, _obj, config, QMType::REAL_CHIP, true);

            rabbit::array arr_specified_block;
            for(auto &ele : config.specified_block){
                arr_specified_block.push_back(ele);
            }

            rabbit::object _cfg_obj;
			_cfg_obj.insert("shot", config.shot);
            _cfg_obj.insert("amendFlag", config.is_amend);
            _cfg_obj.insert("PointLabel", config.point_lable);
            _cfg_obj.insert("Priority", config.priority);       /* task priority */
            _cfg_obj.insert("mappingFlag", config.is_mapping);
            _cfg_obj.insert("circuitOptimization", config.is_optimization);
            _cfg_obj.insert("IsProbCount", config.is_prob_counts);
            _cfg_obj.insert("specified_block", arr_specified_block);

            if (config.pulse_period != 0)
            {
                _cfg_obj.insert("PulsePeriod", config.pulse_period);
            }
			_obj.insert("Configuration", _cfg_obj);
            _obj.insert("callbackAddr", "TCP_CB");
            _obj.insert("token", m_token);

            return _obj.str();
        }

        std::string build_chip_measure_task_json_msg(const CalcConfig& config)
        {
            rabbit::object _obj;
            rabbit::array arr_ir;
            build_meta_json(arr_ir, _obj, config, QMType::REAL_CHIP, true);

            rabbit::array arr_specified_block;
            for (auto& ele : config.specified_block) {
                arr_specified_block.push_back(ele);
            }

            rabbit::object _cfg_obj;
            _cfg_obj.insert("shot", config.shot);
            _cfg_obj.insert("amendFlag", config.is_amend);
            _cfg_obj.insert("PointLabel", config.point_lable);
            _cfg_obj.insert("mappingFlag", config.is_mapping);
            _cfg_obj.insert("circuitOptimization", config.is_optimization);
            _cfg_obj.insert("isPostProcess", config.is_post_process);
            _cfg_obj.insert("IsProbCount", config.is_prob_counts);
            _cfg_obj.insert("specified_block", arr_specified_block);
            _obj.insert("Configuration", _cfg_obj);
            _obj.insert("callbackAddr", "TCP_CB");
            _obj.insert("token", m_token);

            return _obj.str();
        }

        std::string build_chip_noise_learning_task_json_msg(const std::string& parameter_json)
        {
            rabbit::object _obj;
            JsonParser jp;
            jp.load_json(parameter_json);
            rabbit::array ir_vec;
            if (!jp.has_member_string("ir"))
            {
                PTraceError("Parameter error!");
                throw std::runtime_error("Parameter error!");
            }
            ir_vec.push_back(jp.get_string("ir"));
            _obj.insert("ir", ir_vec);
            _obj.insert("callbackAddr", "TCP_CB");
            _obj.insert("ChipID", "72");
            _obj.insert("QMachineType", "5");

            rabbit::object _cfg_obj;
            if (jp.has_member_int32("pattern")) {
                _cfg_obj.insert("pattern", jp.get_int32("pattern"));
            }
            if (jp.has_member_string("file")) {
                _cfg_obj.insert("file", jp.get_string("file"));
            }
            if (jp.has_member_string("noise_model_file")) {
                _cfg_obj.insert("noise_model_file", jp.get_string("noise_model_file"));
            }
            if (jp.has_member_uint32("samples")) {
                _cfg_obj.insert("samples", jp.get_uint32("samples"));
            }
            if (jp.has_member_uint32("nl_shots")) {
                _cfg_obj.insert("nl_shots", jp.get_uint32("nl_shots"));
            }
            rabbit::array depths;
            if (jp.has_member_array("depths"))
            {
                std::vector<std::uint32_t> depth_v;
                jp.get_array("depths", depth_v);

                for (auto& ele : depth_v) {
                    depths.push_back(ele);
                }
                _cfg_obj.insert("depths", depths);
            }
            if (jp.has_member_int32("qem_base_method")) {
                _cfg_obj.insert("qem_base_method", jp.get_int32("qem_base_method"));
            }
            _obj.insert("Configuration", _cfg_obj);
            return _obj.str();
        }

        std::string build_chip_em_compute_task_json_msg(const std::string& parameter_json)
        {            
            rabbit::object _obj;
            rabbit::object _cfg_obj;
            rabbit::array expectation;
            JsonParser jp;
            rabbit::array ir_vec;
            std::vector<std::string>expectation_vec;
            jp.load_json(parameter_json);
            if (!jp.has_member_string("ir") || !jp.has_member_array("expectations")
                || !jp.has_member_int32("qem_method") /*|| !jp.has_member_string("noise_model_file")*/)
            {
                PTraceError("Parameter error!");
                throw std::runtime_error("Parameter error!");
            }              
            ir_vec.push_back(jp.get_string("ir"));
            _obj.insert("ir", ir_vec);            
            _obj.insert("callbackAddr", "TCP_CB");
            _obj.insert("ChipID", 72);
            _obj.insert("QMachineType", 5);
            _cfg_obj.insert("qem_method", jp.get_int32("qem_method"));

            jp.get_array("expectations", expectation_vec);
            for (auto& ele : expectation_vec) {
                expectation.push_back(ele);
            }
            _cfg_obj.insert("expectations", expectation);
            if (jp.has_member_int32("pattern")) {
                _cfg_obj.insert("pattern", jp.get_int32("pattern"));
            }
            if (jp.has_member_string("file")) {
                _cfg_obj.insert("file", jp.get_string("file"));
            }
            if (jp.has_member_uint32("samples")) {
                _cfg_obj.insert("samples", jp.get_uint32("samples"));
            }
            if (jp.has_member_uint32("nl_shots")) {
                _cfg_obj.insert("nl_shots", jp.get_uint32("nl_shots"));
            }
            rabbit::array depths;
            if (jp.has_member_array("depths")) 
            {
                std::vector<std::uint32_t> depth_v;
                jp.get_array("depths", depth_v);                

                for (auto& ele : depth_v) {
                    depths.push_back(ele);
                }
                _cfg_obj.insert("depths", depths);
            }
            if (jp.has_member_int32("qem_base_method")) {
                _cfg_obj.insert("qem_base_method", jp.get_int32("qem_base_method"));
            }
            if (jp.has_member_string("noise_model_file")) {
                _cfg_obj.insert("noise_model_file", jp.get_string("noise_model_file"));
            }
            if (jp.has_member_int32("qem_samples")) {
                int val = jp.get_int32("qem_samples");
                if (val > 0) {
                    _cfg_obj.insert("qem_samples", val);
                }
            }
            if (jp.has_member_uint32("qem_shots")) {
                _cfg_obj.insert("qem_shots", jp.get_uint32("qem_shots"));
            }
            rabbit::array noise_strength;
            if (jp.has_member_array("noise_strength")) 
            {               
                std::vector<double>noise_vec;
                jp.get_array("noise_strength", noise_vec);
                for (auto& ele : noise_vec) {
                    noise_strength.push_back(ele);
                }
                _cfg_obj.insert("noise_strength", noise_strength);
            }
            _obj.insert("Configuration", _cfg_obj);
            return _obj.str();
        }

        std::string build_full_amplitude_measure_task_json_msg(const std::string& prog_str,
            const uint32_t& cluster_id, const uint32_t& shots)
        {
            rabbit::object _obj;
            _obj.insert("QProg", prog_str);
            _obj.insert("QMachineType", to_string((int)QMType::FULL_AMPLITUDE));
            _obj.insert("QProgLength", prog_str.size());
            _obj.insert("ClusterID", cluster_id);
            if (!m_token.empty()) {
                _obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }

            rabbit::object _cfg_obj;
            _cfg_obj.insert("shot", shots);
            _cfg_obj.insert("measure_type", (int)MeasureType::MONTE_CARLO_MEASURE);
            _obj.insert("Configuration", _cfg_obj);
			_obj.insert("callbackAddr", "TCP_CB");
			return _obj.str();
        }

        std::string build_full_amplitude_pmeasure_task_json_msg(const std::string& prog_str,
            const std::vector<uint32_t>& qubit_vec,
            const uint32_t& cluster_id)
        {
            rabbit::object _obj;
            _obj.insert("QProg", prog_str);
            _obj.insert("QMachineType", to_string((int)QMType::FULL_AMPLITUDE));
            _obj.insert("QProgLength", prog_str.size());
            _obj.insert("ClusterID", cluster_id);
            if (!m_token.empty()) {
                _obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }

            rabbit::object _cfg_obj;
            rabbit::array _qbs_arr;
            for (auto& _q : qubit_vec) {
                _qbs_arr.push_back(_q);
            }
            _cfg_obj.insert("qubits", _qbs_arr);
            _cfg_obj.insert("measure_type", (int)MeasureType::PMEASURE);
            _obj.insert("Configuration", _cfg_obj);
			_obj.insert("callbackAddr", "TCP_CB");
			return _obj.str();
        }

        std::string build_noise_measure_task_json_msg(const std::string& prog_str,
            const PilotNoiseParams& noise_params,
            const uint32_t& cluster_id,
            const uint32_t& shots)
        {
            rabbit::object _obj;
            _obj.insert("QProg", prog_str);
            _obj.insert("QMachineType", to_string((int)QMType::NOISE));
            _obj.insert("QProgLength", prog_str.size());
            _obj.insert("ClusterID", cluster_id);
            if (!m_token.empty()) {
                _obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }

            rabbit::object _cfg_obj;
            _cfg_obj.insert("mode", 1);
            _cfg_obj.insert("shot", shots);
            _cfg_obj.insert("measure_type", (int)MeasureType::MONTE_CARLO_MEASURE);
            rabbit::object _model_obj;

            rabbit::array single_array;
            rabbit::array double_array;
            auto str_model = noise_model_mapping.at(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR);
            single_array.push_back(noise_params.noise_model);
            single_array.push_back(noise_params.single_gate_param);
            double_array.push_back(noise_params.noise_model);
            double_array.push_back(noise_params.double_gate_param);
            if (str_model == noise_params.noise_model)
            {
                single_array.push_back(noise_params.single_p2);
                single_array.push_back(noise_params.single_pgate);

                double_array.push_back(noise_params.double_p2);
                double_array.push_back(noise_params.double_pgate);
            }
            _model_obj.insert("single_gate", single_array);
            _model_obj.insert("double_gate", double_array);

            _cfg_obj.insert("noisemodel", _model_obj);
            _obj.insert("Configuration", _cfg_obj);
			_obj.insert("callbackAddr", "TCP_CB");

            return _obj.str();
        }

        std::string build_partial_amplitude_task_json_msg(const std::string& prog_str,
            const std::vector<std::string>& target_amplitude_vec,
            const uint32_t& cluster_id)
        {
            rabbit::object _obj;
            _obj.insert("QProg", prog_str);
            _obj.insert("QMachineType", to_string((int)QMType::PARTIAL_AMPLITUDE));
            _obj.insert("QProgLength", prog_str.size());
            _obj.insert("ClusterID", cluster_id);
            _obj.insert("token", m_token);

            rabbit::object _cfg_obj;
            _cfg_obj.insert("measure_type", (int)MeasureType::PMEASURE);

            rabbit::array _amp_arr;
            for (auto& _amp : target_amplitude_vec) {
                _amp_arr.push_back(_amp);
            }
            _cfg_obj.insert("Amplitude", _amp_arr);

			_obj.insert("Configuration", _cfg_obj);
			_obj.insert("callbackAddr", "TCP_CB");

            return _obj.str();
        }

        std::string build_single_amplitude_task_json_msg(const std::string& prog_str,
            const std::string& target_amplitude,
            const uint32_t& cluster_id)
        {
            rabbit::object _obj;
            _obj.insert("QProg", prog_str);
            _obj.insert("QMachineType", to_string((int)QMType::SINGLE_AMPLITUDE));
            _obj.insert("QProgLength", prog_str.size());
            _obj.insert("ClusterID", cluster_id);
            if (!m_token.empty()) {
                _obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }

            rabbit::object _cfg_obj;
            _cfg_obj.insert("measure_type", (int)MeasureType::PMEASURE);
            _cfg_obj.insert("Amplitude", target_amplitude);
			_obj.insert("Configuration", _cfg_obj);
			_obj.insert("callbackAddr", "TCP_CB");

            return _obj.str();
        }

        std::string build_full_amplitude_expectation_task_json_msg(const std::string& prog_str,
            const QuantumHamiltonianData& hamiltonian,
            const std::vector<uint32_t>& qubit_vec,
            const uint32_t& cluster_id)
        {
            rabbit::object _obj;
            _obj.insert("QProg", prog_str);
            _obj.insert("QMachineType", to_string((int)QMType::FULL_AMPLITUDE));
            _obj.insert("QProgLength", prog_str.size());
            _obj.insert("ClusterID", cluster_id);
            if (!m_token.empty()) {
                _obj.insert("token", m_token);
            }
            else {
                PTraceInfo("token is empty");
            }

            PTraceInfo("on build_full_amplitude_expectation_task_json_msg prog_str: " << prog_str);
            rabbit::object _cfg_obj;
            rabbit::array qubit_arr;
            for (auto _q : qubit_vec) {
                qubit_arr.push_back(_q);
            }
            _cfg_obj.insert("qubits", qubit_arr);

            PTraceInfo("on _hamiltonian_to_obj 1111111, hamiltonian.size()= " << hamiltonian.size());
            rabbit::array _ham_param_arr;
            rabbit::array _pauli_param_arr;
            rabbit::array _pauli_type_arr;
            for (auto i = 0; i < hamiltonian.size(); ++i)
            {
                const auto& item = hamiltonian[i];
                rabbit::array _temp_param_arr;
                rabbit::array _temp_type_arr;

                for (auto val : item.first)
                {
                    PTraceError("on item:<" << val.first << ", " << val.second << ">, " << std::string(1, (char)(val.second)));
                    _temp_param_arr.push_back(val.first);
                    _temp_type_arr.push_back(std::string(1, (char)(val.second)));
                }

                _pauli_param_arr.push_back(_temp_param_arr);
                _pauli_type_arr.push_back(_temp_type_arr);
                _ham_param_arr.push_back(item.second);
            }

            PTraceError("on _hamiltonian_to_obj");
            rabbit::object _ham_val_obj;
            _ham_val_obj.insert("pauli_type", _pauli_type_arr);
            _ham_val_obj.insert("pauli_parm", _pauli_param_arr);

            rabbit::object _ham_obj;
            _ham_obj.insert("hamiltonian_value", _ham_val_obj);
            _ham_obj.insert("hamiltonian_param", _ham_param_arr);
            PTraceError("on _hamiltonian_to_obj _ham_obj:" << _ham_obj.str());

            _cfg_obj.insert("hamiltonian", _ham_obj);
            PTraceInfo("on build_full_amplitude_expectation_task_json_msg, obj:" << _obj.str());

            _cfg_obj.insert("measure_type", (uint32_t)(MeasureType::EXPECTATION));
            PTraceInfo("on build_full_amplitude_expectation_task_json_msg, _cfg_obj:" << _cfg_obj.str());
			_obj.insert("Configuration", _cfg_obj);
            PTraceInfo("on build_full_amplitude_expectation_task_json_msg: " << _obj.str());
			_obj.insert("callbackAddr", "TCP_CB");
            PTraceInfo("on build_full_amplitude_expectation_task_json_msg");
            PTraceInfo("build_full_amplitude_expectation_task_json_msg: " << std::string(_obj.str()));

            return _obj.str();
        }

    private:
        std::string m_pilot_url;
        std::string m_token;
        std::thread m_tcp_thread;
		std::atomic<bool> m_log_cout{ false };
		std::string m_server_host;
		unsigned short m_server_port{ 0 };

        std::mutex m_func_info_mutex;
        std::unordered_map<std::string, std::function<void(std::shared_ptr<JsonParser>)>> m_cb_func_map;

        std::mutex m_mutex;           /**< Locks are used to block threads */
        std::condition_variable m_cv; /**< Conditional variables are used to block threads */
        std::unordered_map<std::string, std::string> m_task_info_map;
        std::atomic<bool> m_b_init_ok{ false };
    };
}

/*******************************************************************
*                      class QPilotMachine
********************************************************************/
QPilotMachine::QPilotMachine(){}

QPilotMachine::~QPilotMachine(){}

bool QPilotMachine::init(const std::string& pilot_url,
    bool log_cout /*= false*/)
{
    m_pilot_url = pilot_url;
    m_log_cout = log_cout;
	_abort();
	m_imp_obj = std::make_unique<QPilotMachineImp>();
    return m_imp_obj->init(pilot_url, log_cout);
}

void QPilotMachine::_abort(void)
{
	std::signal(SIGFPE, QPilotMachine::abort);
	std::signal(SIGILL, QPilotMachine::abort);
	std::signal(SIGINT, QPilotMachine::abort);
	std::signal(SIGABRT, QPilotMachine::abort);
	std::signal(SIGSEGV, QPilotMachine::abort);
	std::signal(SIGTERM, QPilotMachine::abort);
}

void QPilotMachine::abort(int _signal)
{
	std::exit(0);
}

bool QPilotMachine::init_withconfig(const std::string& config_path)
{
    try
    {
        JsonParser  json_conf;
        std::string s_pilot_conf;
        std::string s_pilot_conf_buffer;
        std::fstream f_pilot_conf(config_path);
        if (f_pilot_conf.is_open())
        {
            while (!f_pilot_conf.eof())
            {
                f_pilot_conf >> s_pilot_conf_buffer;
                s_pilot_conf += s_pilot_conf_buffer;
            }
            f_pilot_conf.close();
            if (!s_pilot_conf.empty())
            {
                json_conf.load_json(s_pilot_conf);
                bool log_cout = json_conf.get_bool("log_cout");
                std::string pilot_url = json_conf.get_string("pilot_url");
                return init(pilot_url, log_cout);
            }
        }
        PTraceError("Error: pilotmachine configure error:\n{\"log_out\":false,\"pilot_url\":\"https://ip:port\"}");
    }
    catch (const std::exception& e)
    {
        PTraceError("Error: on exception: " << e.what());
        throw;
    }
}

ErrorCode QPilotMachine::get_token(std::string& rep_json)
{
    JsonMsg::JsonParser jp;
    jp.load_json(rep_json);
    if (jp.has_member("errCode"))
    {
        auto errCode = jp.get_int32("errCode");
        if (errCode == 0) {
            auto& token_ref = m_imp_obj->get_token_str();
            if (!jp.get_string("token").empty())
            {
                token_ref = jp.get_string("token");
            }
            PTraceInfo("After login your token is: " + token_ref);
            return ErrorCode::NO_ERROR_FOUND;
        }
        else
        {
            PTraceError("Login failed! response: " << rep_json);
            return (ErrorCode)errCode;
        }
    }
    else
    {
        return ErrorCode::UNDEFINED_ERROR;
    }

}

std::string QPilotMachine::build_measure_task_msg(const CalcConfig& config)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->build_chip_measure_task_json_msg_vec(config);
    }
    return std::string("");
}

std::string QPilotMachine::build_expectation_task_msg(const CalcConfig& config, const std::vector<uint32_t>& qubits)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->build_chip_expectation_task_json_msg(config, qubits);
    }
    return std::string("");
}

std::string QPilotMachine::build_qst_task_msg(const CalcConfig& config)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->build_chip_qst_task_json_msg(config);
    }
    return std::string("");
}

std::string QPilotMachine::build_query_msg(const std::string& task_id)
{
    auto token = m_imp_obj->get_token_str();
    JsonMsg::JsonBuilder jb;
    jb.add_member("token", token);
    jb.add_member("taskId", task_id);
    return jb.get_json_str();
}

PilotQVM::ErrorCode QPilotMachine::parser_probability_result(JsonMsg::JsonParser& json_parser, std::vector<std::map<std::string, double>>& result)
{
    return m_imp_obj->parser_probability_result_vec(json_parser, result);
}

PilotQVM::ErrorCode QPilotMachine::parser_expectation_result(JsonMsg::JsonParser& json_parser, std::vector<double>& result)
{
    return m_imp_obj->_parser_expectation_result_vec(json_parser, result);
}

#ifdef USE_CURL
ErrorCode QPilotMachine::execute_expectation_task(const CalcConfig &config, const std::vector<uint32_t> &qubits, std::vector<double>& result)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_expectation_task(config, qubits, result);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

//ErrorCode QPilotMachine::execute_expectation_task_vec(const CalcConfig& config, const std::vector<uint32_t>& qubits, std::vector<double>& result)
//{
//    if (m_imp_obj->init(m_pilot_url, m_log_cout))
//    {
//        return m_imp_obj->execute_expectation_task_vec(config, qubits, result);
//    }
//    return ErrorCode::ERR_TCP_INIT_FATLT;
//}

std::string QPilotMachine::async_execute_expectation_task(const CalcConfig &config, const std::vector<uint32_t> &qubits, std::vector<double>& result)
{
    std::string task_id = "default";
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->async_execute_expectation_task(config, qubits, result);
    }
    return task_id;
}

//std::vector<std::string> QPilotMachine::async_execute_expectation_task_vec(const CalcConfig& config, const std::vector<uint32_t>& qubits, std::vector<double>& result)
//{
//    //std::string task_id = "default";
//    /*if (m_imp_obj->init(m_pilot_url, m_log_cout))
//    {
//        return m_imp_obj->async_execute_expectation_task_vec(config, qubits, result);
//    }*/
//    return std::vector<std::string>();
//}

//ErrorCode QPilotMachine::execute_measure_task(const CalcConfig &config)
//{
//    if (m_imp_obj->init(m_pilot_url, m_log_cout))
//    {
//        return m_imp_obj->execute_measure_task(config);
//    }
//    return ErrorCode::ERR_TCP_INIT_FATLT;
//}

//std::string QPilotMachine::async_execute_measure_task(const CalcConfig &config)
//{
//    std::string task_id = "default";
//    if (m_imp_obj->init(m_pilot_url, m_log_cout))
//    {
//        return m_imp_obj->async_execute_measure_task(config);
//    }
//    return task_id;
//}

ErrorCode QPilotMachine::execute_measure_task_vec(const CalcConfig &config, std::vector<std::map<std::string, double>> &result)
{
    /* 初始化m_imp_obj */
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_measure_task_vec(config, result);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_measure_task_vec(const CalcConfig &config, std::vector<std::map<std::string, size_t>> &result)
{
    /* 初始化m_imp_obj */
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_measure_task_vec(config, result);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_measure_task_vec(const CalcConfig &config, std::string &result)
{
    /* 初始化m_imp_obj */
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_measure_task_vec(config, result);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_measure_task(const CalcConfig& config, std::map<std::string, double>& result)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_measure_task(config, result);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::async_execute_measure_task_vec(const CalcConfig &config, std::string& task_id)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->async_execute_measure_task_vec(config, task_id);
    }
    task_id = "task not execute";
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

std::string QPilotMachine::async_execute_measure_task(const CalcConfig& config)
{
    std::string task_id = "task not execute";
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->async_execute_measure_task(config);
    }
    return task_id;
}

/*
 * @brief execute_noise_learn_task
 */

ErrorCode QPilotMachine::execute_noise_learning_task(const std::string& parameter_json, std::string& task_id)
{
    task_id = "task not execute";
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_noise_learning_task(parameter_json, task_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::async_execute_em_compute_task(const std::string& parameter_json, std::string& task_id)
{
    task_id = "task not execute";
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->async_execute_em_compute_task(parameter_json, task_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_em_compute_task(const std::string& parameter_json, std::string& task_id, std::vector<double>& result)
{
    task_id = "task not execute";
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_em_compute_task(parameter_json, task_id, result);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

bool QPilotMachine::execute_query_task_state(const std::string& task_id, PilotTaskQueryResult& result)
{
    return m_imp_obj->execute_query_task_state(task_id, result);
}

bool QPilotMachine::execute_query_compile_prog(const std::string task_id, std::string& compile_prog, bool& without_compensate)
{
    return m_imp_obj->execute_query_compile_prog(task_id, compile_prog, without_compensate);
}


ErrorCode QPilotMachine::execute_login_pilot_api(const std::string&api_key)
{
    return m_imp_obj->execute_login_pilot_api(api_key);
}

ErrorCode QPilotMachine::execute_login_pilot(const std::string&username, const std::string&pwd)
{
    return m_imp_obj->execute_login_pilot(username, pwd);
}

ErrorCode QPilotMachine::execute_measure_task(const CalcConfig &config, std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_measure_task(config, cb_func);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_full_amplitude_measure_task(const std::string& prog_str,
    std::map<std::string, double>& result,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/,
    const uint32_t& shots /*= 1000*/)
{

    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {

        return m_imp_obj->execute_full_amplitude_measure_task(prog_str, result, cluster_id, shots);
    }

    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_full_amplitude_measure_task(const std::string& prog_str,
    std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/,
    const uint32_t& shots /*= 1000*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_full_amplitude_measure_task(prog_str, cb_func, cluster_id, shots);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_full_amplitude_pmeasure_task(const std::string& prog_str,
    const std::vector<uint32_t>& qubit_vec,
    std::map<std::string, double>& result,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_full_amplitude_pmeasure_task(prog_str, qubit_vec, result, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_full_amplitude_pmeasure_task(const std::string& prog_str,
    const std::vector<uint32_t>& qubit_vec,
    std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_full_amplitude_pmeasure_task(prog_str, qubit_vec, cb_func, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

bool QPilotMachine::build_noise_params(const uint32_t& nose_model_type,
    const std::vector<double>& single_params,
    const std::vector<double>& double_params,
    PilotNoiseParams& noise_params)
{
    auto _found_iter = noise_model_mapping.find((NOISE_MODEL)nose_model_type);
    if (noise_model_mapping.end() == _found_iter
        || single_params.empty()
        || double_params.empty())
    {
        return false;
    }

    if (((NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR == (NOISE_MODEL)nose_model_type)
        && ((single_params.size() != 3) || (double_params.size() != 3)))
        ||
        ((NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != (NOISE_MODEL)nose_model_type)
            && (single_params.size() != 1 || double_params.size() != 1)))
    {
        return false;
    }

    noise_params.noise_model = _found_iter->second;
    noise_params.single_gate_param = single_params[0];
    noise_params.double_gate_param = double_params[0];

    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR == (NOISE_MODEL)nose_model_type)
    {
        noise_params.single_p2 = single_params[1];
        noise_params.double_p2 = double_params[1];
        noise_params.single_pgate = single_params[2];
        noise_params.double_pgate = double_params[2];
    }

    return true;
}

ErrorCode QPilotMachine::execute_noise_measure_task(const std::string& prog_str,
    const PilotNoiseParams& noise_params,
    std::map<std::string, double>& result,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/,
    const uint32_t& shots /*= 1000*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_noise_measure_task(prog_str, noise_params, result, cluster_id, shots);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_noise_measure_task(const std::string& prog_str,
    const PilotNoiseParams& noise_params,
    std::function<void(ErrorCode, std::map<std::string, double>&)> cb_func,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/,
    const uint32_t& shots /*= 1000*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_noise_measure_task(prog_str, noise_params, cb_func, cluster_id, shots);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_partial_amplitude_task(const std::string& prog_str,
    const std::vector<std::string>& target_amplitude_vec,
    std::map<std::string, Complex_>& result,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_partial_amplitude_task(prog_str, target_amplitude_vec, result, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_partial_amplitude_task(const std::string& prog_str,
    const std::vector<std::string>& target_amplitude_vec,
    std::function<void(ErrorCode, const std::map<std::string, Complex_>&)> cb_func,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_partial_amplitude_task(prog_str, target_amplitude_vec, cb_func, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_single_amplitude_task(const std::string& prog_str,
    const std::string& target_amplitude,
    Complex_& result,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_single_amplitude_task(prog_str, target_amplitude, result, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_single_amplitude_task(const std::string& prog_str,
    const std::string& target_amplitude,
    std::function<void(ErrorCode, const Complex_&)> cb_func,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_single_amplitude_task(prog_str, target_amplitude, cb_func, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_full_amplitude_expectation(const std::string& prog_str,
    const QuantumHamiltonianData& hamiltonian,
    const std::vector<uint32_t>& qubit_vec,
    double& result,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_full_amplitude_expectation(prog_str, hamiltonian, qubit_vec, result, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

ErrorCode QPilotMachine::execute_full_amplitude_expectation(const std::string& prog_str,
    const QuantumHamiltonianData& hamiltonian,
    const std::vector<uint32_t>& qubit_vec,
    std::function<void(ErrorCode, double)> cb_func,
    const uint32_t& cluster_id /*= ANY_CLUSTER_BACKEND*/)
{
    if (m_imp_obj->init(m_pilot_url, m_log_cout))
    {
        return m_imp_obj->execute_full_amplitude_expectation(prog_str, hamiltonian, qubit_vec, cb_func, cluster_id);
    }
    return ErrorCode::ERR_TCP_INIT_FATLT;
}

#endif
bool QPilotMachine::parse_task_result(const std::string& result_str, 
    std::map<std::string, double>& result_mp)
{
    JsonParser json_parser_res;
    if (!json_parser_res.load_json(result_str))
    {
        PTraceError("Error: Failed to parse str: " << result_str);
        return false;
    }

    auto& res_str = json_parser_res.get_json_obj();
    if (res_str.HasMember("key") && res_str.HasMember("value"))
    {
        auto key_vec = res_str["key"].GetArray();
        auto value_vec = res_str["value"].GetArray();

        auto sizek = key_vec.Size();
        auto sizev = value_vec.Size();
        if (sizek == sizev)
        {
            for (auto i = 0; i < sizek; ++i)
            {
                auto key = key_vec[i].GetString();
                auto val = value_vec[i].IsDouble() ? value_vec[i].GetDouble() : (double)value_vec[i].GetInt();

                result_mp.insert(make_pair(key, val));
            }
        }
    }

    return true;
}

bool QPilotMachine::parse_task_result(const std::vector<std::string>& result_vec,
    std::vector<std::map<std::string, double>>& result_mp_vec)
{
    rabbit::document doc;
    for (size_t i = 0; i < result_vec.size(); i++) 
    {
        doc.parse(result_vec[i]);
        if (doc["key"].size() != doc["value"].size()) {
            PTraceError("key and value is mismatch!");
            return false;
        }

        std::map<std::string, double>temp;
        for (int j = 0; j < doc["key"].size(); ++j)
        {
            if (!doc["key"][j].is_string() || !doc["value"][j].is_double())
            {
                PTraceError("key or value is wrong type!");
                return false;
            }
            temp.insert(make_pair(doc["key"][j].as_string(), doc["value"][j].as_double()));
        }
        result_mp_vec.push_back(temp);
        temp.clear();
    }

    return true;
}

bool QPilotMachine::parse_task_result(const std::vector<std::string>& result_vec,
    std::vector<std::map<std::string, uint64_t>>& result_mp_vec)
{
    rabbit::document doc;
    for (size_t i = 0; i < result_vec.size(); i++) {
        doc.parse(result_vec[i]);
        if (doc["key"].size() != doc["value"].size()) {
            PTraceError("key and value is mismatch!");
            return false;
        }
        std::map<std::string, uint64_t>temp;
        for (int j = 0; j < doc["key"].size(); ++j) 
        {
            if (!doc["key"][j].is_string() || !doc["value"][j].is_int())
            {
                PTraceError("key or value is wrong type!");
                return false;
            }
            temp.insert(make_pair(doc["key"][j].as_string(), doc["value"][j].as_uint64()));
        }
        result_mp_vec.push_back(temp);
        temp.clear();
    }
    return true;
}

bool QPilotMachine::parse_qst_density(const std::string& result,
    std::vector<std::map<std::string, double>>& result_mp_vec)
{
    try
    {
        rapidjson::Document doc;
        doc.Parse(result.c_str());
        if (doc.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < doc.Size(); i++)
            {
                const auto& obj = doc[i];
                if (obj.IsObject())
                {
                    if (obj.HasMember("r") && obj.HasMember("i"))
                    {
                        std::map<std::string, double> density;
                        density.emplace("r", obj["r"].GetDouble());
                        density.emplace("i", obj["i"].GetDouble());
                        result_mp_vec.push_back(density);
                    }
                    else
                        throw "json has no member r or i";
                }
                else
                    throw "json is not legal!";
            }
            return true;
        }
        else
            throw "result is not Array";
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        return false;
    }
}

bool QPilotMachine::parse_qst_fidelity(const std::string& result_str,
    double& result)
{
    try
    {
        result = std::stod(result_str);
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        return false;
    }
}

