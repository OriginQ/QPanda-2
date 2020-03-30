#include <fstream>
#include <math.h>
#include <algorithm>
#include "ThirdParty/TinyXML/tinyxml.h"
#include "Core/Utilities/Tools/base64.hpp"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"

#ifdef USE_CURL
#define DEFAULT_CLOUD_TOKEN         "3CD107AEF1364924B9325305BF046FF3"
#define DEFAULT_CLUSTER_TOKEN         "9060A09CC66543F194819EFF2834DE54"

#define DEFAULT_CLOUD_COMPUTEAPI    "http://10.10.12.53:4630/api/QCode/QRunes2/submitTask.json"
#define DEFAULT_CLOUD_INQUREAPI     "http://10.10.12.53:4630/api/QCode/QRunes2/queryTask.json"

#define DEFAULT_CLUSTER_COMPUTEAPI    "http://10.10.12.176:8060/api/taskApi/submitTask.json"
#define DEFAULT_CLUSTER_INQUREAPI     "http://10.10.12.176:8060/api/taskApi/getResultDetail.json"

USING_QPANDA
using namespace std;
using namespace Base64;
using namespace rapidjson;

#ifdef USE_CURL

QCloudMachine::QCloudMachine()
{
    curl_global_init(CURL_GLOBAL_ALL);
}

QCloudMachine::~QCloudMachine()        
{
    curl_global_cleanup();
}

void QCloudMachine::init()
{
    XmlConfigParam config;
    if (!config.loadFile(CONFIG_PATH))
    {
        m_compute_url = DEFAULT_CLOUD_COMPUTEAPI;
        m_inqure_url = DEFAULT_CLOUD_INQUREAPI;
        m_token = DEFAULT_CLOUD_TOKEN;
    }
    else
    {
        map<string, string> QCloudConfig;
        bool is_success = config.getQuantumCloudConfig(QCloudConfig);
        if (!is_success)
        {
            QCERR("config error");
            m_compute_url = DEFAULT_CLOUD_COMPUTEAPI;
            m_inqure_url = DEFAULT_CLOUD_INQUREAPI;
            m_token = DEFAULT_CLOUD_TOKEN;
        }
        else
        {
            m_compute_url = QCloudConfig["ComputeAPI"];
            m_inqure_url = QCloudConfig["InqureAPI"];
            m_token = QCloudConfig["APIKEY"];
        }
    }

    _start();
}

size_t recvJsonData
(void *ptr, size_t size, size_t nmemb, void *stream)
{
    std::string data((const char*)ptr, 0, (size_t)(size * nmemb));

    *((std::stringstream*)stream) << data << std::endl;

    return size * nmemb;
}

string QCloudMachine::runWithConfiguration
(QProg &prog,Document &parm)
{
    auto qubit_num = getAllocateQubit();
    rapidjson::Document::AllocatorType &allocator = parm.GetAllocator();
    auto prog_bin = qProgToBinary(prog,this);

    rapidjson::Value json_elem(kStringType);
    json_elem.SetString(prog_bin.c_str(), (rapidjson::SizeType)prog_bin.size(), allocator);
    parm.AddMember("QProg", json_elem, allocator);
    parm.AddMember("TaskType", CLOUD_TASK_TYPE::CLOUD_MEASURE, allocator);
    parm.AddMember("typ", qubit_num <= 25 ? 0 : qubit_num > 31 ? 2 : 1, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    parm.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson("http://10.10.12.50/api/QCode/QRunes2/submitTask.json", post_json);

    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError() ||
        !recv_doc.HasMember("obj"))
    {
        return "error";
    }
    else
    {
        const rapidjson::Value &Obj = recv_doc["obj"];
        if (!Obj.IsObject() || 
            !Obj.HasMember("TaskId")|| 
            !Obj.HasMember("TaskState"))
        {
            return "error";
        }
        else
        {
            const rapidjson::Value &taskid = Obj["TaskId"];
            const rapidjson::Value &tasksa = Obj["TaskState"];
            string sRes;
            sRes.append("{\"TaskId\":\"")
                .append(taskid.GetString())
                .append("\",\"TaskState\":\"")
                .append(tasksa.GetString())
                .append("\"}");
            return sRes;
        }
    }
}

std::string QCloudMachine::postHttpJson
(const std::string &sUrl, std::string & sJson)
{
    std::stringstream out;

    auto pCurl = curl_easy_init();

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers,"Content-Type: application/json;charset=UTF-8");
    headers = curl_slist_append(headers, "Connection: keep-alive");
    headers = curl_slist_append(headers, "Server: nginx/1.16.1");
    headers = curl_slist_append(headers,"Transfer-Encoding: chunked"); 
    curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(pCurl, CURLOPT_TIMEOUT, 3);

    curl_easy_setopt(pCurl, CURLOPT_CONNECTTIMEOUT, 3);

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
        stringstream errMsg;
        errMsg << "post failed : " << curl_easy_strerror(res) << std::endl;
        return errMsg.str();
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(pCurl);

    return out.str().substr(out.str().find("{"));
}

string QCloudMachine::probRunDict
(QProg &prog, QVec qvec, rapidjson::Document &parm)
{
    auto qubit_num = getAllocateQubit();
    auto prog_bin = qProgToBinary(prog,this);
    rapidjson::Document::AllocatorType &allocator = parm.GetAllocator();

    rapidjson::Value prog_elem(kStringType);
    prog_elem.SetString(prog_bin.c_str(),(SizeType)prog_bin.size(), allocator);

    parm.AddMember("QProg", prog_elem, allocator);
    parm.AddMember("TaskType",CLOUD_TASK_TYPE::CLOUD_PMEASURE, allocator);

    vector<uint32_t> qvec_elem;
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit)
    {qvec_elem.emplace_back((unsigned int)qubit->getPhysicalQubitPtr()->getQubitAddr()); });
    uint256_t qubit_array;
    if (!qvec_elem.empty())
    {
        qubit_array = (uint256_t)1 << qvec_elem[0];
        for_each(qvec_elem.begin() + 1, qvec_elem.end(), 
            [&qubit_array](size_t qubit) {qubit_array = qubit_array | ((uint256_t)1 << qubit); });
    }
    else
    {
        return "error";
    }

    auto qubits = integerToString(qubit_array);
    prog_elem.SetString(qubits.c_str(), (SizeType)qubits.size(), allocator);
    parm.AddMember("Qubits", prog_elem, allocator);
    parm.AddMember("typ", qubit_num <= 25 ? 0 : qubit_num > 31 ? 2 : 1, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    parm.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(m_compute_url, post_json);

    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError() ||
        !recv_doc.HasMember("obj"))
    {
        return "error";
    }
    else
    {
        const rapidjson::Value &Obj = recv_doc["obj"];
        if (!Obj.IsObject() ||
            !Obj.HasMember("TaskId") ||
            !Obj.HasMember("TaskState"))
        {
            return "error";
        }
        else
        {
            const rapidjson::Value &taskid = Obj["TaskId"];
            const rapidjson::Value &tasksa = Obj["TaskState"];
            string sRes;
            sRes.append("{\"TaskId\":\"")
                .append(taskid.GetString())
                .append("\",\"TaskState\":\"")
                .append(tasksa.GetString())
                .append("\"}");
            return sRes;
        }
    }
}

std::map<std::string, double> QCloudMachine::getResult(std::string taskid)
{
    Document json_doc;
    Document::AllocatorType &allocator = json_doc.GetAllocator();
    Value root(kObjectType);
    Value json_elem(kStringType);

    json_elem.SetString(m_token.c_str(), (SizeType)m_token.size(), allocator);
    root.AddMember("token", json_elem, allocator);

    json_elem.SetString(taskid.c_str(), (SizeType)taskid.size(), allocator);
    root.AddMember("taskid", json_elem, allocator);
    root.AddMember("TaskType", 0, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    root.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(m_inqure_url, post_json);

    std::map<std::string, double> recv_res;
    parserRecvJson(recv_json, recv_res);

    return recv_res;
}

std::string QCloudMachine::parserRecvJson
(std::string recv_json, std::map<std::string, double>& recv_res)
{
    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError())
    {
        return "inqure result failed : parser recv json error";
    }
    else
    {
        if (!recv_doc.HasMember("obj") 
         || !recv_doc.HasMember("message")
         || !recv_doc.HasMember("success"))
        {
            return "inqure result failed : obj/message/success not found";
        } 
        else
        {
            Value &is_success = recv_doc["success"];
            if (!is_success.GetBool())
            {
                Value &message = recv_doc["message"];
                return message.GetString();
            }
        }

        Value &Obj = recv_doc["obj"];
        if (!Obj.IsObject() || !Obj.HasMember("TaskState"))
        {
            return "inqure result failed : TaskState not found";
        }
        else
        {
            Value &tasksta = Obj["TaskState"];
            switch (atoi(tasksta.GetString()))
            {
            case TASK_STATUS::WAITING: return "Waiting ...";
            case TASK_STATUS::COMPUTING: return "Computing ...";
            case TASK_STATUS::FAILED:
            {
                Value &err_msg = Obj["ErrorMsg"];
                return err_msg.GetString();
            }
            case TASK_STATUS::FINISHED:
            {
                Document result_doc;
                Value &res = Obj["TaskResult"];
                if (result_doc.Parse(res.GetString()).HasParseError() 
                || !result_doc.HasMember("key") 
                || !result_doc.HasMember("value"))
                {
                    return "inqure result failed : key/value not found";
                }
                else
                {
                    Value &key = result_doc["key"];
                    Value &value = result_doc["value"];

                    for (SizeType i = 0; i < key.Size(); ++i)
                    {
                        recv_res.insert(make_pair(key[i].GetString(), value[i].GetDouble()));
                    }
                    return "inqure result success";
                }
            }
            default:
                return "inqure result failed : task status error";
                break;
            }
        }
    }
}

string QCloudMachine::full_amplitude_measure
(QProg &prog, int shot)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", DEFAULT_CLUSTER_TOKEN);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubit());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", shot);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(DEFAULT_CLUSTER_COMPUTEAPI, post_json);

    return parser_cluster_result_json(recv_json);
}

string QCloudMachine::full_amplitude_pmeasure
(QProg &prog,const Qnum &qubit_vec)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    rapidjson::Value qubit_array(rapidjson::kArrayType);
    for_each(qubit_vec.begin(), qubit_vec.end(), [&](const size_t qubit)
    {
        if (qubit >= getAllocateQubitNum())
        {
            QCERR("qubit error");
            return;
        }
        else
        {
            qubit_array.PushBack((SizeType)qubit, allocator);
        }
    });

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", DEFAULT_CLUSTER_TOKEN);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    doc.AddMember("qubits", qubit_array, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(DEFAULT_CLUSTER_COMPUTEAPI, post_json);

    return parser_cluster_result_json(recv_json);
}

string QCloudMachine::partial_amplitude_pmeasure
(QProg &prog, std::vector<std::string> &amplitude_vec)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    auto qubit_num = getAllocateQubit();
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    rapidjson::Value amplitude_array(rapidjson::kArrayType);
    for_each(amplitude_vec.begin(), amplitude_vec.end(), [&](string &amplitude)
    {
        rapidjson::Value amplitude_value(rapidjson::kStringType);
        amplitude_value.SetString(amplitude.c_str(), amplitude.size());
        amplitude_array.PushBack(amplitude_value, allocator);
    });

    rapidjson::Value code_str(rapidjson::kStringType);
    code_str.SetString(prog_str.c_str(), prog_str.size());

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", DEFAULT_CLUSTER_TOKEN);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::PARTIAL_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    doc.AddMember("Amplitude", amplitude_array, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(DEFAULT_CLUSTER_COMPUTEAPI, post_json);

    return parser_cluster_result_json(recv_json);
}

string QCloudMachine::single_amplitude_pmeasure
(QProg &prog, std::string amplitude)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", DEFAULT_CLUSTER_TOKEN);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::SINGLE_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "Amplitude", amplitude);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(DEFAULT_CLUSTER_COMPUTEAPI, post_json);

    return parser_cluster_result_json(recv_json);
}

std::string QCloudMachine::parser_cluster_result_json(std::string &recv_json)
{
    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError()
        || !recv_doc.HasMember("obj") || !recv_doc.HasMember("success"))
    {
        return "error";
    }
    else
    {
        const rapidjson::Value &success = recv_doc["success"];
        if (!success.GetBool())
        {
            return "false";
        }
        else
        {
            const rapidjson::Value &Obj = recv_doc["obj"];
            if (!Obj.IsObject() ||
                !Obj.HasMember("taskId") ||
                !Obj.HasMember("id"))
            {
                return "error";
            }
            else
            {
                const rapidjson::Value &taskid = Obj["taskId"];
                return taskid.GetString();
            }
        }
    }
}

std::string QCloudMachine::get_result(CLOUD_QMACHINE_TYPE type, std::string taskid)
{
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "taskId", taskid);
    add_string_value(doc, "apiKey", DEFAULT_CLUSTER_TOKEN);
    add_string_value(doc, "QMachineType", (int)type);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(DEFAULT_CLUSTER_INQUREAPI, post_json);

    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError()
        || !recv_doc.HasMember("obj") || !recv_doc.HasMember("success"))
    {
        return "result json error";
    }
    else
    {
        try
        {
            const rapidjson::Value &success = recv_doc["success"];
            if (!success.GetBool())
            {
                return "get result failure";
            }
            else
            {
                const rapidjson::Value &Obj = recv_doc["obj"];
                const rapidjson::Value &Val = Obj["qcodeTaskNewVo"];
                const rapidjson::Value &List = Val["taskResultList"];
                const rapidjson::Value &result = List[0]["taskResult"];

                std::string state = List[0]["taskState"].GetString();
                if (state != "3")
                {
                    return "task state : " + state;
                }
                else
                {
                    Document result_doc;
                    if (result_doc.Parse(result.GetString()).HasParseError()
                        || !result_doc.HasMember("Key"))
                    {
                        return "inqure result json failed";
                    }
                    else
                    {
                        Value &key = result_doc["Key"];

                        ofstream file;
                        file.open(taskid.c_str(), ios::app);
                        
                        switch (type)
                        {
                            case CLOUD_QMACHINE_TYPE::Full_AMPLITUDE:
                                {
                                    Value &value = result_doc["Value"];

                                    for (SizeType i = 0; i < key.Size(); ++i)
                                    {
                                        file << key[i].GetInt() << " : " << value[i].GetDouble() << endl;
                                    }
                                    break;
                                }

                            case CLOUD_QMACHINE_TYPE::PARTIAL_AMPLITUDE:
                            case CLOUD_QMACHINE_TYPE::SINGLE_AMPLITUDE:
                                {
                                    Value &value_real = result_doc["ValueReal"];
                                    Value &value_imag = result_doc["ValueImag"];

                                    for (SizeType i = 0; i < key.Size(); ++i)
                                    {
                                        file << key[i].GetInt() << " : ("
                                             << value_real[i].GetDouble() << ","
                                             << value_imag[i].GetDouble() << ")" << endl;
                                    }
                                        break;
                                }

                            default: return "quantum machine type error";
                        }

                        file.close();
                        return "inqure result success";
                    }
                }
            }
        }
        catch (const std::exception&e)
        {
            return "json error";
        }
    }
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

void QCloudMachine::add_string_value(rapidjson::Document &doc, const string &key, const int &int_value)
{
    std::string value = to_string(int_value);
    add_string_value(doc, key, value);
}


REGISTER_QUANTUM_MACHINE(QCloudMachine);
#endif //USE_CURL

#endif


std::string QPanda::qProgToBinary(QProg prog, QuantumMachine* qm)
{
    auto avec = transformQProgToBinary(prog, qm);

    auto res = Base64::encode(avec);

    std::string bin_data(res.begin(), res.end());
    return bin_data;
}