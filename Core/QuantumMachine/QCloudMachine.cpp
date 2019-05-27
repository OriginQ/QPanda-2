
#include <fstream>
#include <math.h>
#include <algorithm>
#include "ThirdParty/TinyXML/tinyxml.h"
#include "include/Core/Utilities/base64.hpp"
#include "include/Core/QuantumMachine/QCloudMachine.h"
#ifdef USE_CURL

#define DEFAULT_COMPUTEAPI    "http://10.10.12.53:4630/api/QCode/QRunes2/submitTask.json"
#define DEFAULT_INQUREAPI     "http://10.10.12.53:4630/api/QCode/QRunes2/queryTask.json"
#define DEFAULT_TOKEN         "3CD107AEF1364924B9325305BF046FF3"

USING_QPANDA
using namespace std;
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
        m_compute_url = DEFAULT_COMPUTEAPI;
        m_inqure_url = DEFAULT_INQUREAPI;
        m_token = DEFAULT_TOKEN;
    }
    else
    {
        map<string, string> QCloudConfig;
        bool is_success = config.getQuantumCloudConfig(QCloudConfig);
        if (!is_success)
        {
            QCERR("config error");
            m_compute_url = DEFAULT_COMPUTEAPI;
            m_inqure_url = DEFAULT_INQUREAPI;
            m_token = DEFAULT_TOKEN;
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
    auto prog_bin = QProgToBinary(prog,this);

    rapidjson::Value json_elem(kStringType);
    json_elem.SetString(prog_bin.c_str(), (rapidjson::SizeType)prog_bin.size(), allocator);
    parm.AddMember("QProg", json_elem, allocator);
    parm.AddMember("TaskType", TASK_TYPE::MEASURE, allocator);
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
    headers = curl_slist_append(headers,"Content-type: application/json");
    headers = curl_slist_append(headers,"accept: application/json");
    headers = curl_slist_append(headers,"Charset: UTF-8");
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
    auto prog_bin = QProgToBinary(prog,this);
    rapidjson::Document::AllocatorType &allocator = parm.GetAllocator();

    rapidjson::Value prog_elem(kStringType);
    prog_elem.SetString(prog_bin.c_str(),(SizeType)prog_bin.size(), allocator);

    parm.AddMember("QProg", prog_elem, allocator);
    parm.AddMember("TaskType",TASK_TYPE::PMEASURE, allocator);

#if 0
    rapidjson::Value qvec_elem(rapidjson::kArrayType);
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit)
    {qvec_elem.PushBack(qubit->getPhysicalQubitPtr()->getQubitAddr(), allocator); });
    parm.AddMember("Qubits", qvec_elem, allocator);
#endif

    vector<size_t> qvec_elem;    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit)    {qvec_elem.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr()); });    uint64_t qubit_array;    if (!qvec_elem.empty())    {        qubit_array = 1ull << qvec_elem[0];        for_each(qvec_elem.begin() + 1, qvec_elem.end(), [&](size_t qubit)        {qubit_array = qubit_array | (1ull << qubit); });    }    else    {        return "error";    }    parm.AddMember("Qubits", qubit_array, allocator);
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
    std::cout << parserRecvJson(recv_json, recv_res) << std::endl;

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

REGISTER_QUANTUM_MACHINE(QCloudMachine);
#endif //USE_CURL

#endif


std::string QPanda::QProgToBinary
(QProg prog, QuantumMachine* qm)
{
    auto avec = transformQProgToBinary(prog, qm);

    auto res = Base64::encode(avec);

    std::string bin_data(res.begin(), res.end());
    return bin_data;
}