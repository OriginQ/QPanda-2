#include <fstream>
#include "QuantumCloudHTTP.h"
#include "TinyXML/tinyxml.h"
#include "QPanda.h"
#include <map>
#include <math.h>
#include <algorithm>
#if 0
//#ifdef USE_CURL

#define COMPUTEAPI    "https://qcode.qubitonline.cn/api/QCode/submitTask.json"
#define INQUREAPI     "https://qcode.qubitonline.cn/api/QCode/quer
yTask.json"
#define TERMINATEAPI  "https://qcode.qubitonline.cn/api/QCode/terminateTask.json"

#define BETA_COMPUTEAPI    "http://10.10.12.53:4630/api/QCode/submitTask.json"
#define BETA_INQUREAPI     "http://10.10.12.53:4630/api/QCode/queryTask.json"
#define BETA_TERMINATEAPI  "http://10.10.12.53:4630/api/QCode/terminateTask.json"

#define BETA_APIKEY "4570596AD0F545BEA0A7D81D2169EF98"
#define RELEASE_APIKEY "02A564B301D34666AAE82CE5A4E6A389"

USING_QPANDA

enum TASK_STATUS
{
    WAITING = 1,
    COMPUTING,
    FINISHED
};

using std::ifstream;
using std::stringstream;
using namespace rapidjson;


int getQRunesTyp(std::string task_type, std::string &sQRunes)
{
    int    iQnum{ 0 };

    auto aPos = sQRunes.find("QINIT");
    if (aPos != sQRunes.npos)
    {
        auto spacePos = sQRunes.find(" ", aPos);
        auto linefeedPos = sQRunes.find("\n", aPos);

        if ((linefeedPos - spacePos) <= 1)
        {
            QCERR("Qubit Number Error");
            throw std::invalid_argument("invalid argument : qubit number");
        }
        else
        {
            iQnum = atoi(sQRunes.substr(spacePos + 1, linefeedPos - spacePos).c_str());
            if (iQnum <= 0)
            {
                QCERR("Qubit Number Error");
                throw std::invalid_argument("invalid argument : qubit number");
            }
        }
    }
    else
    {
        QCERR("QINIT NOT FOUND");
        throw std::invalid_argument("invalid argument : qubit number");
    }

    if (sQRunes.find(task_type) == sQRunes.npos)
    {
        return -1;
    }
    else if (task_type == "PMEASURE")
    {
        return iQnum > 20 ? 4 : 3;
    }
    else
    {
        return 2;
    }
}

size_t recvJsonData(void *ptr, size_t size, size_t nmemb, void *stream)
{
    std::string data((const char*)ptr,0,(size_t)(size * nmemb));

    *((stringstream*)stream) << data << std::endl;

    return size * nmemb;
}

void QuantumCloudHttp::configQuantumCloudHttp(const std::string& config_filepath)
{
    TiXmlDocument config_doc(config_filepath.c_str());
    bool loadSuccess = config_doc.LoadFile();
    if (!loadSuccess)
    {
        std::cout << "could not load the test file.Error:" << config_doc.ErrorDesc() << std::endl;
        throw std::invalid_argument("load failed");
    }

    TiXmlElement *RootElement = config_doc.RootElement();  

    TiXmlNode *BetaAPIKEY = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaAPIKEY")->FirstChild();
    TiXmlNode *BetaComputeAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaComputeAPI")->FirstChild();
    TiXmlNode *BetaInqureAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaInqureAPI")->FirstChild();
    TiXmlNode *BetaTerminateAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaTerminateAPI")->FirstChild();
    TiXmlNode *repeat_num = RootElement->FirstChildElement("QProg")
        ->FirstChildElement("MonteCarloRepeatNum");

    if (nullptr == BetaAPIKEY || nullptr == BetaComputeAPI   || nullptr == BetaInqureAPI
                              || nullptr == BetaTerminateAPI || nullptr == repeat_num)
    {
        QCERR("config Error");
        throw std::invalid_argument("config Error");
    }
    else
    {
        m_APIKey = BetaAPIKEY->Value();
        m_computeAPI = BetaComputeAPI->Value();
        m_inqureAPI = BetaInqureAPI->Value();
        m_terminateAPI = BetaTerminateAPI->Value();
        m_repeat_num = atoi(repeat_num->Value());
    }

}

QuantumCloudHttp::QuantumCloudHttp() :
    pCurl(nullptr),
    m_repeat_num(100)
{
    std::string config_filepath;
    char * QPanda_config_path = nullptr;
    QPanda_config_path = getenv("QPANDA_CONFIG_PATH");
    if (nullptr != QPanda_config_path)
    {
        config_filepath.append(QPanda_config_path);
        config_filepath.append("\\QuantumCloudConfig.xml");
        configQuantumCloudHttp(config_filepath);
    }
    else
    {
        m_APIKey = BETA_APIKEY;
        m_computeAPI = BETA_COMPUTEAPI;
        m_inqureAPI = BETA_INQUREAPI;
        m_terminateAPI = BETA_TERMINATEAPI;
    }

    curl_global_init(CURL_GLOBAL_ALL);
}

QuantumCloudHttp::~QuantumCloudHttp()
{
    curl_global_cleanup();
}

void QuantumCloudHttp::MeasureRun(int repeat_num)
{
    if (!getQProg())
    {
        QCERR("Bad program");
        throw std::invalid_argument("Bad program");
    }

    QProgToQRunes temp;
    temp.qProgToQRunes(dynamic_cast<AbstractQuantumProgram*>(getQProg().get()));
    std::string sQRunes(temp.insturctionsQRunes());

    if (-1 == getQRunesTyp("MEASURE", sQRunes))
    {
        QCERR("MEASURE NOT FOUND");
        throw std::invalid_argument("Bad program");
    }

    rapidjson::Document json_doc;
    rapidjson::Document::AllocatorType &allocator = json_doc.GetAllocator();
    rapidjson::Value root(kObjectType);
    rapidjson::Value json_elem(kStringType);

    json_elem.SetString(sQRunes.c_str(), (rapidjson::SizeType)sQRunes.size(), allocator);
    root.AddMember("qprog", json_elem, allocator);

    json_elem.SetString(m_APIKey.c_str(), (rapidjson::SizeType)m_APIKey.size(), allocator);
    root.AddMember("token", json_elem, allocator);

    root.AddMember("meaarr", "{1,10}", allocator);
    root.AddMember("taskTyp", "2", allocator);
    root.AddMember("typ", "mcpr", allocator);

    root.AddMember("repeat", repeat_num, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    root.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(m_computeAPI, post_json);

    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError() || !recv_doc.HasMember("obj"))
    {
        throw std::invalid_argument("invalid argument");
    }
    else
    {
        const rapidjson::Value &Obj = recv_doc["obj"];
        if (!Obj.IsObject() || !Obj.HasMember("taskid"))
        {
            throw std::invalid_argument("invalid argument");
        }
        else
        {
            const rapidjson::Value &Taskid = Obj["taskid"];
            m_taskid = Taskid.GetString();
        }
    }
}

bool QuantumCloudHttp::parserRecvJson(std::string recv_json,std::map<std::string,std::string>& recv_res)
{
    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError() || !recv_doc.HasMember("obj"))
    {
        throw std::invalid_argument("invalid argument");
    }
    else
    {
        Value &Obj = recv_doc["obj"];
        if (!Obj.IsObject() || !Obj.HasMember("tasksta"))
        {
            throw std::invalid_argument("invalid argument");
        }
        else
        {
            Value &tasksta = Obj["tasksta"];
            int status = atoi(tasksta.GetString());
            switch (status)
            {
            case TASK_STATUS::WAITING:
            case TASK_STATUS::COMPUTING:
            {
                std::cout << "Waiting or Computing ..." << std::endl << std::endl;
                return false;
            }
            case TASK_STATUS::FINISHED:
            {
                Document result_doc;
                Value &taskrs = Obj["taskrs"];
                if (result_doc.Parse(taskrs.GetString()).HasParseError() ||
                    !result_doc.HasMember("key") ||
                    !result_doc.HasMember("value"))
                {
                    QCERR("invalid argument , no result return");
                    throw std::invalid_argument("invalid argument");
                }
                else
                {
                    Value &key = result_doc["key"];
                    Value &value = result_doc["value"];

                    if (!key.IsArray() || !value.IsArray())
                    {
                        throw std::invalid_argument("invalid argument");
                    }
                    else
                    {
                        for (SizeType i = 0; i < key.Size(); ++i)
                        {
                            recv_res.insert(std::pair<std::string, std::string>
                                    (key[i].GetString(), std::to_string(value[i].GetDouble())));
                        }
                        return true;
                    }
                }
            }
            default:
                throw std::invalid_argument("invalid argument");
                break;
            }
        }
    }
}

void QuantumCloudHttp::run()
{
    MeasureRun(1);
}

std::string QuantumCloudHttp::postHttpJson(const std::string &sUrl, std::string & sJson)
{
    std::stringstream out;

    pCurl = curl_easy_init();

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-type: application/json");
    headers = curl_slist_append(headers, "accept: application/json");
    headers = curl_slist_append(headers, "Charset: UTF-8");
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

    res = curl_easy_perform(pCurl);
    if (CURLE_OK != res)
    {
        std::cout << "post failed : " << curl_easy_strerror(res) << std::endl;
        throw std::invalid_argument("post failed");
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(pCurl);

    return out.str().substr(out.str().find("{"));
}

std::map<std::string, double> QuantumCloudHttp::getProbDict(QVec& qubit_vec, int select_max)
{
    int task_typ = PMeasureRun(qubit_vec);

    std::string recv_json = inqureResult(std::to_string(task_typ));


    std::map<std::string, std::string> recv_res;
    while (!parserRecvJson(recv_json, recv_res))
    {
        Sleep(3000);
        recv_json = inqureResult(std::to_string(task_typ));
    }

    std::map<std::string, double> result_map;
    for (auto val : recv_res)
    {
        result_map.insert(std::pair<std::string, double>(val.first,atof(val.second.c_str())));
    }
    return result_map;
}

std::map<std::string, double> QuantumCloudHttp::
probRunDict(QProg & qProg, QVec& qubit_vec, int select_max)
{
    load(qProg);
    return getProbDict(qubit_vec, select_max);
}

std::string QuantumCloudHttp::inqureResult(std::string task_typ)
{
    if (0 == m_taskid.size())
    {
        throw std::invalid_argument("invalid_argument");
    }

    rapidjson::Document json_doc;
    rapidjson::Document::AllocatorType &allocator = json_doc.GetAllocator();
    rapidjson::Value root(kObjectType);
    rapidjson::Value json_elem(kStringType);


    json_elem.SetString(m_APIKey.c_str(), (rapidjson::SizeType)m_APIKey.size(), allocator);
    root.AddMember("token", json_elem, allocator);

    json_elem.SetString(m_taskid.c_str(), (rapidjson::SizeType)m_taskid.size(), allocator);
    root.AddMember("taskid", json_elem, allocator);
    root.AddMember("impTyp", 1, allocator);

    json_elem.SetString(task_typ.c_str(), (rapidjson::SizeType)task_typ.size(), allocator);
    root.AddMember("task_typ", json_elem, allocator);

    root.AddMember("typ", "qrytask", allocator);


    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    root.Accept(writer);

    std::string post_json = buffer.GetString();
    return postHttpJson(m_inqureAPI, post_json);

}

std::map<std::string, bool> QuantumCloudHttp::getResultMap()
{
    std::string recv_json = inqureResult("mcpr");
    std::map<std::string, std::string> recv_res;

    while (!parserRecvJson(recv_json,recv_res))
    {
        Sleep(3000);
        recv_json = inqureResult("mcpr");
    } 

    std::map<std::string, bool> result_map;
    std::string sKey = recv_res.cbegin()->first;

    for (int i = 0; i < (1ull << sKey.size()); ++i)
    {
        stringstream bin;
        for (size_t j = sKey.size() - 1; j > -1; --j)
        {
            bin << ((i >> j) & 1);
        }
        result_map.insert(std::pair<std::string, bool>(bin.str(), bin.str() == sKey));
    }
    return result_map;

}

std::map<std::string, size_t> QuantumCloudHttp::runWithConfiguration(QProg & qProg, 
                                    std::vector<ClassicalCondition>& cbit_vec, int shots)
{
    load(qProg);
    MeasureRun(shots);

    std::string recv_json = inqureResult("mcpr");

    std::map<std::string, std::string> recv_res;
    while (!parserRecvJson(recv_json, recv_res))
    {
        Sleep(3000);
        recv_json = inqureResult("mcpr");
    }

    std::map<std::string, size_t> result_map;

    size_t len = recv_res.cbegin()->first.size();
    for (int i = 0; i < (1ull << len); ++i)
    {
        stringstream bin;
        for (size_t j = len - 1; j > -1; --j)
        {
            bin << ((i >> j) & 1);
        }

        double prob= recv_res.find(bin.str()) == recv_res.end()?
            0.0 : atof(recv_res.find(bin.str())->second.c_str());

        result_map.insert(std::pair<std::string, size_t>
            (bin.str(), (size_t)round(prob*shots)));
    }

    return result_map;
}

int QuantumCloudHttp::PMeasureRun(QVec& qubit_vec)
{
    if (!getQProg())
    {
        QCERR("Bad program");
        throw std::invalid_argument("Bad program");
    }

    QProgToQRunes temp;
    temp.qProgToQRunes(dynamic_cast<AbstractQuantumProgram*>(getQProg().get()));
    std::string sQRunes(temp.insturctionsQRunes());
    
    sQRunes.append("\nPMEASURE ");
    for_each(qubit_vec.begin(), qubit_vec.end(), [&](Qubit *Qn) 
        {sQRunes.append(std::to_string(Qn->getPhysicalQubitPtr()->getQubitAddr())).append(","); });
    sQRunes.erase(sQRunes.size() - 1);

    rapidjson::Document json_doc;
    rapidjson::Document::AllocatorType &allocator = json_doc.GetAllocator();
    rapidjson::Value root(kObjectType);
    rapidjson::Value json_elem(kStringType);
    
    int task_typ = getQRunesTyp("PMEASURE", sQRunes);
    switch (task_typ)
    {
    case -1:
        QCERR("MEASURE NOT FOUND");
        throw std::invalid_argument("Bad program");
    case 3:
        root.AddMember("typ", "smapr", allocator);
        root.AddMember("task_typ", "3", allocator);
        break;
    case 4:
        root.AddMember("typ", "midpr", allocator);
        root.AddMember("task_typ", "4", allocator);
        break;
    default:
        throw std::invalid_argument("invalid argument");
    }
    
    json_elem.SetString(sQRunes.c_str(), (rapidjson::SizeType)sQRunes.size(), allocator);
    root.AddMember("qprog", json_elem, allocator);
    
    json_elem.SetString(m_APIKey.c_str(), (rapidjson::SizeType)m_APIKey.size(), allocator);
    root.AddMember("token", json_elem, allocator);
    root.AddMember("meaarr", "{1,10}", allocator);
        
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    root.Accept(writer);
    
    std::string post_json = buffer.GetString();
    std::string recv_json = postHttpJson(m_computeAPI, post_json);
    
    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError() || !recv_doc.HasMember("obj"))
    {
        throw std::invalid_argument("invalid argument");
    }
    else
    {
        const rapidjson::Value &Obj = recv_doc["obj"];
        if (!Obj.IsObject() || !Obj.HasMember("taskid"))
        {
            throw std::invalid_argument("invalid argument");
        }
        else
        {
            const rapidjson::Value &Taskid = Obj["taskid"];
            m_taskid = Taskid.GetString();
        }
    }

    return task_typ;
}

std::vector<std::pair<size_t, double>> QuantumCloudHttp::getProbTupleList(QVec &qubit_vec, int select_max)
{
    std::vector<std::pair<size_t, double>> result_vec;

    auto res = getProbDict(qubit_vec, select_max);
    int i{ 0 };
    for (auto val : res)
    {
        result_vec.emplace_back(std::pair<size_t, double>(++i, val.second));
    }

    return result_vec;

}

std::vector<std::pair<size_t, double>> QuantumCloudHttp::probRunTupleList(QProg &qProg, QVec &qubit_vec, int select_max)
{
    load(qProg);
    return getProbTupleList(qubit_vec, select_max);
}

std::vector<double> QuantumCloudHttp::getProbList(QVec &qubit_vec, int select_max)
{
    std::vector<double> result_vec;

    auto res = getProbDict(qubit_vec, select_max);
    for (auto val : res)
    {
        result_vec.emplace_back(val.second);
    }

    return result_vec;

}

std::vector<double> QuantumCloudHttp::probRunList(QProg &qProg, QVec &qubit_vec, int select_max)
{
    load(qProg);
    return getProbList(qubit_vec, select_max);
}

std::map<std::string, bool> QuantumCloudHttp::directlyRun(QProg & qProg)
{
    load(qProg);
    return getResultMap();
}

std::string QuantumCloudHttp::ResultToBinaryString(std::vector<ClassicalCondition>& cbit_vec)
{
    QCERR("Unsupported Function");
    throw std::invalid_argument("Unsupported Function");
}

std::map<std::string, size_t> QuantumCloudHttp::quickMeasure(QVec &, size_t)
{
    QCERR("Unsupported Function");
    throw std::invalid_argument("Unsupported Function");
}

REGISTER_QUANTUM_MACHINE(QuantumCloudHttp);

#endif 
