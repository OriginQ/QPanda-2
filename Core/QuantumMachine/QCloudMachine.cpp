#include <math.h>
#include <fstream>
#include <algorithm>
#include "ThirdParty/TinyXML/tinyxml.h"
#include "Core/Utilities/Tools/base64.hpp"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"

#ifdef USE_CURL
#include <curl/curl.h>

#define DEFAULT_CLUSTER_COMPUTEAPI    "http://10.10.12.176:8060/api/taskApi/submitTask.json"
#define DEFAULT_CLUSTER_INQUREAPI     "http://10.10.12.176:8060/api/taskApi/getResultDetail.json"

USING_QPANDA
using namespace std;
using namespace Base64;
using namespace rapidjson;

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
	QCERR("missing parameter : token");
	throw invalid_argument("token");
}

void QCloudMachine::init(string token)
{
    XmlConfigParam config;
    if (!config.loadFile(CONFIG_PATH))
    {
        m_compute_url = DEFAULT_CLUSTER_COMPUTEAPI;
        m_inqure_url = DEFAULT_CLUSTER_INQUREAPI;
    }
    else
    {
        map<string, string> QCloudConfig;
        bool is_success = config.getQuantumCloudConfig(QCloudConfig);
        if (!is_success)
        {
            QCERR("config error");
            m_compute_url = DEFAULT_CLUSTER_COMPUTEAPI;
            m_inqure_url = DEFAULT_CLUSTER_INQUREAPI;
        }
        else
        {
            m_compute_url = QCloudConfig["ComputeAPI"];
            m_inqure_url = QCloudConfig["InqureAPI"];
        }
    }

	m_token = token;
    _start();
}

size_t recvJsonData
(void *ptr, size_t size, size_t nmemb, void *stream)
{
    std::string data((const char*)ptr, 0, (size_t)(size * nmemb));

    *((std::stringstream*)stream) << data << std::endl;

    return size * nmemb;
}

std::string QCloudMachine::post_json
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

    /*curl_easy_setopt(pCurl, CURLOPT_VERBOSE, 1);*/

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

bool QCloudMachine::full_amplitude_measure(QProg &prog, int shot, std::string& taskid)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubit());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_MEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMem());
    add_string_value(doc, "shot", shot);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(DEFAULT_CLUSTER_COMPUTEAPI, post_json_str);

	return parser_cluster_submit_json(recv_json_str, taskid);
}

bool QCloudMachine::full_amplitude_pmeasure
(QProg &prog,const Qnum &qubit_vec, std::string& taskid)
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
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    doc.AddMember("qubits", qubit_array, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(DEFAULT_CLUSTER_COMPUTEAPI, post_json_str);

	return parser_cluster_submit_json(recv_json_str, taskid);
}

bool QCloudMachine::partial_amplitude_pmeasure
(QProg &prog, std::vector<std::string> &amplitude_vec, std::string& taskid)
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
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::PARTIAL_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    doc.AddMember("Amplitude", amplitude_array, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(DEFAULT_CLUSTER_COMPUTEAPI, post_json_str);
	
	return parser_cluster_submit_json(recv_json_str, taskid);
}

bool QCloudMachine::single_amplitude_pmeasure
(QProg &prog, std::string amplitude, std::string& taskid)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    //construct json
    rapidjson::Document doc;

    add_string_value(doc, "code", prog_str);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (int)CLOUD_QMACHINE_TYPE::SINGLE_AMPLITUDE);
    add_string_value(doc, "codeLen", prog_str.size());
    add_string_value(doc, "qubitNum", getAllocateQubitNum());
    add_string_value(doc, "measureType", (int)CLUSTER_TASK_TYPE::CLUSTER_PMEASURE);
    add_string_value(doc, "classicalbitNum", getAllocateCMemNum());
    add_string_value(doc, "Amplitude", amplitude);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(DEFAULT_CLUSTER_COMPUTEAPI, post_json_str);

	return parser_cluster_submit_json(recv_json_str, taskid);
}

bool QCloudMachine::parser_cluster_submit_json(std::string &recv_json, std::string& taskid)
{
    Document recv_doc;
    if (recv_doc.Parse(recv_json.c_str()).HasParseError()
        || !recv_doc.HasMember("obj") || !recv_doc.HasMember("success"))
    {
		std::cout << "recv json error" << std::endl;
        return false;
    }
    else
    {
        const rapidjson::Value &success = recv_doc["success"];
        if (!success.GetBool())
        {
			std::cout << "recv json error" << std::endl;
            return false;
        }
        else
        {
            const rapidjson::Value &Obj = recv_doc["obj"];
            if (!Obj.IsObject() ||
                !Obj.HasMember("taskId") ||
                !Obj.HasMember("id"))
            {
				std::cout << "json object error" << std::endl;
				return false;
            }
            else
            {
                const rapidjson::Value &task_value = Obj["taskId"];
				taskid = task_value.GetString();

				std::cout << "submit task " << taskid 
					      << " success" << std::endl;
				return true;
            }
        }
    }
}

bool QCloudMachine::parser_cluster_result_json(std::string &recv_json, std::string& taskid)
{
	Document recv_doc;
	if (recv_doc.Parse(recv_json.c_str()).HasParseError()
		|| !recv_doc.HasMember("obj") || !recv_doc.HasMember("success"))
	{
		std::cout << "result json error" << std::endl;
		return false;
	}
	else
	{
		try
		{
			const rapidjson::Value &Obj = recv_doc["obj"];
			const rapidjson::Value &Val = Obj["qcodeTaskNewVo"];
			const rapidjson::Value &List = Val["taskResultList"];
			const rapidjson::Value &result = List[0]["taskResult"];

			std::string state = List[0]["taskState"].GetString();
			std::string qtype = List[0]["rQMachineType"].GetString();

			switch (atoi(state.c_str()))
			{
				case TASK_STATUS::FINISHED:
				{
					Document result_doc;
					result_doc.Parse(result.GetString());
					Value &key = result_doc["Key"];

					ofstream file;
					file.open(taskid.c_str(), ios::app);

					switch (atoi(qtype.c_str()))
					{
						case CLOUD_QMACHINE_TYPE::Full_AMPLITUDE:
						{
							Value &value = result_doc["Value"];

							for (SizeType i = 0; i < key.Size(); ++i)
							{
								file << key[i].GetInt() << " : " 
									 << value[i].GetDouble() 
									 << std::endl;
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
									 << value_imag[i].GetDouble() << ")" 
									 << std::endl;
							}
							break;
						}

						default:
						{
							std::cout << "QMachine type error" << std::endl;
							return false;
						}
					}

					file.close();
					std::cout << "result " << taskid<< " download completed" 
						      << std::endl;
					return true;
				}

				case TASK_STATUS::FAILED:
					std::cout << "Task " << taskid << " Failed " << std::endl;
					return false;

				case TASK_STATUS::WAITING:
					std::cout << "Task " << taskid << " is Waiting " << std::endl;
					return false;

				case TASK_STATUS::QUEUING:
					std::cout << "Task " << taskid << " is Queuing " << std::endl;
					return false;

				case TASK_STATUS::COMPUTING:
					std::cout << "Task " << taskid << " is Computing " << std::endl;
					return false;

				default:
					std::cout << "Task " << taskid << " status error " << std::endl;
					return false;
			}

		}
		catch (const std::exception&e)
		{
			std::cout << "parse result exception : " << e.what() << std::endl;
			return false;
		}
	}
}

bool QCloudMachine::get_result(std::string taskid, CLOUD_QMACHINE_TYPE type)
{
    rapidjson::Document doc;
    doc.SetObject();

    add_string_value(doc, "taskId", taskid);
    add_string_value(doc, "apiKey", m_token);
    add_string_value(doc, "QMachineType", (int)type);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json_str = buffer.GetString();
    std::string recv_json_str = post_json(DEFAULT_CLUSTER_INQUREAPI, post_json_str);

	return parser_cluster_result_json(recv_json_str, taskid);
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
