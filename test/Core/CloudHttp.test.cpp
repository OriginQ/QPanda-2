#ifdef USE_CURL
#include "QPanda.h"
#include "gtest/gtest.h"
#include "QuantumMachine/QCloudMachine.h"
using namespace std;
using namespace rapidjson;
USING_QPANDA

static size_t recv_json_data
(void *ptr, size_t size, size_t nmemb, void *stream)
{
    std::string data((const char*)ptr, 0, (size_t)(size * nmemb));

    *((std::stringstream*)stream) << data << std::endl;

    return size * nmemb;
}


void curl_test()
{
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    std::string prog_str = "QINIT 10\nCREG 10\nH q[0]\n";
    rapidjson::Value code_str(rapidjson::kStringType);
    code_str.SetString(prog_str.c_str(), prog_str.size());

    doc.AddMember("code", code_str, allocator);
    doc.AddMember("apiKey", "AB3FE594418E49F6B5BD0ABD0FF7EEAD", allocator);
    doc.AddMember("QMachineType", (int)CLOUD_QMACHINE_TYPE::Full_AMPLITUDE, allocator);
    doc.AddMember("codeLen", (int)prog_str.size(), allocator);
    doc.AddMember("qubitNum", 10, allocator);
    doc.AddMember("measureType", 1, allocator);
    doc.AddMember("classicalbitNum", 10, allocator);
    doc.AddMember("shot", 100, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json = buffer.GetString();
    cout << post_json << endl;
    std::string url = "http://10.10.12.176:8060/api/taskApi/submitTask.json";

    std::stringstream out;
    auto pCurl = curl_easy_init();

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type:application/json;charset=UTF-8");
    headers = curl_slist_append(headers, "Connection: keep-alive");
    headers = curl_slist_append(headers, "Server: nginx/1.16.1");
    headers = curl_slist_append(headers, "Transfer-Encoding: chunked");
    curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(pCurl, CURLOPT_TIMEOUT, 3);

    curl_easy_setopt(pCurl, CURLOPT_CONNECTTIMEOUT, 3);

    curl_easy_setopt(pCurl, CURLOPT_URL, url.c_str());

    curl_easy_setopt(pCurl, CURLOPT_HEADER, true);

    curl_easy_setopt(pCurl, CURLOPT_POST, true);

    curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYHOST, false);

    curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYPEER, false);

    curl_easy_setopt(pCurl, CURLOPT_READFUNCTION, NULL);

    curl_easy_setopt(pCurl, CURLOPT_NOSIGNAL, 1);

    curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, post_json.c_str());

    curl_easy_setopt(pCurl, CURLOPT_POSTFIELDSIZE, post_json.size());

    curl_easy_setopt(pCurl, CURLOPT_WRITEFUNCTION, recv_json_data);

    curl_easy_setopt(pCurl, CURLOPT_VERBOSE, 1);

    curl_easy_setopt(pCurl, CURLOPT_WRITEDATA, &out);

    auto res = curl_easy_perform(pCurl);
    if (CURLE_OK != res)
    {
        stringstream errMsg;
        errMsg << "post failed : " << curl_easy_strerror(res) << std::endl;
        cout << errMsg.str() << endl;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(pCurl);

    cout << out.str().substr(out.str().find("{")) << endl;

    Sleep(1);
}




TEST(CloudHttp, Cluster)
{
    QCloudMachine QCM;;
    QCM.init("4B7AFE1E196A4197B7C6845C4E73EF2E");
    auto qlist = QCM.allocateQubits(10);
    auto clist = QCM.allocateCBits(10);
    auto prog = QProg();

    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[7])
        << CZ(qlist[0], qlist[4])
        << RZ(qlist[7], PI / 4)
        << RX(qlist[5], PI / 4)
        << RX(qlist[4], PI / 4);

    std::vector<std::string> amplitude_vector = { "0","1" };
    Qnum qvec = { 0,1 };
    

	std::string task_id;
	if(QCM.full_amplitude_measure(prog, 100, task_id));
	{
		QCM.get_result(task_id, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);
	}

	QCM.get_result(task_id, CLOUD_QMACHINE_TYPE::Full_AMPLITUDE);

	
    //std::cout << QCM.full_amplitude_pmeasure(prog, qvec) << endl;
    //std::cout << QCM.partial_amplitude_pmeasure(prog, amplitude_vector) << endl;
    //std::cout << QCM.single_amplitude_pmeasure(prog, "0") << endl;
    


    QCM.finalize();
}

#endif // USE_CURL


