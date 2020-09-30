#include "QPanda.h"

#ifdef USE_CURL
#include <chrono>
#include <thread>
#include <curl/curl.h>
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
    //std::string prog_str = "QINIT 10 CREG 10 H q[0] MEASURE q[0],c[0]";
    rapidjson::Value code_str(rapidjson::kStringType);
    code_str.SetString(prog_str.c_str(), prog_str.size());

    doc.AddMember("code", code_str, allocator);
    doc.AddMember("apiKey", "3B1AC640AAC248C6A7EE4E8D8537370D", allocator);
    doc.AddMember("QMachineType", "0", allocator);
    doc.AddMember("codeLen", "100", allocator);
    doc.AddMember("qubitNum", "10", allocator);
    doc.AddMember("measureType", "1", allocator);
    doc.AddMember("classicalbitNum", "1", allocator);
    doc.AddMember("shot", "100", allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string post_json = buffer.GetString();
    std::string url = "https://qcloud.qubitonline.cn/api/taskApi/submitTask.json";

    std::stringstream out;
    curl_global_init(CURL_GLOBAL_ALL);

    auto pCurl = curl_easy_init();

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json;charset=UTF-8");
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

    //curl_easy_setopt(pCurl, CURLOPT_VERBOSE, 1);

    curl_easy_setopt(pCurl, CURLOPT_WRITEDATA, &out);

    auto res = curl_easy_perform(pCurl);
    if (CURLE_OK != res)
    {
        stringstream errMsg;
        errMsg << "post failed : " << curl_easy_strerror(res) << std::endl;
        cout << errMsg.str() <<endl;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(pCurl);

    curl_global_cleanup();

    cout << out.str() << endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
}

TEST(CloudHttp, Cluster)
{
    //curl_test();

    QCloudMachine QCM;;
    QCM.init("3B1AC640AAC248C6A7EE4E8D8537370D");
    auto qlist = QCM.allocateQubits(6);
    auto clist = QCM.allocateCBits(6);

    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(qlist)
        << CZ(qlist[2], qlist[3])
        << Measure(qlist[0], clist[0]);

    auto pmeasure_prog = QProg();
    pmeasure_prog << HadamardQCircuit(qlist)
        << CZ(qlist[1], qlist[5])
        << RX(qlist[2], PI / 4)
        << RX(qlist[1], PI / 4);

    //auto result0 = QCM.full_amplitude_measure(measure_prog, 100);
    //for (auto val : result0)
    //{
    //    cout << val.first << " : " << val.second << endl;
    //}

    //auto result1 = QCM.full_amplitude_pmeasure(pmeasure_prog, { 0, 1, 2 });
    //for (auto val : result1)
    //{
    //    cout << val.first << " : " << val.second << endl;
    //}

    //auto result2 = QCM.partial_amplitude_pmeasure(pmeasure_prog, { "0", "1", "2" });
    //for (auto val : result2)
    //{
    //    cout << val.first << " : " << val.second << endl;
    //}

    //auto result3 = QCM.single_amplitude_pmeasure(pmeasure_prog, "0");
    //cout << "0" << " : " << result3 << endl;

    //QCM.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, { 0.01 }, { 0.02 });
    //auto result4 = QCM.noise_measure(measure_prog, 100);
    //for (auto val : result4)
    //{
    //    cout << val.first << " : " << val.second << endl;
    //}

    auto result4 = QCM.real_chip_measure(measure_prog, 1000);
    for (auto val : result4)
    {
        cout << val.first << " : " << val.second << endl;
    }

    //auto result5 = QCM.get_state_tomography_density(measure_prog, 1000);
    //for (auto val : result5)
    //{
    //    for (auto val1 : val)
    //    {
    //        cout << val1 << endl;
    //    }
    //}

    QCM.finalize();
}

#endif // USE_CURL

