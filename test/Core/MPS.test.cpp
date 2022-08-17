#include <atomic>
#include <chrono>
#include <string>
#include "QPanda.h"
#include <algorithm>
#include "gtest/gtest.h"

#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/stringbuffer.h"

#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include <EigenUnsupported/Eigen/KroneckerProduct>

#define CHECK_RUN_TIME_BEGIN                  auto start_time = chrono::system_clock::now()\

#define CHECK_RUN_TIME_END_AND_COUT_SECONDS(argv)   auto final_time = chrono::system_clock::now();\
                                                    auto duration = chrono::duration_cast<chrono::microseconds>(final_time - start_time);\
                                                    std::cout << argv << " -> run_time counts :" << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << std::endl\

using namespace std;
using namespace rapidjson;

QCircuit qft_prog(const QVec& qvec)
{
    QCircuit qft = CreateEmptyCircuit();
    for (auto i = 0; i < qvec.size(); i++)
    {
        qft << H(qvec[qvec.size() - 1 - i]);
        for (auto j = i + 1; j < qvec.size(); j++)
        {
            qft << CR(qvec[qvec.size() - 1 - j],
                qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
        }
    }
    return qft;
}

QProg ghz_prog(const QVec& q)
{
    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < q.size() - 1; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }

    return prog;
}

static std::string build_chain_typed_quantum_chip_config_data(size_t qubit_num)
{
    rapidjson::Document arch_doc;
    arch_doc.SetObject();

    rapidjson::Document::AllocatorType &arch_alloc = arch_doc.GetAllocator();

    rapidjson::Value adj_object(rapidjson::kObjectType);
    for (SizeType qubit_addr = 0; qubit_addr < qubit_num; qubit_addr++)
    {
        rapidjson::Value qubit_array(rapidjson::kArrayType);

        if (qubit_addr != 0)
        {
            rapidjson::Value qubit_value(rapidjson::kObjectType);

            qubit_value.AddMember("v", qubit_addr - 1, arch_alloc);
            qubit_value.AddMember("w", 1.0, arch_alloc);

            qubit_array.PushBack(qubit_value, arch_alloc);
        }

        if (qubit_addr != (qubit_num - 1))
        {
            rapidjson::Value qubit_value(rapidjson::kObjectType);

            qubit_value.AddMember("v", qubit_addr + 1, arch_alloc);
            qubit_value.AddMember("w", 1.0, arch_alloc);

            qubit_array.PushBack(qubit_value, arch_alloc);
        }

        std::string qubit_addr_str = to_string(qubit_addr);

        rapidjson::Value qubit_value(kStringType);
        qubit_value.SetString(qubit_addr_str.c_str(), (rapidjson::SizeType)qubit_addr_str.size(), arch_alloc);

        adj_object.AddMember(qubit_value, qubit_array, arch_alloc);
    }

    arch_doc.AddMember("adj", adj_object, arch_alloc);
    arch_doc.AddMember("QubitCount", (SizeType)qubit_num, arch_alloc);

    //construct final json
    rapidjson::Document doc;
    doc.SetObject();

    rapidjson::Document::AllocatorType &alloc = doc.GetAllocator();

    doc.AddMember("QuantumChipArch", arch_doc, alloc);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    return buffer.GetString();
}
#if 0
void test_real_chip()
{
    QCloudMachine QCM;;

    //QCM.init("C60FBD87EF084DBA820945D052218AA8", true);
    QCM.init("E02BB115D5294012AA88D4BE82603984", true);

    //QCM.set_qcloud_api("http://10.10.10.197:8060");
    auto q = QCM.allocateQubits(4);
    auto c = QCM.allocateCBits(4);

    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(q)
        << CNOT(q[1], q[2]).control({q[0]})
        << Measure(q[0], c[0])
        << Measure(q[1], c[1]);

    std::vector<QProg> progs(3, measure_prog);

    /*auto result = QCM.real_chip_measure_batch(progs, 1000, RealChipType::ORIGIN_WUYUAN_D4, true, true);
    for (auto val : result)
    {
        for (auto val1 : val)
        {
            std::cout << val1.first << " : " << val1.second << std::endl;
        }
    }*/

    //auto result3 = QCM.get_state_fidelity(measure_prog, 1000, RealChipType::ORIGIN_WUYUAN_D4);
    //cout << result3 << endl;

    auto result = QCM.real_chip_measure(measure_prog, 1000, RealChipType::ORIGIN_WUYUAN_D5,true, true);
    for (auto val : result)
    {
        std::cout << val.first << " : " << val.second << std::endl;
    }

    auto result2 = QCM.get_state_tomography_density(measure_prog, 1000, RealChipType::ORIGIN_WUYUAN_D4);
    for (auto val : result2)
    {
        cout << val << endl;
    }




    QCM.finalize();
    return;
}

void test_qcloud()
{
    try
    {
        QCloudMachine QCM;
        QCM.init("E02BB115D5294012AA88D4BE82603984", true);

        //QCM.set_qcloud_api("http://www.72bit.com");
        //QCM.set_qcloud_api("http://10.10.10.39:8060");

        auto q = QCM.allocateQubits(4);
        auto c = QCM.allocateCBits(4);

        std::vector<QProg> measure_prog_array(3);
        measure_prog_array[0] << H(q[0]) << MeasureAll(q, c);
        measure_prog_array[1] << H(q[0]) << CNOT(q[0], q[1]) << MeasureAll(q, c);
        measure_prog_array[2] << H(q[0]) << H(q[1]) << H(q[2]) << MeasureAll(q, c);

        std::vector<QProg> pmeasure_prog_array(3);
        pmeasure_prog_array[0] << H(q[0]) << H(q[1]);
        pmeasure_prog_array[1] << H(q[0]) << CNOT(q[0], q[1]);
        pmeasure_prog_array[2] << H(q[0]) << H(q[1]) << H(q[2]);


        //auto measure_result = QCM.full_amplitude_measure_batch(measure_prog_array, 10000);
        //auto measure_result = QCM.full_amplitude_measure(measure_prog_array[0], 10000);
        //TaskStatus status;
        //auto taskids = QCM.real_chip_measure_batch_commit(measure_prog_array, 10000, status);
        //auto real_result = QCM.real_chip_measure_batch(measure_prog_array, 10000);
        //auto a = QCM.real_chip_measure_batch_query(taskids);
        //auto pmeasure_result = QCM.full_amplitude_pmeasure_batch(pmeasure_prog_array, { 0,1,2,3,4,5 });
        //auto partial_pmeasure_result = QCM.partial_amplitude_pmeasure_batch(pmeasure_prog_array, { "0","1","2" });
        //auto single_pmeasure_result = QCM.single_amplitude_pmeasure_batch(pmeasure_prog_array, "0");

        //for (auto item : measure_result)
        //{
            //for (auto val : item)
            //{
                //cout << val.first << " : " << val.second << endl;
            //}
        //}

        //QCM.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, { 0.9 }, { 0.9 });
        //auto noise_measure_result = QCM.noise_measure(measure_prog_array[0], 10000);

        //for (auto item : measure_result)
        //{
            //for (auto val : item)
            //{
                //cout << val.first << " : " << val.second << endl;
            //}
        //}

        cout << "--------------" << endl;

        //for (auto item : real_measure_result)
        //{
            //cout << "--------------" << endl;
            //for (auto val : item)
            //{
            //    cout << val.first << " : " << val.second << endl;
            //}
        //}

        cout << "--------------" << endl;

        //for (auto item : pmeasure_result)
        //{
            //cout << "--------------" << endl;
            //for (auto val : item)
            //{
                //cout << val.first << " : " << val.second << endl;
            //}
        //}

        cout << "--------------" << endl;


        cout << "--------------" << endl;

        //for (auto item : single_pmeasure_result)
        //{
        //    cout << item << endl;
        //}

        cout << "--------------" << endl;



        QCM.finalize();
    }
    catch (const std::exception e)
    {
        cout << e.what() << endl;
    }

    return;
}

void curl_post_and_recv()
{
    //curl_global_init(CURL_GLOBAL_ALL);

    cout << "start curl_post_and_recv" << endl;

    std::string url = "https://qcloud.originqc.com.cn/api/taskApi/getTaskDetail.json";

    rabbit::document doc;
    doc.parse("{}");

    doc.insert("taskId", "865D1C7D6DF94C2EA7229912529EADE4");
    doc.insert("QMachineType", "5");

    std::string post_string = doc.str();

    auto post_curl = curl_easy_init();
    curl_slist * headers = nullptr;

    headers = curl_slist_append(headers, "Content-Type: application/json;charset=UTF-8");
    headers = curl_slist_append(headers, "Connection: keep-alive");
    headers = curl_slist_append(headers, "Server: nginx/1.16.1");
    headers = curl_slist_append(headers, "Transfer-Encoding: chunked");
    headers = curl_slist_append(headers, "origin-language: en");

    curl_easy_setopt(post_curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(post_curl, CURLOPT_TIMEOUT, 60);
    curl_easy_setopt(post_curl, CURLOPT_CONNECTTIMEOUT, 30);
    curl_easy_setopt(post_curl, CURLOPT_HEADER, 0);
    curl_easy_setopt(post_curl, CURLOPT_POST, 1);
    curl_easy_setopt(post_curl, CURLOPT_SSL_VERIFYHOST, 0);
    curl_easy_setopt(post_curl, CURLOPT_SSL_VERIFYPEER, 0);

    curl_easy_setopt(post_curl, CURLOPT_READFUNCTION, nullptr);
    curl_easy_setopt(post_curl, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(post_curl, CURLOPT_WRITEFUNCTION, recv_json_data);

    std::stringstream out;
    curl_easy_setopt(post_curl, CURLOPT_URL, url.c_str());
    //curl_easy_setopt(m_post_curl, CURLOPT_VERBOSE, 1);
    curl_easy_setopt(post_curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(post_curl, CURLOPT_POSTFIELDS, post_string.c_str());
    curl_easy_setopt(post_curl, CURLOPT_POSTFIELDSIZE, post_string.size());

    cout << "curl_easy_setopt end" << endl;

    CURLcode res;
    res = curl_easy_perform(post_curl);

    cout << "curl_easy_perform end" << res  << endl;

    if (CURLE_OK != res)
    {
        std::string error_msg = curl_easy_strerror(res);
        QCERR_AND_THROW(run_fail, error_msg);
    }

    cout << out.str() << endl;

    curl_slist_free_all(headers);
    curl_easy_cleanup(post_curl);

    //curl_global_cleanup();

    return;
}

void qcloud_post_and_recv()
{
    QCloudMachine QCM;;

    QCM.init("E02BB115D5294012AA88D4BE82603984", true); // online
    //QCM.init("A774C41F15E44FA19EF2171AA82003E9", true); //test
    
    QCM.set_qcloud_api("http://pyqanda-admin.qpanda.cn");
    auto q = QCM.allocateQubits(4);
    auto c = QCM.allocateCBits(4);

    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(q)
        << CNOT(q[1], q[2]).control({ q[0] })
        << Measure(q[0], c[0])
        << Measure(q[1], c[1]);

    auto result = QCM.real_chip_measure(measure_prog, 1000, RealChipType::ORIGIN_WUYUAN_D5, true, true);
    for (auto val : result)
    {
        std::cout << val.first << " : " << val.second << std::endl;
    }

    QCM.finalize();
    return;
}

#include <thread>
#include "Core/Utilities/Tools/ThreadPool.h"
TEST(MPS, test)
{
    std::thread threads[10];
    std::cout << "spawning threads..." << endl;

    //curl_global_init(CURL_GLOBAL_ALL);

    for (int i = 0; i < 10; i++) 
    {
        threads[i] = std::thread(qcloud_post_and_recv);
    }

    //curl_global_cleanup();

    for (auto& t : threads) 
        t.join();

    std::cout << "all threads joined.";

    //curl_post_and_recv(post_json, url);

}

#endif
