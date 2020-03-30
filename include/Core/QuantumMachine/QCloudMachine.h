/*! \file QCloudMachine.h */
#ifndef QCLOUD_MACHINE_H
#define QCLOUD_MACHINE_H
#include "QPandaConfig.h"
#include "Core/Core.h"


#ifdef USE_CURL
#include <curl/curl.h>

#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
#include "Core/QuantumMachine/Factory.h"

QPANDA_BEGIN

enum CLOUD_QMACHINE_TYPE
{
    Full_AMPLITUDE,
    NOISE_QMACHINE,
    PARTIAL_AMPLITUDE,
    SINGLE_AMPLITUDE,
    CHEMISTRY
};

/**
* @namespace QPanda
*/

/*
* @class QCloudMachine
* @brief Quantum Cloud Machine  for connecting  QCloud server
* @ingroup QuantumMachine
* @see QuantumMachine
* @note  QCloudMachine also provides  python interface
*/
class QCloudMachine:public QVM
{
public:
    QCloudMachine();
    ~QCloudMachine();

    /**
    * @brief  Init the quantum machine environment
    * @return     void
    * @note   use this at the begin
    */
    void init();

    /**
    * @brief  Run a measure quantum program with json or dict configuration
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  rapidjson::Document&   config  message
    * @return     std::string  QCloud taskid  and  task status
    * @see
        * @code
            rapidjson::Document doc1;
            doc1.SetObject();
            rapidjson::Document::AllocatorType &allocator1 = doc1.GetAllocator();
            doc1.AddMember("BackendType", QMachineType::CPU, allocator1);
            doc1.AddMember("RepeatNum", 1000, allocator1);
            doc1.AddMember("token", "E5CD3EA3CB534A5A9DA60280A52614E1", allocator1);
            std::cout << QCM->runWithConfiguration(qprog, doc1) << endl;
        * @endcode
    */
    std::string runWithConfiguration(QProg &,rapidjson::Document &);

    /**
    * @brief  Run a pmeasure quantum program with json or dict configuration
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  QVec qubits list
    * @param[in]  rapidjson::Document&   config  message
    * @return     std::string  QCloud taskid  and  task status
    * @see
        * @code
            rapidjson::Document doc;
            doc.SetObject();
            rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

            doc.AddMember("BackendType", QMachineType::CPU, allocator);
            doc.AddMember("token", "E5CD3EA3CB534A5A9DA60280A52614E1", allocator);
            std::cout << QCM->probRunDict(qprog, qlist, doc) << endl;;
        * @endcode
    */
    std::string probRunDict(QProg &,QVec, rapidjson::Document &);

    std::map<std::string, double> getResult(std::string taskid);

    std::string full_amplitude_measure(QProg &, int shot);
  
    std::string full_amplitude_pmeasure(QProg &prog, const Qnum &qubit_vec);
    
    std::string partial_amplitude_pmeasure(QProg &prog, std::vector<std::string> &amplitude_vec);

    std::string single_amplitude_pmeasure(QProg &prog, std::string amplitude);

    std::string get_result(CLOUD_QMACHINE_TYPE type, std::string taskid);

private:
    std::string m_compute_url;
    std::string m_inqure_url;
    std::string m_token;

    enum CLOUD_TASK_TYPE
    {
        CLOUD_MEASURE = 0,
        CLOUD_PMEASURE
    }; 

    enum CLUSTER_TASK_TYPE
    {
        CLUSTER_MEASURE = 1,
        CLUSTER_PMEASURE
    };

    enum TASK_STATUS
    {
        WAITING = 1,
        COMPUTING,
        FINISHED,
        FAILED
    };

    std::string postHttpJson(const std::string &, std::string &);
    std::string parserRecvJson(std::string recv_json, std::map<std::string, double>& recv_res);

    std::string parser_cluster_result_json(std::string &recv_json);

    void add_string_value(rapidjson::Document &, const string &, const int &);
    void add_string_value(rapidjson::Document &, const string &, const std::string &);
};

QPANDA_END

#endif // USE_CURL

QPANDA_BEGIN
/**
* @brief  Quamtum program tramsform to binary data
* @ingroup  QuantumMachine
* @param[in]  size_t qubit num
* @param[in]  size_t cbit num
* @param[in]  QProg the reference to a quantum program
* @return     std::string  binary data
*/
std::string qProgToBinary(QProg, QuantumMachine*);
QPANDA_END

#endif // ! QCLOUD_MACHINE_H
