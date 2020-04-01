/*! \file QCloudMachine.h */
#ifndef QCLOUD_MACHINE_H
#define QCLOUD_MACHINE_H
#include "QPandaConfig.h"
#include "Core/Core.h"

#ifdef USE_CURL

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
	void init(string token);

	/**
	* @brief  run a measure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  int&   shot
	* @param[out] std::string& empty taskid
	* @return     success or failure
	*/
    bool full_amplitude_measure(QProg &, int shot, std::string&);
  
	/**
	* @brief  run a pmeasure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  Qnum & qubit address vector
	* @param[out] std::string& empty taskid
	* @return     success or failure
	*/
	bool full_amplitude_pmeasure(QProg &prog, const Qnum &qubit_vec, std::string&);
    
	/**
	* @brief  run a pmeasure quantum program with partial amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::vector<std::string> & amplitude subset
	* @param[out] std::string& empty taskid
	* @return     success or failure
	*/
	bool partial_amplitude_pmeasure(QProg &prog, std::vector<std::string> &amplitude_vec, std::string&);

	/**
	* @brief  run a pmeasure quantum program with single amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::string amplitude
	* @param[out] std::string& empty taskid
	* @return     success or failure
	*/
	bool single_amplitude_pmeasure(QProg &prog, std::string amplitude, std::string&);

	/**
	* @brief  get task result
	* @param[in]  std::string taskid
	* @param[in]  CLOUD_QMACHINE_TYPE type
	* @param[out] std::string& empty taskid
	* @return     success or failure
	*/
	bool get_result(std::string taskid, CLOUD_QMACHINE_TYPE type);

private:
	std::string m_token;
	std::string m_inqure_url;
    std::string m_compute_url;

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
        FAILED,
		QUEUING
    };

    std::string post_json(const std::string &, std::string &);

	bool parser_cluster_result_json(std::string &recv_json, std::string&);
	bool parser_cluster_submit_json(std::string &recv_json, std::string&);

    void add_string_value(rapidjson::Document &, const string &, const int &);
    void add_string_value(rapidjson::Document &, const string &, const std::string &);
};

QPANDA_END

#endif // ! USE_CURL


#endif // ! QCLOUD_MACHINE_H
