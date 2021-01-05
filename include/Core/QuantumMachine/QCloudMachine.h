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

enum class CLOUD_QMACHINE_TYPE
{
    Full_AMPLITUDE,
    NOISE_QMACHINE,
    PARTIAL_AMPLITUDE,
    SINGLE_AMPLITUDE,
    CHEMISTRY,
    REAL_CHIP
};

enum class REAL_CHIP_TYPE
{
    ORIGIN_WUYUAN
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

struct NoiseConfigs
{
    string noise_model;
    double single_gate_param;
    double double_gate_param;

    double single_p2; 
    double double_p2;

    double single_pgate; 
    double double_pgate;
};

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

    void set_compute_api(std::string url) { m_compute_url = url; }
    void set_inqure_api(std::string url) { m_inqure_url = url; }

    void set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params);

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> noise_measure(QProg &, int shot, string task_name = "Qurator Experiment");

	/**
	* @brief  run a measure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  int&   shot
	* @param[out] std::map<std::string, double>
	* @return     measure result
	*/
    std::map<std::string, double> full_amplitude_measure(QProg &, int shot, string task_name = "Qurator Experiment");
  
	/**
	* @brief  run a pmeasure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  Qnum & qubit address vector
    * @param[out] std::map<std::string, double>
    * @return     pmeasure result
	*/
    std::map<std::string, double> full_amplitude_pmeasure(QProg &prog, Qnum qubit_vec, string task_name = "Qurator Experiment");
    
	/**
	* @brief  run a pmeasure quantum program with partial amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::vector<std::string> & amplitude subset
    * @param[out] std::map<std::string, qcomplex_t>
    * @return     pmeasure result
	*/
    std::map<std::string, qcomplex_t> partial_amplitude_pmeasure(QProg &prog, std::vector<std::string> amplitude_vec, string task_name = "Qurator Experiment");

	/**
	* @brief  run a pmeasure quantum program with single amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::string amplitude
    * @param[out] qcomplex_t
    * @return     pmeasure result
	*/
    qcomplex_t single_amplitude_pmeasure(QProg &prog, std::string amplitude, string task_name = "Qurator Experiment");

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> real_chip_measure(QProg &, int shot, string task_name = "Qurator Experiment", REAL_CHIP_TYPE type = REAL_CHIP_TYPE::ORIGIN_WUYUAN);

    /**
    * @brief  get real chip qst matrix
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot 
    * @param[out] QStat matrix
    * @return     matrix
    */
    std::vector<QStat> get_state_tomography_density(QProg &, int shot);

	/**
	* @brief  get task result
	* @param[in]  std::string taskid
	* @param[in]  CLOUD_QMACHINE_TYPE type
	* @param[out] std::string& empty taskid
	* @return     string
	*/
	std::string get_result_json(std::string taskid, CLOUD_QMACHINE_TYPE type);

private:
    size_t m_retry_num = 0;

	std::string m_token;
	std::string m_inqure_url;  
    std::string m_compute_url;
     
    std::map<std::string, double> m_measure_result;
    std::map<std::string, qcomplex_t> m_pmeasure_result;
    std::vector<QStat> m_qst_result;
    qcomplex_t m_single_result;
    NoiseConfigs m_noise_params;

    enum CLUSTER_TASK_TYPE
    {
        CLUSTER_MEASURE = 1,
        CLUSTER_PMEASURE
    };

    enum class TASK_STATUS
    {
        WAITING = 1, 
        COMPUTING,
        FINISHED,
        FAILED,
		QUEUING,

        //The next status only appear in real chip backend
        SENT_TO_BUILD_SYSTEM,
        BUILD_SYSTEM_ERROR,   
        SEQUENCE_TOO_LONG,
        BUILD_SYSTEM_RUN
    };

    std::string post_json(const std::string &, std::string &);

    void inqure_result(std::string, CLOUD_QMACHINE_TYPE);

    bool parser_cluster_result_json(std::string &recv_json, std::string&);
	bool parser_cluster_submit_json(std::string &recv_json, std::string&);

    void add_string_value(rapidjson::Document &, const string &, const size_t);
    void add_string_value(rapidjson::Document &, const string &, const double);
    void add_string_value(rapidjson::Document &, const string &, const std::string &);
};

QPANDA_END

#endif // ! USE_CURL


#endif // ! QCLOUD_MACHINE_H
