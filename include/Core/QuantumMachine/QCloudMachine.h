/*! \file QCloudMachine.h */
#ifndef QCLOUD_MACHINE_H
#define QCLOUD_MACHINE_H
#include "QPandaConfig.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"

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
    REAL_CHIP,
    QST,
    FIDELITY
};

enum class REAL_CHIP_TYPE
{
    ORIGIN_WUYUAN_D4 = 5, //wuyuan no.2
    ORIGIN_WUYUAN_D5 =2 //wuyuan no.1
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
    std::string noise_model;
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
    void init(std::string token, bool is_logged = false);

    void set_compute_api(std::string url) { m_compute_url = url; }
    void set_inqure_api(std::string url) { m_inqure_url = url; }

    void set_real_chip_compute_api(std::string url) { m_real_chip_task_compute_url = url; }
    void set_real_chip_inqure_api(std::string url) { m_real_chip_task_inqure_url = url; }

    void set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params);

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> noise_measure(QProg &, int shot, std::string task_name = "Qurator Experiment");

	/**
	* @brief  run a measure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  int&   shot
	* @param[out] std::map<std::string, double>
	* @return     measure result
	*/
    std::map<std::string, double> full_amplitude_measure(QProg &, int shot, std::string task_name = "Qurator Experiment");
  
	/**
	* @brief  run a pmeasure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  Qnum & qubit address vector
    * @param[out] std::map<std::string, double>
    * @return     pmeasure result
	*/
    std::map<std::string, double> full_amplitude_pmeasure(QProg &prog, Qnum qubit_vec, std::string task_name = "Qurator Experiment");
    
	/**
	* @brief  run a pmeasure quantum program with partial amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::vector<std::string> & amplitude subset
    * @param[out] std::map<std::string, qcomplex_t>
    * @return     pmeasure result
	*/
    std::map<std::string, qcomplex_t> partial_amplitude_pmeasure(QProg &prog, std::vector<std::string> amplitude_vec, std::string task_name = "Qurator Experiment");

	/**
	* @brief  run a pmeasure quantum program with single amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::string amplitude
    * @param[out] qcomplex_t
    * @return     pmeasure result
	*/
    qcomplex_t single_amplitude_pmeasure(QProg &prog, std::string amplitude, std::string task_name = "Qurator Experiment");

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> real_chip_measure(
        QProg &prog, 
        int shot, 
        REAL_CHIP_TYPE chipid = REAL_CHIP_TYPE::ORIGIN_WUYUAN_D5,
        bool mapping_flag = true, 
        bool circuit_optimization = true, 
        std::string task_name = "Qurator Experiment");

    /**
    * @brief  get real chip qst matrix
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot  
    * @param[out] QStat matrix
    * @return     matrix
    */
    std::vector<QStat> get_state_tomography_density(
        QProg &prog, 
        int shot, 
        REAL_CHIP_TYPE chipid = REAL_CHIP_TYPE::ORIGIN_WUYUAN_D5,
        bool mapping_flag = true,
        bool circuit_optimization = true,
        std::string task_name = "Qurator Experiment");

    /**
    * @brief  get real chip qst fidelity
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] QStat matrix
    * @return     matrix
    */
    double get_state_fidelity(
        QProg &prog,
        int shot,
        REAL_CHIP_TYPE chipid = REAL_CHIP_TYPE::ORIGIN_WUYUAN_D5,
        bool mapping_flag = true,
        bool circuit_optimization = true,
        std::string task_name = "Qurator Experiment");

private:

    int m_retry_times;

    //Whether to print log
    bool m_is_logged = false;

    //url & token setting
	std::string m_token;
	std::string m_inqure_url;  
    std::string m_compute_url;

    std::string m_real_chip_task_inqure_url;
    std::string m_real_chip_task_compute_url;
     
    //measure result for full amplitude & noise 
    std::map<std::string, double> m_measure_result;

    //pmeasure result
    std::map<std::string, qcomplex_t> m_pmeasure_result;

    //qst result
    std::vector<QStat> m_qst_result;

    //qst result
    double m_qst_fidelity;

    //single amplitude
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
    std::string get_result_json(std::string taskid, std::string url,  CLOUD_QMACHINE_TYPE type);

    void inqure_result(std::string json, std::string url, CLOUD_QMACHINE_TYPE);

    bool parser_result_json(std::string &recv_json, std::string& taskid);
    bool parser_submit_json(std::string &recv_json, std::string&);

    void add_string_value(rapidjson::Document &, const std::string &, const size_t);
    void add_string_value(rapidjson::Document &, const std::string &, const double);
    void add_string_value(rapidjson::Document &, const std::string &, const std::string &);
};

QPANDA_END


#endif // ! QCLOUD_MACHINE_H
