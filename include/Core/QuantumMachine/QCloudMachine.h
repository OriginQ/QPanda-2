#ifndef QCLOUD_MACHINE_H
#define QCLOUD_MACHINE_H
#include "QPandaConfig.h"
#include "Core/Module/DataStruct.h"
#include "Core/QuantumMachine/Factory.h"
#include "Core/Utilities/Tools/QCloudConfig.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN

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

    void init(std::string token, bool is_logged = false);

    void set_qcloud_api(std::string url);
    void set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params);

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> noise_measure(QProg &, int shot, std::string task_name = "QPanda Experiment");

	/**
	* @brief  run a measure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  int&   shot
	* @param[out] std::map<std::string, double>
	* @return     measure result
	*/
    std::map<std::string, double> full_amplitude_measure(QProg &, int shot, std::string task_name = "QPanda Experiment");
  
	/**
	* @brief  run a pmeasure quantum program
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  Qnum & qubit address vector
    * @param[out] std::map<std::string, double>
    * @return     pmeasure result
	*/
    std::map<std::string, double> full_amplitude_pmeasure(QProg &prog, Qnum qubit_vec, std::string task_name = "QPanda Experiment");
    
	/**
	* @brief  run a pmeasure quantum program with partial amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::vector<std::string> & amplitude subset
    * @param[out] std::map<std::string, qcomplex_t>
    * @return     pmeasure result
	*/
    std::map<std::string, qcomplex_t> partial_amplitude_pmeasure(QProg &prog, std::vector<std::string> amplitude_vec, std::string task_name = "QPanda Experiment");

	/**
	* @brief  run a pmeasure quantum program with single amplitude backend
	* @param[in]  QProg& the reference to a quantum program
	* @param[in]  std::string amplitude
    * @param[out] qcomplex_t
    * @return     pmeasure result
	*/
    qcomplex_t single_amplitude_pmeasure(QProg &prog, std::string amplitude, std::string task_name = "QPanda Experiment");

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
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true, 
        bool is_optimization = true, 
        std::string task_name = "QPanda Experiment");

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
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

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
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    double get_expectation(QProg, const QHamiltonian&, const QVec&, TaskStatus& status, std::string task_name = "QPanda Experiment");
    std::string get_expectation_commit(QProg, const QHamiltonian&, const QVec&, TaskStatus& status, std::string task_name = "QPanda Experiment");
    double get_expectation_exec(std::string taskid, TaskStatus& status);
    double get_expectation_query(std::string taskid, TaskStatus& status);

    std::string full_amplitude_measure_commit(QProg &prog, int shot, TaskStatus& status, std::string task_name = "QPanda Experiment");
    std::string full_amplitude_pmeasure_commit(QProg &prog, Qnum qubit_vec, TaskStatus& status, std::string task_name = "QPanda Experiment");
    
    std::map<std::string, double> full_amplitude_measure_exec(std::string taskid, TaskStatus& status);
    std::map<std::string, qcomplex_t> full_amplitude_pmeasure_exec(std::string taskid, TaskStatus& status);

    std::map<std::string, double> full_amplitude_measure_query(std::string taskid, TaskStatus& status);
    std::map<std::string, qcomplex_t> full_amplitude_pmeasure_query(std::string taskid, TaskStatus& status);

    std::string get_last_error() { return m_error_info; };

    std::vector<std::map<std::string, double>> full_amplitude_measure_batch(
        std::vector<QProg>&, 
        int shot,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, double>> full_amplitude_pmeasure_batch(
        std::vector<QProg>&,
        Qnum qubit_vec,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, qcomplex_t>> partial_amplitude_pmeasure_batch(
        std::vector<QProg>&,
        std::vector<std::string> amplitude_vec,
        std::string task_name = "QPanda Experiment");

    std::vector<qcomplex_t> single_amplitude_pmeasure_batch(
        std::vector<QProg>&,
        std::string amplitude_vec,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, double>> noise_measure_batch(
        std::vector<QProg>&,
        int shot,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, double>> real_chip_measure_batch(
        std::vector<QProg>&, 
        int shot, RealChipType 
        chip_id = RealChipType::ORIGIN_WUYUAN_D3,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    std::map<size_t, std::string> full_amplitude_measure_batch_commit(
        std::vector<QProg>&,
        int shot, 
        TaskStatus& status,
        std::string task_name = "QPanda Experiment");

    std::map<size_t, std::string> full_amplitude_pmeasure_batch_commit(
        std::vector<QProg>&,
        Qnum qubit_vec, 
        TaskStatus& status,
        std::string task_name = "QPanda Experiment");

    std::map<size_t, std::string> real_chip_measure_batch_commit(
        std::vector<QProg>&,
        int shot, 
        TaskStatus& status,
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D3,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    std::map<size_t, std::map<std::string, double>> full_amplitude_measure_batch_query(
        std::map<size_t, std::string> taskid_map);

    std::map<size_t, std::map<std::string, double>> full_amplitude_pmeasure_batch_query(
        std::map<size_t, std::string> taskid_map);

    std::map<size_t, std::map<std::string, double>> real_chip_measure_batch_query(
        std::map<size_t, std::string> taskid_map);

    bool parser_submit_json_batch(std::string &recv_json, std::map<size_t, std::string>& taskid_vector);
    void inquire_batch_result(std::string json, std::string url, CloudQMchineType);
    std::string get_result_json_batch(std::map<size_t, std::string> taskid, std::string url, CloudQMchineType type);
    bool parser_result_json_batch(std::string &recv_json, std::map<size_t, std::string>& taskid_map);

    void set_inquire_url(std::string url) { m_inquire_url = url; }
    void set_compute_url(std::string url) { m_compute_url = url; }
    void set_batch_inquire_url(std::string url) { m_batch_inquire_url = url; }
    void set_batch_compute_url(std::string url) { m_batch_compute_url = url; }

private:
    
    TaskStatus m_task_status = TaskStatus::WAITING;

    //Whether to print log
    bool m_is_logged = false;

    //url & token setting
	std::string m_token;
	std::string m_inquire_url;  
    std::string m_compute_url;

    std::string m_batch_inquire_url;
    std::string m_batch_compute_url;
     
    //measure result for full amplitude & noise 
    std::map<std::string, double> m_measure_result;

    //error message taskid : error msg
    std::string m_error_info;

    //pmeasure result
    std::map<std::string, qcomplex_t> m_pmeasure_result;

    //qst result
    std::vector<QStat> m_qst_result;

    //qst result
    double m_qst_fidelity;

    //expectation result
    double m_expectation;

    //single amplitude
    qcomplex_t m_single_result;

    //noise config
    NoiseConfigs m_noise_params;

    //batch result
    std::map<size_t, std::map<std::string, double>> m_batch_measure_result;
    std::map<size_t, std::map<std::string, qcomplex_t>> m_batch_pmeasure_result;
    std::map<size_t, qcomplex_t> m_batch_single_result;

    enum ClusterTaskType
    {
        CLUSTER_MEASURE = 1,
        CLUSTER_PMEASURE = 2,
        CLUSTER_EXPECTATION
    };

    enum ClusterResultType
    {
        STATE_PROBS = 1,
        SINGLE_AMPLITUDE = 2,
        AMPLITUDE_ARRAY = 3,
        EXPECTATION
    };

    std::string post_json(const std::string &, std::string &);
    std::string get_result_json(std::string taskid, std::string url,  CloudQMchineType type);

    void inquire_result(std::string json, std::string url, CloudQMchineType);

    bool parser_result_json(std::string &recv_json, std::string&);
    bool parser_submit_json(std::string &recv_json, std::string&);

};

QPANDA_END


#endif // ! QCLOUD_MACHINE_H
