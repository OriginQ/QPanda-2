#pragma once

#include "QPandaConfig.h"
#include "Core/QuantumCloud/QCloudMachineImp.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"

#if defined(USE_CURL)

QPANDA_BEGIN

/*
* @class QCloudMachine
* @brief Quantum Cloud Machine For Connecting Origin QCloud
* @ingroup QuantumMachine
*/
class QCloudMachine :public QVM
{
public:
    QCloudMachine();
    ~QCloudMachine();

    void init(std::string token,
        bool is_logged = false,
        bool use_bin_or_hex_format = true /*true -> bin, false -> hex*/,
        bool enable_pqc_encryption = false,
        std::string random_num = generate_random_hex(96));

    void set_qcloud_url(std::string url);
    void set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params);

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> noise_measure(
        QProg& prog,
        int shots,
        std::string task_name = "QPanda Experiment");

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> full_amplitude_measure(
        QProg& prog, 
        int shot,
        std::string task_name = "QPanda Experiment");

    /**
    * @brief  run a pmeasure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  Qnum & qubit address vector
    * @param[out] std::map<std::string, double>
    * @return     pmeasure result
    */
    std::map<std::string, double> full_amplitude_pmeasure(
        QProg& prog,
        Qnum qubit_vec,
        std::string task_name = "QPanda Experiment");

    std::string async_noise_measure(
        QProg& prog,
        int shots,
        std::string task_name = "QPanda Experiment");

    std::string async_full_amplitude_measure(
        QProg& prog,
        int shot,
        std::string task_name = "QPanda Experiment");

    std::string async_full_amplitude_pmeasure(
        QProg& prog,
        Qnum qubit_vec,
        std::string task_name = "QPanda Experiment");

    std::string async_real_chip_measure(
        QProg& prog,
        int shot,
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    std::string async_batch_real_chip_measure(
        std::vector<QProg>& prog_vector,
        int shot,
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    std::map<std::string, double> query_state_result(std::string task_id);
    std::vector<std::map<std::string, double>> query_batch_state_result(std::string task_id, bool open_loop = false);


    /**
    * @brief  run a pmeasure quantum program with partial amplitude backend
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  std::vector<std::string> & amplitude subset
    * @param[out] std::map<std::string, qcomplex_t>
    * @return     pmeasure result
    */
    std::map<std::string, qcomplex_t> partial_amplitude_pmeasure(
        QProg& prog, 
        std::vector<std::string> amplitudes, 
        std::string task_name = "QPanda Experiment");

    /**
    * @brief  run a pmeasure quantum program with single amplitude backend
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  std::string amplitude
    * @param[out] qcomplex_t
    * @return     pmeasure result
    */
    qcomplex_t single_amplitude_pmeasure(
        QProg& prog, 
        std::string amplitude, 
        std::string task_name = "QPanda Experiment");

    /**
    * @brief  run a measure quantum program
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] std::map<std::string, double>
    * @return     measure result
    */
    std::map<std::string, double> real_chip_measure(
        QProg& prog,
        int shot,
        RealChipType chip_id = RealChipType::ORIGIN_72,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    //zne
    std::vector<double> zne_error_mitigation(
        QProg& prog,
        int shot,
        std::vector<std::string> expectations,
        std::vector<double> noise_strength,
        RealChipType chip_id = RealChipType::ORIGIN_72,
        std::string task_name = "QPanda Experiment");

    //pec
    std::vector<double> pec_error_mitigation(
        QProg& prog,
        int shot,
        std::vector<std::string> expectations,
        RealChipType chip_id = RealChipType::ORIGIN_72,
        std::string task_name = "QPanda Experiment");

    //read out
    std::map<std::string, double> read_out_error_mitigation(
        QProg& prog,
        int shot,
        std::vector<std::string> expectations,
        RealChipType chip_id = RealChipType::ORIGIN_72,
        std::string task_name = "QPanda Experiment");

    /**
    * @brief  get real chip qst matrix
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int&   shot
    * @param[out] QStat matrix
    * @return     matrix
    */
    std::vector<QStat> get_state_tomography_density(
        QProg& prog,
        int shot,
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    /**
    * @brief  get real chip qst fidelity
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  int& shot
    * @param[out] QStat matrix
    * @return     matrix
    */
    double get_state_fidelity(
        QProg& prog,
        int shot,
        RealChipType chip_id = RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

    /**
    * @brief  get expectation
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  QHamiltonian& hamiltonian
    * @param[in]  QVec& qubits
    * @return     expectation
    */
    double get_expectation(
        QProg& prog, 
        const QHamiltonian& hamiltonian,
        const QVec& qubits,
        std::string task_name = "QPanda Experiment");

    double estimate_price(size_t qubit_num,
        size_t shot,
        size_t qprogCount = 1,
        size_t epoch = 1);

    std::vector<std::map<std::string, double>> batch_full_amplitude_measure(
        std::vector<QProg>& prog_vector,
        int shot,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, double>> batch_full_amplitude_pmeasure(
        std::vector<QProg>& prog_vector,
        Qnum qubits,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, qcomplex_t>> batch_partial_amplitude_pmeasure(
        std::vector<QProg>& prog_vector,
        std::vector<std::string> amplitudes,
        std::string task_name = "QPanda Experiment");

    std::vector<qcomplex_t> batch_single_amplitude_pmeasure(
        std::vector<QProg>& prog_vector,
        std::string amplitudes,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, double>> batch_noise_measure(
        std::vector<QProg>& prog_vector,
        int shot,
        std::string task_name = "QPanda Experiment");

    std::vector<std::map<std::string, double>> batch_real_chip_measure(
        std::vector<QProg>& prog_vector,
        int shot,
        RealChipType chip_id = RealChipType::ORIGIN_72,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        std::string task_name = "QPanda Experiment");

private:

    NoiseConfigs m_noisy_args;
    std::shared_ptr<QCloudMachineImp> m_cloud_imp;

};

QPANDA_END

#endif
