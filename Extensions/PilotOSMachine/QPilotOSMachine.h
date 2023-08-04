#ifndef QPILOTOS_MACHINE_H
#define QPILOTOS_MACHINE_H

#include "QPandaConfig.h"

#include "Core/QuantumNoise/NoiseModelV2.h"
#include "OSDef.h"
#include "Core/QuantumCloud/QCloudMachineImp.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "data_struct.h"
#include "JsonParser.h"

namespace PilotQVM {
    struct PilotNoiseParams;
    struct PilotTaskQueryResult;
    class QPilotMachine;
}

QPANDA_BEGIN

/*
 * @class QPilotOSMachine
 * @brief Quantum Cloud Machine  for connecting  QCloud server
 * @ingroup QuantumMachine
 * @see QuantumMachine
 * @note  QPilotOSMachine also provides  python interface
 */
class QPilotOSMachine :public QVM
{
public:
    QPilotOSMachine(std::string machine_type = "Pilot");
    ~QPilotOSMachine();
    void init();

#if defined(USE_CURL)
    void init(std::string url, bool log_cout, const std::string& api_key);
    void init(std::string url, bool log_cout, const std::string& username, const std::string& pwd);

    double pMeasureBinindex(QProg& prog, std::string index, int backendID = ANY_CLUSTER_BACKEND);

    double pMeasureDecindex(QProg& prog, std::string index, int backendID = ANY_CLUSTER_BACKEND);

    std::unordered_map<std::string, std::complex<double>> pmeasure_subset(QProg& prog, const std::vector<std::string>& amplitude, int backendID = ANY_CLUSTER_BACKEND);

    double real_chip_expectation(const QProg& prog,
        const std::string& hamiltonian,
        const std::vector<uint32_t>& qubits = {},
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "");
    std::string async_real_chip_expectation(const QProg& prog,
        const std::string& hamiltonian,
        const std::vector<uint32_t>& qubits = {},
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "");

    /**< Sync Measure Task
     * Result: Double
     */
    std::map<std::string, double> real_chip_measure(const QProg& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::map<std::string, double> real_chip_measure(const std::string& ir,
        const int shot = 1000, 
        const int chip_id = ANY_QUANTUM_CHIP, 
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::string real_chip_measure(const std::vector<QProg>& prog,
        const std::string& config_str);
    std::vector<std::map<std::string, double>> real_chip_measure_vec(const std::vector<QProg>& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::vector<std::map<std::string, double>> real_chip_measure_vec(const std::vector<std::string>& ir,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);

    /**< Sync Measure Task
     * Result: Size_t
     */
    std::map<std::string, size_t> real_chip_measure_prob_count(const QProg& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::map<std::string, size_t> real_chip_measure_prob_count(const std::string& ir,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::vector<std::map<std::string, size_t>> real_chip_measure_prob_count(const std::vector<QProg>& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::vector<std::map<std::string, size_t>> real_chip_measure_prob_count(const std::vector<std::string>& ir,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);

    /**< Async Measure Task */
    std::string async_real_chip_measure(const QProg& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const bool is_prob_counts = true,
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::string async_real_chip_measure(const std::string& ir,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const bool is_prob_counts = true,
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::string async_real_chip_measure(const std::vector<QProg>& prog,
        const std::string& config_str);
    std::string async_real_chip_measure_vec(const std::vector<QProg>& prog,
        int shot = 1000,
        int chip_id = ANY_QUANTUM_CHIP,
        bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const bool is_prob_counts = true,
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::string async_real_chip_measure_vec(const std::vector<std::string>& ir,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const bool is_prob_counts = true,
        const std::string& task_describe = "",
        const int point_lable = 0);

    /**< QST Task */
    std::string async_real_chip_QST(const QProg& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::string async_real_chip_QST_density(const QProg& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);
    std::string async_real_chip_QST_fidelity(const QProg& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "",
        const int point_lable = 0);

    bool get_measure_result(const std::string& task_id, std::vector<std::map<std::string, double>>& result, PilotQVM::ErrorCode& errCode, std::string& errInfo);
    bool get_measure_result(const std::string& task_id, std::vector<std::map<std::string, uint64_t>>& result, PilotQVM::ErrorCode& errCode, std::string& errInfo);
    bool get_expectation_result(const std::string& task_id, double& result, PilotQVM::ErrorCode& errCode, std::string& errInfo);
    bool get_qst_result(const std::string& task_id, std::vector<std::map<std::string, double>>& result, PilotQVM::ErrorCode& errCode, std::string& errInfo);
    bool get_qst_density_result(const std::string& task_id, std::vector<std::map<std::string, double>>& result, PilotQVM::ErrorCode& errCode, std::string& errInfo);
    bool get_qst_fidelity_result(const std::string& task_id, double& result, PilotQVM::ErrorCode& errCode, std::string& errInfo);

    std::map<std::string, double> probRunDict(QProg& prog, const std::vector<uint32_t>& qubit_vec, int backendID = ANY_CLUSTER_BACKEND);
    std::map<std::string, size_t> runWithConfiguration(QProg& prog, int shots, const uint32_t& backendID, const QPanda::NoiseModel& noise_model = QPanda::NoiseModel());
    std::map<std::string, size_t> runWithConfiguration(QProg&, int, const QPanda::NoiseModel & = QPanda::NoiseModel()) override;

    void set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params);

    /**
     * @brief  run a measure quantum program
     * @param[in]  QProg& the reference to a quantum program
     * @param[in]  int&   shot
     * @param[out] std::map<std::string, double>
     * @return     measure result
     */
    std::map<std::string, double> noise_measure(QProg&, int shot);

    /**
     * @brief run a noise_learning.sh
     */
    std::string noise_learning(const std::string& parameter_json);

    /**
     * @brief Error mitigation compute
     */
    std::vector<double> em_compute(const std::string& parameter_json);
    std::string async_em_compute(const std::string& parameter_json);

    /**
     * @brief  run a measure quantum program
     * @param[in]  QProg& the reference to a quantum program
     * @param[in]  int&   shot
     * @param[out] std::map<std::string, double>
     * @return     measure result
     */
    std::map<std::string, double> full_amplitude_measure(QProg&, int shot);

    /**
     * @brief  run a pmeasure quantum program
     * @param[in]  QProg& the reference to a quantum program
     * @param[in]  Qnum & qubit address vector
     * @param[out] std::map<std::string, double>
     * @return     pmeasure result
     */
    std::map<std::string, double> full_amplitude_pmeasure(QProg& prog, Qnum qubit_vec);

    std::vector<std::map<std::string, double>> full_amplitude_pmeasure(std::vector<QProg>& prog, Qnum qubit_vec);

    /**
     * @brief  run a pmeasure quantum program with partial amplitude backend
     * @param[in]  QProg& the reference to a quantum program
     * @param[in]  std::vector<std::string> & amplitude subset
     * @param[out] std::map<std::string, qcomplex_t>
     * @return     pmeasure result
     */
    std::map<std::string, qcomplex_t> partial_amplitude_pmeasure(QProg& prog, std::vector<std::string> amplitude_vec);

    /**
     * @brief  run a pmeasure quantum program with single amplitude backend
     * @param[in]  QProg& the reference to a quantum program
     * @param[in]  std::string amplitude
     * @param[out] qcomplex_t
     * @return     pmeasure result
     */
    qcomplex_t single_amplitude_pmeasure(QProg& prog, std::string amplitude);

    /**
     * @brief  get real chip qst fidelity
     * @param[in]  QProg& the reference to a quantum program
     * @param[in]  int&   shot
     * @param[out] QStat matrix
     * @return     matrix
     */
    double get_state_fidelity(
        QProg& prog,
        int shot,
        uint32_t chip_id = 72,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true);

    double get_expectation(QProg, const QHamiltonian&, const QVec&) override;
    bool query_task_state(const std::string& task_id, PilotQVM::PilotTaskQueryResult& result);
    bool query_task_state(const std::string& task_id, PilotQVM::PilotTaskQueryResult& result,
        const bool save_to_file, std::string& file_path);
    bool query_compile_prog(const std::string task_id, std::string& compile_prog, bool with_compensate = true);

    bool login_pilot_with_api_key(const std::string& api_key);
    bool login_pilot(const std::string& username, const std::string& pwd);
#endif

    void init_config(std::string& url, bool log_cout);

    std::string OutputVersionInfo();


    PilotQVM::ErrorCode get_token(std::string& rep_json);

    std::string buil_init_msg(std::string& api_key);

    std::string build_measure_task_msg(const std::vector<QProg>& prog, 
        const int shot, 
        const int chip_id,
        const bool is_amend, 
        const bool is_mapping, 
        const bool is_optimization, 
        const std::vector<uint32_t>& specified_block,
        const std::string& task_describe,
        const int point_lable = 0,
        const int priority = 0);
    
    
    /**
     * @brief  Used for building expected task requests for Python API 
     * 
     * @param[in] QProg& the reference to a prog 
     * @param[in] const std::string& Hamiltonian parameters
     * @param[in] std::vector<uint32_t>& measurement qubits
     * @param[in] const int  repeat run quantum program times 
     * @param[in] chip_id the quantum chip ID
     * @param[in] is_amend Whether amend task result
     * @param[in] is_mapping Whether mapping logical Qubit to Physical Qubit
     * @param[in] is_optimization Whether optimize your quantum program
     * @param[in] specified_block your specified Qubit block
     * @param[in] task_describe the detail information to describe your quantum program
     * @return built expectation task msg
     */
    std::string build_expectation_task_msg(const QProg& prog,  
        const std::string& hamiltonian,
        const std::vector<uint32_t>& qubits = {},
        const int shot = 1000, 
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true, 
        const bool is_mapping = true, 
        const bool is_optimization = true, 
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "");

    /**
     * @brief  Used for building qst task requests for Python API 
     * 
     * @param[in] QProg& the reference to a prog 
     * @param[in] const int  repeat run quantum program times 
     * @param[in] const int the quantum chip ID
     * @param[in] const bool Whether amend task result
     * @param[in] const bool Whether mapping logical Qubit to Physical Qubit
     * @param[in] const bool Whether optimize your quantum program
     * @param[in] const std::vector<uint32_t>& your specified Qubit block
     * @param[in] const std::string& the detailed information to describe your quantum program
     * @return built qst task msg
     */
    std::string build_qst_task_msg(const QProg& prog,
        const int shot = 1000,
        const int chip_id = ANY_QUANTUM_CHIP,
        const bool is_amend = true,
        const bool is_mapping = true,
        const bool is_optimization = true,
        const std::vector<uint32_t>& specified_block = {},
        const std::string& task_describe = "");
    
    std::string build_query_msg(const std::string& task_id);

    bool tcp_recv(const std::string& ip, const unsigned short& port,const string& task_id, std::string& resp);

    PilotQVM::ErrorCode parser_probability_result(const std::string& json_msg, std::vector<std::map<std::string, double>>& result);

    /**
     * @brief parse expectation task result, and use param to bring out the result
     * 
     * @param[in] JsonMsg::JsonParser& Json data to be parsed
     * @param[out] std::vector<double>& parsed result
     * @return ErrorCode
     */
    PilotQVM::ErrorCode parser_expectation_result(const std::string& json_msg, std::vector<double>& result);

    
    bool parse_task_result(const std::string& result_str, std::map<std::string, double>& val);
    bool parse_task_result(const std::vector<std::string>& result_str, std::vector<std::map<std::string, double>>& val);
    bool parse_task_result(const std::vector<std::string>& result_str, std::vector<std::map<std::string, uint64_t>>& val);
    bool parse_qst_density(const std::string& result_str, std::vector<std::map<std::string, double>>& val);
    bool parse_qst_fidelity(const std::string& result_str, double& val);


private:
    /* measure result for full amplitude & noise  */
    std::string binary_to_inter(std::string& str);
#ifdef USE_CURL
    void json_string_transfer_encoding(std::string& str);
    void real_chip_task_validation(int shots, QProg& prog);

    void construct_real_chip_task_json(
        rabbit::document& doc,
        std::string prog_str,
        std::string token,
        bool is_amend,
        bool is_mapping,
        bool is_optimization,
        size_t qvm_type,
        size_t qubit_num,
        size_t cbit_num,
        size_t measure_type,
        size_t shot,
        size_t chip_id);

    void construct_cluster_task_json(
        rabbit::document& doc,
        std::string prog_str,
        std::string token,
        size_t qvm_type,
        size_t qubit_num,
        size_t cbit_num,
        size_t measure_type);

    bool parser_result_json(std::string& recv_json, std::string&);
    bool parser_submit_json(std::string& recv_json, std::string&);
#endif

private:
    std::string m_machine_type;
    //PilotQVM::ErrorCode m_err_code;
    QPanda::CPUQVM* m_cpu_machine;
    PilotQVM::QPilotMachine* m_pilot_machine;
    PilotQVM::PilotNoiseParams* m_noise_params;
    PilotQVM::TaskStatus m_task_status = PilotQVM::TaskStatus::RUNNING;

    /* url & token setting */
    std::string m_token;

    /* measure result for full amplitude & noise */
    std::map<std::string, double> m_measure_result;

    /* error message taskid : error msg */
    std::string m_error_info;

    /* pmeasure result */
    std::map<std::string, qcomplex_t> m_pmeasure_result;

    /* qst result */
    std::vector<QStat> m_qst_result;
    double m_qst_fidelity;

    /* expectation result */
    double m_expectation;

    /* single amplitude */
    qcomplex_t m_single_result;

    enum PilotTaskType
    {
        CLUSTER_MEASURE = 1,
        CLUSTER_PMEASURE = 2,
        CLUSTER_EXPECTATION
    };

    enum PilotResultType
    {
        STATE_PROBS = 1,
        SINGLE_AMPLITUDE = 2,
        AMPLITUDE_ARRAY = 3,
        EXPECTATION
    };
};

QPANDA_END

#endif