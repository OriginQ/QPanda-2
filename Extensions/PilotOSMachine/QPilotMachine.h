#ifndef QPILOT_MACHINE_H
#define QPILOT_MACHINE_H

#include "QPandaConfig.h"

#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include "OSDef.h"
#include <atomic>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Module/DataStruct.h"
#include "data_struct.h"
#include "JsonParser.h"

namespace PilotQVM {
    enum class PauliOpTy : unsigned char
    {
        PAULI_X = 'X',
        PAULI_Y = 'Y',
        PAULI_Z = 'Z',
        PAULI_I = 'I'
    };

    /**
     * @note reference resources QPanda::QHamiltonian
     */
    /*using QuantumHamiltonianData = std::vector< std::pair< std::map<size_t, PauliOpTy>, double > >;*/
    using QuantumHamiltonianData = QPanda::QHamiltonian;

    class QPilotMachineImp;

    struct PilotNoiseParams
    {
        std::string noise_model;
        double single_gate_param; /* T1 */
        double double_gate_param;

        double single_p2; /* T2 */
        double double_p2;

        double single_pgate; /* Coherence time */
        double double_pgate;
    };

    struct PilotTaskQueryResult
    {
        std::string m_taskId;
        std::size_t m_errCode;
        std::string m_errInfo;
        std::string m_state;
        std::string m_result;
        std::string m_qst_density;
        std::string m_qst_fidelity;
        std::vector<std::string>m_result_vec;
        std::string m_resultJson;
    };

    /**
     * @class QPilotMachine
     * @brief Realize asynchronous call of Sinan system interface
     * @note  QPilotMachine also provides python interface
     */
    class QPilotMachine
    {
        ErrorCode m_test_ee{ ErrorCode::NO_ERROR_FOUND };

    public:
        QPilotMachine();
        virtual ~QPilotMachine();

        /**
         * @brief Initialize service address information and local listening port information of Sinan system
         * @param[in] const std::string& Sinan URL address
         * @param[in] const std::string& Local binding address
         * @param[in] const uint32_t& local port
         * @param[in] const uint32_t& Number of worker threads used to execute asynchronous callback functions
         * @return Whether the initialization is successful. The true table is initialized successfully. Otherwise, it returns false
         */
        bool init(const std::string& pilot_url, bool log_cout = false);       
        bool init_withconfig(const std::string& config_path="pilotmachine.conf");
        ErrorCode get_token(std::string& rep_json);

        std::string build_measure_task_msg(const CalcConfig &config);

        /**
         * @brief Used to build pyqpanda expectation task message body
         * @param[in] CalcConfig& expectation task configuration
         * @param[in] std::vector<uint32_t>& specified qubits
         * @return If construction is successful, return config, otherwise return an empty string.
         */
        std::string build_expectation_task_msg(const CalcConfig& config, const std::vector<uint32_t>& qubits);
        std::string build_query_msg(const std::string& task_id);
        ErrorCode parser_probability_result(JsonMsg::JsonParser& json_parser, std::vector<std::map<std::string, double>>& result);

        /**
         * @brief parse expectation task result, and use param(result) to bring out
         * @param[in] JsonMsg::JsonParser& Json data to be parsed
         * @param[out] std::vector<double>& parsed result
         * @return ErrorCode
         */
        ErrorCode parser_expectation_result(JsonMsg::JsonParser& json_parser, std::vector<double>& result);
        bool parse_task_result(const std::string& result_str, std::map<std::string, double>& val);
        bool parse_task_result(const std::vector<std::string>& result_str, std::vector<std::map<std::string, double>>& result_mp);
        bool parse_task_result(const std::vector<std::string>& result_str, std::vector<std::map<std::string, uint64_t>>& result_mp);
        bool parse_qst_density(const std::string& result, std::vector<std::map<std::string, double>>& result_mp_vec);
        bool parse_qst_fidelity(const std::string& result_str, double& result);

        /**
         * @brief Used to build pyqpanda qst task message body
         * @param[in] CalcConfig& expectation task configuration
         * @return If construction is successful, return config, otherwise return an empty string.
         */
        std::string build_qst_task_msg(const CalcConfig& config);

#ifdef USE_CURL

        ErrorCode execute_expectation_task(const CalcConfig& config, const std::vector<uint32_t> &qubits, std::vector<double>& result);
        std::string async_execute_expectation_task(const CalcConfig& config, const std::vector<uint32_t> &qubits, std::vector<double>& result);
        //ErrorCode execute_expectation_task_vec(const CalcConfig& config, const std::vector<uint32_t>& qubits, std::vector<double>& result);
        //std::vector<std::string> async_execute_expectation_task_vec(const CalcConfig& config, const std::vector<uint32_t>& qubits, std::vector<double>& result);

        ErrorCode execute_measure_task(const CalcConfig& config, std::map<std::string, double>& result);
        ErrorCode execute_measure_task_vec(const CalcConfig& config, std::vector<std::map<std::string, double>>& result);
        ErrorCode execute_measure_task_vec(const CalcConfig& config, std::vector<std::map<std::string, size_t>>& result);
        ErrorCode execute_measure_task_vec(const CalcConfig& config, std::string& result);

        //std::string async_execute_measure_task(const CalcConfig &config, std::map<std::string, double> &result);
        std::string async_execute_measure_task(const CalcConfig& config);
        ErrorCode async_execute_measure_task_vec(const CalcConfig& config, std::string& task_id_vec);

        /*
         * @brief execute_noise_learning_task
         */
        ErrorCode execute_noise_learning_task(const std::string& parameter_json, std::string& task_id);

        /**
        * @brief execute_error_mitigation_compute
        */
        ErrorCode async_execute_em_compute_task(const std::string& parameter_json, std::string& task_id);
        ErrorCode execute_em_compute_task(const std::string& parameter_json, std::string& task_id, std::vector<double>& result);

        ErrorCode execute_measure_task(const CalcConfig& config, std::function<void (ErrorCode, const std::map<std::string, double>&)> cb_func);

        /**
         * @brief System interface (asynchronous execution): perform full amplitude analog quantum computing tasks
         * @param[in] const std::string& Quantum program, originir representation
         * @param[out] std::map<std::string, double>& Calculation results
         * @param[in] const uint32_t& Specify backend information, default \ P any_ CLUSTER_ Backend, indicating that the system automatically allocates the computing backend
         * @param[in] const uint32_t& Operation times
         * @return ErrorCode reference resources PilotQVM::ErrorCode
         */
        ErrorCode execute_full_amplitude_measure_task(const std::string& prog_str,
            std::map<std::string, double>& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND,
            const uint32_t& shots = 1000);

        ErrorCode execute_full_amplitude_measure_task(const std::string& prog_str,
            std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND,
            const uint32_t& shots = 1000);

        /**
         * @brief System interface (asynchronous execution): perform full amplitude pmeasure simulation and quantum computation tasks
         * @param[in] const std::string& Quantum program, originir representation
         * @param[in] const std::vector<uint32_t>& Target measurement qubit
         * @param[out] std::map<std::string, double>& Calculation results
         * @param[in] const uint32_t& Specify back-end information, default any_ CLUSTER_ Backend, indicating that the system automatically allocates the computing backend
         * @return ErrorCode reference resources PilotQVM::ErrorCode
         */
        ErrorCode execute_full_amplitude_pmeasure_task(const std::string& prog_str,
            const std::vector<uint32_t>& qubit_vec,
            std::map<std::string, double>& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        ErrorCode execute_full_amplitude_pmeasure_task(const std::string& prog_str,
            const std::vector<uint32_t>& qubit_vec,
            std::function<void(ErrorCode, const std::map<std::string, double>&)> cb_func,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        /**
         * @brief Building noise model
         * @param[in] const uint32_t& Noise model type, reference QPanda::NOISE_MODEL
         * @param[in] const std::vector<double>& Single door noise parameters
         * @param[in] const std::vector<double>& Double door noise parameters
         * @param[out] PilotNoiseParams& Target noise model
         * @return bool Is the build successful
         */
        static bool build_noise_params(const uint32_t& nose_model_type, const std::vector<double>& single_params, const std::vector<double>& double_params,PilotNoiseParams& noise_params);

        /**
         * @brief System interface (asynchronous execution): perform quantum computation tasks including noise measurement simulation computation
         * @param[in] const std::string& Quantum program, originir representation
         * @param[in] const PilotNoiseParams& Noise parameters, refer to interface build_ noise_ params()
         * @param[out] std::map<std::string, double>& Calculation results
         * @param[in] const uint32_t& Specify back-end information, default any_ CLUSTER_ Backend, indicating that the system automatically allocates the computing backend
         * @param[in] const uint32_t& Specify the number of measurements
         * @return ErrorCode reference resources PilotQVM::ErrorCode
         */
        ErrorCode execute_noise_measure_task(const std::string& prog_str,
            const PilotNoiseParams& noise_params,
            std::map<std::string, double>& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND,
            const uint32_t& shots = 1000);

        ErrorCode execute_noise_measure_task(const std::string& prog_str,
            const PilotNoiseParams& noise_params,
            std::function<void(ErrorCode, std::map<std::string, double>&)> cb_func,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND,
            const uint32_t& shots = 1000);

        /**
         * @brief System interface (asynchronous execution): perform partial amplitude simulation and quantum computation tasks
         * @param[in] const std::string& Quantum program, originir representation
         * @param[in] const std::vector<std::string>& Target amplitude
         * @param[out] std::map<std::string, Complex_>& Calculation results
         * @param[in] const uint32_t& Specify back-end information, default any_ CLUSTER_ Backend, indicating that the system automatically allocates the computing backend
         * @return ErrorCode reference resources PilotQVM::ErrorCode
         */
        ErrorCode execute_partial_amplitude_task(const std::string& prog_str,
            const std::vector<std::string>& target_amplitude_vec,
            std::map<std::string, Complex_>& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        ErrorCode execute_partial_amplitude_task(const std::string& prog_str,
            const std::vector<std::string>& target_amplitude_vec,
            std::function<void(ErrorCode, const std::map<std::string, Complex_>&)> cb_func,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        /**
         * @brief System interface (asynchronous execution): perform single amplitude analog computation and quantum computation tasks
         * @param[in] const std::string& Quantum program, originir representation
         * @param[in] const std::string& Target amplitude
         * @param[out] Complex_& Calculation results
         * @param[in] const uint32_t& Specify back-end information, default any_ CLUSTER_ Backend, indicating that the system automatically allocates the computing backend
         * @return ErrorCode reference resources PilotQVM::ErrorCode
         */
        ErrorCode execute_single_amplitude_task(const std::string& prog_str,
            const std::string& target_amplitude,
            Complex_& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        ErrorCode execute_single_amplitude_task(const std::string& prog_str,
            const std::string& target_amplitude,
            std::function<void(ErrorCode, const Complex_&)> cb_func,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        /**
         * @brief System interface (asynchronous execution): find the expected simulation and quantum computing tasks
         * @param[in] const std::string& Quantum program, originir representation
         * @param[in] const QuantumHamiltonianData& Hamiltonian
         * @param[in] const std::vector<uint32_t>& Target qubit
         * @param[out] double& Calculation results
         * @param[in] const uint32_t& Specify back-end information, default any_ CLUSTER_ Backend, indicating that the system automatically allocates the computing backend
         * @return ErrorCode reference resources PilotQVM::ErrorCode
         */
        ErrorCode execute_full_amplitude_expectation(const std::string& prog_str,
            const QuantumHamiltonianData& hamiltonian,
            const std::vector<uint32_t>& qubit_vec,
            double& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        ErrorCode execute_full_amplitude_expectation(const std::string& prog_str,
            const QuantumHamiltonianData& hamiltonian,
            const std::vector<uint32_t>& qubit_vec,
            std::function<void(ErrorCode, double)> cb_func,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        ErrorCode execute_full_amplitude_expectation(const std::string& prog_str,
            const QuantumHamiltonianData& hamiltonian,
            const std::vector<uint32_t>& qubit_vec,
            const uint32_t& shots,
            double& result,
            const uint32_t& cluster_id = ANY_CLUSTER_BACKEND);

        bool execute_query_task_state(const std::string& task_id, PilotTaskQueryResult& result);
        bool execute_query_compile_prog(const std::string task_id, std::string& compile_prog, bool& with_compensate);

        ErrorCode execute_login_pilot(const std::string&username, const std::string& pwd);
        ErrorCode execute_login_pilot_api(const std::string&api_key);
#endif
    protected:
        std::unique_ptr<QPilotMachineImp> m_imp_obj;

    private:
        void _abort(void);
        static void abort(int _signal);

        std::string m_pilot_url;
        bool m_log_cout = { false };
    };
}



#endif // ! QPILOT_MACHINE_H
