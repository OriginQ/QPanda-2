#pragma once

#include <ctime>
#include <string>
#include "Core/QuantumCloud/QRabbit.h"

#include "Core/Module/DataStruct.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCloud/QCloudMachineImp.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

#include "QPandaConfig.h"
#include "Core/Utilities/QProgInfo/QProgClockCycle.h"

QPANDA_BEGIN

using std::map;
using std::string;

#define MAX_LAYER_LIMIT 500

class QCloudService :public QVM
{
public:

    QCloudService();
    ~QCloudService();

    void init(std::string user_token, bool is_logged = false);
    void set_qcloud_url(std::string cloud_url);

    std::string build_full_amplitude_measure(int shots);

    void set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params);
    string build_noise_measure(int shots);

    string build_full_amplitude_pmeasure(Qnum qubit_vec);

    string build_partial_amplitude_pmeasure(std::vector<string> amplitudes);

    string build_single_amplitude_pmeasure(std::string amplitude);

    string build_error_mitigation(
        int shots,
        RealChipType chip_id,
        std::vector<string> expectations,
        const std::vector<double>& noise_strength,
        EmMethod qemMethod);

    string build_read_out_mitigation(
        int shots,
        RealChipType chip_id,
        std::vector<string> expectations,
        const std::vector<double>& noise_strength,
        EmMethod qem_method);

    string build_real_chip_measure(
        int shots,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    string build_get_state_tomography_density(
        int shot,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    string build_get_state_fidelity(
        int shot,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization);

    string build_get_expectation(
        const QHamiltonian& hamiltonian,
        const Qnum& qubits);

    string build_real_chip_measure_batch(
        std::vector<QProg>& prog_vector,
        int shots,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization,
        bool enable_compress_check = false,
        std::string batch_id = "",
        int task_from = 4);

    string build_real_chip_measure_batch(
        std::vector<std::string>& prog_vector,
        int shots,
        RealChipType chip_id,
        bool is_amend,
        bool is_mapping,
        bool is_optimization,
        bool enable_compress_check = false,
        std::string batch_id = "",
        int task_from = 4);

    std::string compress_data(std::string submit_string);

    void build_init_object(QProg& prog, 
        std::string task_name = "QPanda Experiment",
        int task_from = 4);

    void build_init_object(std::string& originir, 
        std::string task_name = "QPanda Experiment",
        int task_from = 4);

    void build_init_object_batch(std::vector<QProg>& prog_vector, 
        std::string task_name = "QPanda Experiment",
        int task_from = 4);

    void build_init_object_batch(std::vector<string>& prog_strings, 
        std::string task_name = "QPanda Experiment",
        int task_from = 4);

    std::string parse_get_task_id(std::string recv_json)
    {
        std::string taskid;
        parse_submit_json(taskid, recv_json);
        return taskid;
    }

    std::vector<double> query_prob_vec_result(std::string result_string)
    {
        std::vector<double> result;
        parse_result<std::vector<double>>(result_string, result);
        return result;
    }

    std::map<std::string, qcomplex_t> query_state_dict_result(std::string result_string)
    {
        std::map<std::string, qcomplex_t> result;
        parse_result<std::map<std::string, qcomplex_t>>(result_string, result);
        return result;
    }

    double query_prob_result(std::string result_string)
    {
        double result;
        parse_result<double>(result_string, result);
        return result;
    }

    qcomplex_t query_comolex_result(std::string result_string)
    {
        qcomplex_t result;
        parse_result<qcomplex_t>(result_string, result);
        return result;
    }

    std::vector<QStat> query_qst_result(std::string result_string)
    {
        std::vector<QStat> result;
        parse_result<std::vector<QStat>>(result_string, result);
        return result;
    }

    map<string, double> query_prob_dict_result(std::string result_string)
    {
        map<string, double> result;
        parse_result<map<string, double>>(result_string, result);
        return result;
    }

    std::vector<map<string, double>> query_prob_dict_result_batch(std::vector<string>& result_string_array)
    {
        std::vector<map<string, double>> result_array;

        for (auto i = 0; i < result_string_array.size(); ++i)
        {
            map<string, double> result;
            parse_result<map<string, double>>(result_string_array[i], result);
            result_array.emplace_back(result);
        }

        return result_array;
    }

public:

    void init_pqc_api(std::string url);

    void parse_submit_json(std::string& taskid, const std::string& submit_recv_string);
    void parse_submit_json(map<size_t, std::string>& taskid, const std::string& submit_recv_string);

    void cyclic_query(const std::string& recv_json, bool& is_retry_again, std::string& result_string);
    void batch_cyclic_query(const std::string& recv_json, bool& is_retry_again, std::vector<string>& result_array);

    void object_init(uint32_t qbits_num, uint32_t cbits_num, std::string& prog, std::string& name, int task_form);
    void object_init(uint32_t qbits_num, uint32_t cbits_num, std::vector<string>& prog_array, std::string& name, int task_form);

    template<typename T>
    void object_append(const std::string& key, const T& value)
    {
        m_object.insert(key, value);
    }

    void object_append_chip_args(RealChipType chip_id, 
        bool is_amend, 
        bool is_mapping, 
        bool is_optimization)
    {
        m_object.insert("chipId", (size_t)chip_id);
        m_object.insert("isAmend", is_amend);
        m_object.insert("mappingFlag", is_mapping);
        m_object.insert("circuitOptimization", is_optimization);
        //m_object.insert("compileLevel", compile_level);
        return;
    }

    //using namespace rabbit;
    void object_append_em_args(RealChipType chip_id, 
        std::vector<string> expectations,
        const std::vector<double>& noise_strength,
        rabbit::array& noise_strength_array,
        EmMethod qem_method)
    {
        switch (qem_method)
        {
        case QPanda::ZNE:
        {
            rabbit::array exp_array;
            for (auto i = 0; i < expectations.size(); ++i)
                exp_array.push_back(expectations[i]);

            for (auto i = 0; i < noise_strength.size(); ++i)
                noise_strength_array.push_back(noise_strength[i]);

            m_object.insert("expectations", exp_array.str());
            //m_object.insert("noiseStrength", noise_strength_array);
            break;
        }

        case QPanda::PEC:
        case QPanda::READ_OUT:
        {
            rabbit::array exp_array, noise_strength_array;
            for (auto i = 0; i < expectations.size(); ++i)
                exp_array.push_back(expectations[i]);

            m_object.insert("expectations", exp_array.str());
            break;
        }

        default:
            break;
        }

        m_object.insert("chipId", (size_t)chip_id);
        m_object.insert("isEm", 1);
        m_object.insert("qemMethod", qem_method);
        
        return;
    }

#if defined(USE_QHETU) 

    std::string sm4_encode(std::string_view key, std::string_view IV, std::string_view data);
    std::string sm4_decode(std::string_view key, std::string_view IV, std::string_view enc_data);
    std::vector<std::string> enc_hybrid(std::string_view pk_str, std::string& rdnum);

#endif

public:

    //origin qcloud user token 
    std::string m_user_token;

    //m_measure_qubits_num[0]  -> normal task measure_qubit_num
    //m_measure_qubits_num     -> batch  task measure_qubit_num
    Qnum m_measure_qubits_num;

    std::string m_chip_config_url;

    //cloud url
    std::string m_inquire_url;
    std::string m_compute_url;
    std::string m_estimate_url;

    std::string m_batch_inquire_url;
    std::string m_batch_compute_url;
    std::string m_big_data_batch_compute_url;

    //PQC Cryption
    std::string m_pqc_init_url;
    std::string m_pqc_keys_url;

    std::string m_pqc_inquire_url;
    std::string m_pqc_compute_url;
    std::string m_pqc_batch_inquire_url;
    std::string m_pqc_batch_compute_url;

    bool m_use_compress_data = false;
    std::string m_configuration_header_data;

private:

    NoiseConfigs m_noisy_args;

    //rabbit object for submit task
    rabbit::object m_object;
};

QPANDA_END