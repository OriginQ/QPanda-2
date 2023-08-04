#include <fstream>
#include <algorithm>
#include "Core/Core.h"
#include "Core/QuantumCloud/QCloudMachine.h"

using namespace std;
USING_QPANDA

#if defined(USE_CURL)

void static real_chip_task_validation(int shots, QProg& prog)
{
    QPANDA_ASSERT(shots > 30000 || shots < 1, "real chip shots must be in range [1,100000]");

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    if (!traver_param.m_can_optimize_measure)
        throw std::invalid_argument("measure operation must be last");

    return;
}

static std::map<NOISE_MODEL, std::string> noise_model_mapping =
{
  {NOISE_MODEL::BITFLIP_KRAUS_OPERATOR,"BITFLIP_KRAUS_OPERATOR"},
  {NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR,"BIT_PHASE_FLIP_OPRATOR"},
  {NOISE_MODEL::DAMPING_KRAUS_OPERATOR,"DAMPING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR,"DECOHERENCE_KRAUS_OPERATOR"},
  {NOISE_MODEL::DEPHASING_KRAUS_OPERATOR,"DEPHASING_KRAUS_OPERATOR"},
  {NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR,"DEPOLARIZING_KRAUS_OPERATOR"},
  //{NOISE_MODEL::KRAUS_MATRIX_OPRATOR,"KRAUS_MATRIX_OPRATOR"},
  //{NOISE_MODEL::MIXED_UNITARY_OPRATOR,"MIXED_UNITARY_OPRATOR"},
  //{NOISE_MODEL::PAULI_KRAUS_MAP,"PAULI_KRAUS_MAP"},
  {NOISE_MODEL::PHASE_DAMPING_OPRATOR,"PHASE_DAMPING_OPRATOR"}
};



QCloudMachine::QCloudMachine()
{
    m_cloud_imp = std::make_shared<QCloudMachineImp>();
}

QCloudMachine::~QCloudMachine()
{}

void QCloudMachine::set_qcloud_url(std::string url)
{
    m_cloud_imp->set_qcloud_url(url);
    return;
}

void QCloudMachine::init(std::string user_token, bool is_logged, 
    bool use_bin_or_hex_format,
    bool enable_pqc_encryption,
    std::string random_num)
{
    QVM::init();
    _start();
    _QMachine_type = QMachineType::QCloud;
    m_cloud_imp->init(user_token, 
                    is_logged, 
                    use_bin_or_hex_format,
                    enable_pqc_encryption,
                    random_num);

    return;
}

void QCloudMachine::set_noise_model(NOISE_MODEL model, const std::vector<double> single_params, const std::vector<double> double_params)
{
    auto iter = noise_model_mapping.find(model);
    if (noise_model_mapping.end() == iter || single_params.empty() || double_params.empty())
        QCERR_AND_THROW(run_fail, "NOISE MODEL ERROR");

    m_noisy_args.noise_model = iter->second;
    m_noisy_args.single_gate_param = single_params[0];
    m_noisy_args.double_gate_param = double_params[0];

    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR == iter->first)
    {
        if (3 != single_params.size())
            QCERR_AND_THROW(run_fail, "DECOHERENCE_KRAUS_OPERATOR PARAM SIZE ERROR");

        if (3 != double_params.size())
            QCERR_AND_THROW(run_fail, "DECOHERENCE_KRAUS_OPERATOR PARAM SIZE ERROR");

        m_noisy_args.single_p2 = single_params[1];
        m_noisy_args.double_p2 = double_params[1];

        m_noisy_args.single_pgate = single_params[2];
        m_noisy_args.double_pgate = double_params[2];
    }

    return;
}

std::map<std::string, double> QCloudMachine::real_chip_measure(
    QProg& prog,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    if (m_cloud_imp->is_enable_pqc_encryption())
    {
        std::vector<QProg> prog_array = { prog };
        auto batch_result =  batch_real_chip_measure(prog_array, shots, chip_id, is_amend, is_mapping, is_optimization, task_name);

        return batch_result.empty() ? std::map<std::string, double>() : batch_result[0];

    }

    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    try
    {
        std::map<std::string, double> result;
        m_cloud_imp->execute_real_chip_measure(result, 
            shots, 
            chip_id, 
            is_amend, 
            is_mapping, 
            is_optimization);

        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

double QCloudMachine::get_state_fidelity(
    QProg& prog,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    try
    {
        double result;
        m_cloud_imp->execute_get_state_fidelity(result, 
            shots, 
            chip_id, 
            is_amend, 
            is_mapping, 
            is_optimization);

        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}


std::vector<QStat> QCloudMachine::get_state_tomography_density(
    QProg& prog,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    try
    {
        std::vector<QStat> result;
        m_cloud_imp->execute_get_state_tomography_density(result, 
            shots, 
            chip_id, 
            is_amend, 
            is_mapping, 
            is_optimization);

        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}


std::map<std::string, double> QCloudMachine::noise_measure(QProg& prog, int shots, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    try
    {
        std::map<std::string, double> result;
        m_cloud_imp->execute_noise_measure(result, shots, m_noisy_args);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_measure(QProg& prog, int shots, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);
   
    try
    {
        std::map<std::string, double> result;
        m_cloud_imp->execute_full_amplitude_measure(result, shots);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::map<std::string, double> QCloudMachine::full_amplitude_pmeasure(QProg& prog, Qnum qubit_vec, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    try
    {
        std::map<std::string, double> result;
        m_cloud_imp->execute_full_amplitude_pmeasure(result, qubit_vec);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::map<std::string, qcomplex_t> QCloudMachine::partial_amplitude_pmeasure(QProg& prog, std::vector<std::string> amplitudes, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    for_each(amplitudes.begin(), amplitudes.end(), [&](const std::string& amplitude)
    {
        uint128_t max_amplitude = (uint128_t("1") << getAllocateQubitNum()) - 1;
        if (max_amplitude < uint128_t(amplitude.c_str()))
            QCERR_AND_THROW(run_fail, "amplitude params > max_amplitude");
    });

    try
    {
        std::map<std::string, qcomplex_t> result;
        m_cloud_imp->execute_partial_amplitude_pmeasure(result, amplitudes);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

qcomplex_t QCloudMachine::single_amplitude_pmeasure(QProg& prog, std::string amplitude, string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    uint128_t max_amplitude = (uint128_t("1") << getAllocateQubitNum()) - 1;
    if (max_amplitude < uint128_t(amplitude.c_str()))
        QCERR_AND_THROW(run_fail, "amplitude params > max_amplitude");

    try
    {
        qcomplex_t result;
        m_cloud_imp->execute_single_amplitude_pmeasure(result, amplitude);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

double QCloudMachine::get_expectation(QProg& prog, const QHamiltonian& hamiltonian, const QVec& qvec, std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    Qnum qubits;
    for (auto qubit : qvec)
        qubits.emplace_back(qubit->get_phy_addr());

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);


    try
    {
        double result;
        m_cloud_imp->execute_get_expectation(result, hamiltonian, qubits);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}


std::vector<std::map<std::string, double>> QCloudMachine::batch_full_amplitude_measure(std::vector<QProg>& prog_array, int shots, std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<std::map<std::string, double>> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), task_name);
        m_cloud_imp->execute_full_amplitude_measure_batch(result, originir_array, shots);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, double>> QCloudMachine::batch_full_amplitude_pmeasure(std::vector<QProg>& prog_array, Qnum qubits, std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<std::map<std::string, double>> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), task_name);
        m_cloud_imp->execute_full_amplitude_pmeasure_batch(result, originir_array, qubits);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, qcomplex_t>> QCloudMachine::batch_partial_amplitude_pmeasure(
    std::vector<QProg>& prog_array,
    std::vector<std::string> amplitudes,
    std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<std::map<std::string, qcomplex_t>> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), task_name);
        m_cloud_imp->execute_partial_amplitude_pmeasure_batch(result, originir_array, amplitudes);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<qcomplex_t> QCloudMachine::batch_single_amplitude_pmeasure(
    std::vector<QProg>& prog_array,
    std::string amplitude,
    std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<qcomplex_t> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), task_name);
        m_cloud_imp->execute_single_amplitude_pmeasure_batch(result, originir_array, amplitude);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, double>> QCloudMachine::batch_noise_measure(
    std::vector<QProg>& prog_array,
    int shots,
    std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<std::map<std::string, double>> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), task_name);
        m_cloud_imp->execute_noise_measure_batch(result, originir_array, shots, m_noisy_args);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}


std::vector<double> QCloudMachine::pec_error_mitigation(
    QProg& prog,
    int shots,
    std::vector<std::string> expectations,
    RealChipType chip_id,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    QVec qubits;
    std::vector<ClassicalCondition> cbit_vector;
    auto qubits_num = prog.get_used_qubits(qubits);
    auto cbits_num = prog.get_used_cbits(cbit_vector);

    m_cloud_imp->object_init(qubits_num, cbits_num, prog_str, task_name);

    try
    {
        std::vector<double> result;
        m_cloud_imp->execute_error_mitigation(result, shots, chip_id, expectations, {}, EmMethod::PEC);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

//read out
std::map<std::string, double> QCloudMachine::read_out_error_mitigation(
    QProg& prog,
    int shots,
    std::vector<std::string> expectations,
    RealChipType chip_id,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    QVec qubits;
    std::vector<ClassicalCondition> cbit_vector;
    auto qubits_num = prog.get_used_qubits(qubits);
    auto cbits_num = prog.get_used_cbits(cbit_vector);

    m_cloud_imp->object_init(qubits_num, cbits_num, prog_str, task_name);

    try
    {
        std::map<std::string, double> result;
        m_cloud_imp->read_out_error_mitigation(result, shots, chip_id, expectations, {}, EmMethod::READ_OUT);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<double> QCloudMachine::zne_error_mitigation(
    QProg& prog,
    int shots,
    std::vector<std::string> expectations,
    std::vector<double> noise_strength,
    RealChipType chip_id,
    std::string task_name)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    QVec qubits;
    std::vector<ClassicalCondition> cbit_vector;
    auto qubits_num = prog.get_used_qubits(qubits);
    auto cbits_num = prog.get_used_cbits(cbit_vector);

    m_cloud_imp->object_init(qubits_num, cbits_num, prog_str, task_name);

    try
    {
        std::vector<double> result;
        m_cloud_imp->execute_error_mitigation(result, shots, chip_id, expectations, noise_strength, EmMethod::ZNE);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, double>> QCloudMachine::batch_real_chip_measure(
    std::vector<QProg>& prog_array,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<std::map<std::string, double>> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), task_name);
        m_cloud_imp->execute_real_chip_measure_batch(result, 
            originir_array, 
            shots, 
            chip_id, 
            is_amend, 
            is_mapping, 
            is_optimization);

        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::string QCloudMachine::async_noise_measure(
        QProg& prog,
        int shots,
        std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    m_cloud_imp->object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    m_cloud_imp->object_append("QMachineType", (size_t)CloudQMchineType::NOISE_QMACHINE);

    m_cloud_imp->object_append("shot", (size_t)shots);
    m_cloud_imp->object_append("noisemodel", m_noisy_args.noise_model);
    m_cloud_imp->object_append("singleGate", m_noisy_args.single_gate_param);
    m_cloud_imp->object_append("doubleGate", m_noisy_args.double_gate_param);

    if ("DECOHERENCE_KRAUS_OPERATOR" == m_noisy_args.noise_model)
    {
        m_cloud_imp->object_append("singleP2", m_noisy_args.single_p2);
        m_cloud_imp->object_append("doubleP2", m_noisy_args.double_p2);
        m_cloud_imp->object_append("singlePgate", m_noisy_args.single_pgate);
        m_cloud_imp->object_append("doublePgate", m_noisy_args.double_pgate);
    }

    return m_cloud_imp->submit(m_cloud_imp->object_string());
}

std::string QCloudMachine::async_full_amplitude_measure(
    QProg& prog,
    int shot,
    std::string task_name)
{
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    m_cloud_imp->object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    m_cloud_imp->object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);
    m_cloud_imp->object_append("shot", (size_t)shot);

    return m_cloud_imp->submit(m_cloud_imp->object_string());
}

std::string QCloudMachine::async_full_amplitude_pmeasure(
    QProg& prog,
    Qnum qubit_vec,
    std::string task_name)
{
    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    m_cloud_imp->object_append("measureType", (size_t)ClusterTaskType::CLUSTER_PMEASURE);
    m_cloud_imp->object_append("QMachineType", (size_t)CloudQMchineType::Full_AMPLITUDE);

    std::string string_array;
    for (auto val : qubit_vec)
    {
        string_array.append(to_string(val));
        if (val != qubit_vec.back())
            string_array.append(",");
    }

    m_cloud_imp->object_append("qubits", string_array);

    return m_cloud_imp->submit(m_cloud_imp->object_string());
}

std::string QCloudMachine::async_real_chip_measure(
    QProg& prog,
    int shots,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    if (m_cloud_imp->is_enable_pqc_encryption())
    {
        std::vector<QProg> prog_array = { prog };
        return async_batch_real_chip_measure(prog_array, shots, chip_id, is_amend, is_mapping, is_optimization, task_name);
    }

    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    m_cloud_imp->object_append_chip_args(chip_id,
        is_amend,
        is_mapping,
        is_optimization);

    m_cloud_imp->object_append("measureType", (size_t)ClusterTaskType::CLUSTER_MEASURE);
    m_cloud_imp->object_append("QMachineType", (size_t)CloudQMchineType::REAL_CHIP);
    m_cloud_imp->object_append("shot", (size_t)shots);

    return m_cloud_imp->submit(m_cloud_imp->object_string());
}

std::string QCloudMachine::async_batch_real_chip_measure(
    std::vector<QProg>& prog_vector,
    int shot,
    RealChipType chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_vector)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), task_name);
    return m_cloud_imp->async_execute_real_chip_measure_batch(originir_array, 
        shot, 
        chip_id, 
        is_amend, 
        is_mapping, 
        is_optimization);
}

std::map<std::string, double> QCloudMachine::query_state_result(std::string task_id)
{
    return m_cloud_imp->query_state_result(task_id);
}


std::vector<std::map<std::string, double>> QCloudMachine::query_batch_state_result(std::string task_id, bool open_loop)
{
    return m_cloud_imp->query_batch_state_result(task_id, open_loop);
}

double QCloudMachine::estimate_price(size_t qubit_num,
    size_t shot,
    size_t qprogCount,
    size_t epoch)
{
    return m_cloud_imp->estimate_price(qubit_num, shot, qprogCount, epoch);
}

REGISTER_QUANTUM_MACHINE(QCloudMachine);

#endif