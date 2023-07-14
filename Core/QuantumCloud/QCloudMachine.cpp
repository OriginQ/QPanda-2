#include <fstream>
#include <algorithm>
#include "Core/Core.h"
#include "Core/QuantumCloud/QCloudMachine.h"

USING_QPANDA
using namespace std;

#if defined(USE_OPENSSL) && defined(USE_CURL)

void static real_chip_task_validation(int shots, QProg& prog)
{
    QPANDA_ASSERT(shots > 100000 || shots < 1, "real chip shots must be in range [1,100000]");

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

void QCloudMachine::set_qcloud_api(std::string url)
{
    m_cloud_imp->set_qcloud_api(url);
    return;
}

void QCloudMachine::init(string user_token, bool is_logged)
{
    _start();
    m_cloud_imp->init(user_token, is_logged);
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
    real_chip_task_validation(shots, prog);

    //convert prog to originir 
    auto prog_str = convert_qprog_to_originir(prog, this);

    m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), prog_str, task_name);

    try
    {
        std::map<std::string, double> result;
        m_cloud_imp->execute_real_chip_measure(result, shots, chip_id, is_amend, is_mapping, is_optimization);
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
        m_cloud_imp->execute_get_state_fidelity(result, shots, chip_id, is_amend, is_mapping, is_optimization);
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
        m_cloud_imp->execute_get_state_tomography_density(result, shots, chip_id, is_amend, is_mapping, is_optimization);
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


std::vector<std::map<std::string, double>> QCloudMachine::full_amplitude_measure_batch(std::vector<QProg>& prog_array, int shots, std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<std::map<std::string, double>> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), originir_array, task_name);
        m_cloud_imp->execute_full_amplitude_measure_batch(result, originir_array, shots);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, double>> QCloudMachine::full_amplitude_pmeasure_batch(std::vector<QProg>& prog_array, Qnum qubits, std::string task_name)
{
    //convert prog to originir
    std::vector<string> originir_array;
    for (auto& val : prog_array)
        originir_array.push_back(convert_qprog_to_originir(val, this));

    try
    {
        std::vector<std::map<std::string, double>> result;

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), originir_array, task_name);
        m_cloud_imp->execute_full_amplitude_pmeasure_batch(result, originir_array, qubits);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, qcomplex_t>> QCloudMachine::partial_amplitude_pmeasure_batch(
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

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), originir_array, task_name);
        m_cloud_imp->execute_partial_amplitude_pmeasure_batch(result, originir_array, amplitudes);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<qcomplex_t> QCloudMachine::single_amplitude_pmeasure_batch(
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

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), originir_array, task_name);
        m_cloud_imp->execute_single_amplitude_pmeasure_batch(result, originir_array, amplitude);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, double>> QCloudMachine::noise_measure_batch(
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

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), originir_array, task_name);
        m_cloud_imp->execute_noise_measure_batch(result, originir_array, shots, m_noisy_args);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

std::vector<std::map<std::string, double>> QCloudMachine::real_chip_measure_batch(
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

        m_cloud_imp->object_init(getAllocateQubitNum(), getAllocateCMem(), originir_array, task_name);
        m_cloud_imp->execute_real_chip_measure_batch(result, originir_array, shots, chip_id, is_amend, is_mapping, is_optimization);
        return result;
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

REGISTER_QUANTUM_MACHINE(QCloudMachine);

#endif