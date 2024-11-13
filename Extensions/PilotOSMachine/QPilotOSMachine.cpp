#include "QPandaConfig.h"


#include "QPilotOSMachine.h"
//#include "OSDef.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"
#include "Components/Operator/PauliOperator.h"
#include "QPilotMachine.h"
#include "ELog.h"
#include "JsonParser.h"
#include "JsonBuilder.h"
#include "TCPClient.h"

using namespace QPanda;
using namespace PilotQVM;

enum class QPilotQMchineType : uint32_t
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

static std::string g_version = "Version:2.2.0";

QPilotOSMachine::QPilotOSMachine(std::string machine_type)
    :m_noise_params(nullptr), m_pilot_machine(nullptr), m_cpu_machine(nullptr)
{
    if (machine_type == "Pilot"){
        m_machine_type = "Pilot";
    }
    else{
        m_machine_type = "CPU";
    }
}

QPilotOSMachine::~QPilotOSMachine()
{
    if (m_noise_params != nullptr) {
        delete m_noise_params, m_noise_params=nullptr;
    }

    if (m_pilot_machine != nullptr) {
        delete m_pilot_machine, m_pilot_machine=nullptr;
    }

    if (m_cpu_machine != nullptr)
    {
        delete m_cpu_machine, m_cpu_machine=nullptr;
    }
}

void QPilotOSMachine::init()
{
    PTraceInfo("***************** On QPilotOSMachine::init for mode: " << m_machine_type << " *****************\n"
        << g_version << "\n");

    if (m_machine_type == "Pilot")
    {
        PTraceError("Error: faile to init QPilotOSMachine for Pilot model, no available parameters.");
        return;
    }

    m_cpu_machine = new(std::nothrow) QPanda::CPUQVM;
    m_cpu_machine->init();
    _start();
}

std::string QPilotOSMachine::OutputVersionInfo()
{
    return g_version;
}

std::string QPilotOSMachine::binary_to_inter(std::string& str)
{
    size_t m_value = 0;
    size_t m_str_size = str.size() - 1;
    for (size_t i = 0; i <= m_str_size; i++)
    {
        switch (str[m_str_size - i])
        {
        case '0':
            break;
        case '1':
            m_value += size_t(std::pow(2, i));
            break;
        default:
            throw std::runtime_error("params error:" + str);
            break;
        }
    }

    return std::to_string(m_value);
}

void QPilotOSMachine::init_config(std::string& url, bool log_cout)
{
    PilotQVM::ELog::get_instance().set_output_log(log_cout);
    PTraceInfo("***************** On QPilotOSMachine::init for mode: " << m_machine_type << " *****************\n"
        << g_version << "\n");
    try
    {
        if (m_machine_type == "Pilot")
        {
            std::fstream pilot_config("/etc/statetab.d/sysinfo");
            if (pilot_config.is_open())
            {
                pilot_config >> url;
                pilot_config.close();
                if (url.empty()) {
                    PTraceWarn("Warn: No Permission with Pilot, try run local.");
                }
            }
        }
        PTraceInfo("Pilot url: " << url);
        m_machine_type = m_machine_type == "Pilot" && !url.empty() ? "Pilot" : "CPU";
        if (m_machine_type == "Pilot")
        {
            m_pilot_machine = new(std::nothrow) PilotQVM::QPilotMachine;
            m_noise_params = new(std::nothrow) PilotQVM::PilotNoiseParams;
            auto _r = m_pilot_machine->init(url, log_cout);
            PTraceInfo("m_pilot_machine init return " << _r);
        }
        else
        {
            m_cpu_machine = new(std::nothrow) QPanda::CPUQVM;
            m_cpu_machine->init();
        }
        _start();
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }

    PTraceInfo("***************** on QPilotOSMachine::init ok ****************");
}

ErrorCode QPilotOSMachine::get_token(std::string& rep_json)
{
    return m_pilot_machine->get_token(rep_json);
}

std::string QPilotOSMachine::buil_init_msg(std::string& api_key)
{
    JsonMsg::JsonBuilder jb;
    jb.add_member("apiKey", api_key);
    return jb.get_json_str();
}

std::string QPilotOSMachine::build_measure_task_msg(const std::vector<QProg>& prog, const int shot, const int chip_id,
    const bool is_amend, const bool is_mapping, const bool is_optimization, const std::vector<uint32_t>& specified_block,
    const std::string& task_describe, const int point_lable, const int priority)
{
    try
    {
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            for (uint32_t i = 0; i < prog.size(); i++)
            {
                config.ir_vec.push_back("");
                config.ir_vec[i] = convert_qprog_to_originir(const_cast<QProg&>(prog[i]), this);
            }
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            config.priority = priority;
            return m_pilot_machine->build_measure_task_msg(config);
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::build_expectation_task_msg(const QProg& prog, const std::string& hamiltonian,
    const std::vector<uint32_t>& qubits, const int shot, const int chip_id, const bool is_amend, 
    const bool is_mapping, const bool is_optimization,  const std::vector<uint32_t>& specified_block, 
    const std::string& task_describe)
{
    CalcConfig config;
    try
    {
        if (m_machine_type == "Pilot") 
        {
            config.ir = convert_qprog_to_originir(const_cast<QProg&>(prog), this);
            config.hamiltonian = hamiltonian;
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = is_amend;
            config.task_type = static_cast<int>(TaskType::REAL_EXPECTATION);
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
    return m_pilot_machine->build_expectation_task_msg(config, qubits);
}


std::string QPilotOSMachine::build_qst_task_msg(const QProg& prog, const int shot, const int chip_id, 
        const bool is_amend, const bool is_mapping, const bool is_optimization, 
        const std::vector<uint32_t>& specified_block,const std::string& task_describe)
{
    CalcConfig config;
    try
    {
        if (m_machine_type == "Pilot") 
        {
            config.ir = convert_qprog_to_originir(const_cast<QProg&>(prog), this);
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = is_amend;
            config.task_type = static_cast<int>(TaskType::QST_TASK);
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
    return m_pilot_machine->build_qst_task_msg(config);
}

std::string QPilotOSMachine::build_query_msg(const std::string& task_id)
{
    return m_pilot_machine->build_query_msg(task_id);
}

bool QPilotOSMachine::tcp_recv(const std::string& ip, const unsigned short& port, const string& task_id, std::string& resp)
{
    TCPClient tcp_client;

    tcp_client.init(ip.c_str(), port + 1, task_id);
    tcp_client.send_data(task_id, TCPMsg::TcpMsgType::TASK_ID_MSG);
    tcp_client.run_heart_thread();

    const bool b_recv_result_ok = tcp_client.wait_recv_task_result(resp, task_id);
    tcp_client.stop_heart_thread();
    return b_recv_result_ok;
}

PilotQVM::ErrorCode QPilotOSMachine::parser_probability_result(const std::string& json_msg, std::vector<std::map<std::string, double>>& result)
{
    JsonMsg::JsonParser jp;
    jp.load_json(json_msg);
    return m_pilot_machine->parser_probability_result(jp, result);
}

PilotQVM::ErrorCode QPilotOSMachine::parser_expectation_result(const std::string& json_msg, std::vector<double>& result)
{
    JsonMsg::JsonParser jp;
    jp.load_json(json_msg);
    return m_pilot_machine->parser_expectation_result(jp, result);
}

bool QPilotOSMachine::parse_task_result(const std::string& result_str, std::map<std::string, double>& val)
{
    return m_pilot_machine->parse_task_result(result_str, val);
}

bool QPilotOSMachine::parse_task_result(const std::vector<std::string>& result_str, std::vector<std::map<std::string, double>>& val)
{
    return m_pilot_machine->parse_task_result(result_str, val);
}

bool QPilotOSMachine::parse_task_result(const std::vector<std::string>& result_str, std::vector<std::map<std::string, uint64_t>>& val)
{
    return m_pilot_machine->parse_task_result(result_str, val);
}

bool QPilotOSMachine::parse_qst_density(const std::string& result_str, std::vector<std::map<std::string, double>>& val)
{
    return m_pilot_machine->parse_qst_density(result_str, val);
}

bool QPilotOSMachine::parse_qst_fidelity(const std::string& result_str, double& val)
{
    return m_pilot_machine->parse_qst_fidelity(result_str, val);
}

#if /*defined(USE_OPENSSL) &&*/ defined(USE_CURL)

void QPilotOSMachine::init(std::string url, bool log_cout, const std::string& api_key)
{
    PilotQVM::ELog::get_instance().set_output_log(log_cout);
    PTraceInfo("***************** On QPilotOSMachine::init for mode: "<< m_machine_type << " *****************\n" 
        << g_version << "\n");
    try
    {
        if (m_machine_type == "Pilot")
        {
            std::fstream pilot_config("/etc/statetab.d/sysinfo");
            if (pilot_config.is_open())
            {
                pilot_config >> url;
                pilot_config.close();
                if (url.empty()) {
                    PTraceWarn("Warn: No Permission with Pilot, try run local.");
                }
            }
        }
        PTraceInfo("Pilot url: " << url);
        m_machine_type = m_machine_type == "Pilot" && !url.empty() ? "Pilot" : "CPU";
        if (m_machine_type == "Pilot")
        {
            m_pilot_machine = new(std::nothrow) PilotQVM::QPilotMachine;
            m_noise_params = new(std::nothrow) PilotQVM::PilotNoiseParams;
            auto _r = m_pilot_machine->init(url, log_cout);
            PTraceInfo("m_pilot_machine init return " << _r);
        }
        else
        {
            m_cpu_machine = new(std::nothrow) QPanda::CPUQVM;
            m_cpu_machine->init();
        }
        
        _start();
        login_pilot_with_api_key(api_key);
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }

    PTraceInfo("***************** QPilotOSMachine::init ok ****************");
}

void QPilotOSMachine::init(std::string url, bool log_cout, const std::string& username, const std::string& pwd)
{
    PilotQVM::ELog::get_instance().set_output_log(log_cout);
    PTraceInfo("***************** On QPilotOSMachine::init for mode: " << m_machine_type << " *****************\n"
        << g_version << "\n");
    try
    {
        if (m_machine_type == "Pilot")
        {
            std::fstream pilot_config("/etc/statetab.d/sysinfo");
            if (pilot_config.is_open())
            {
                pilot_config >> url;
                pilot_config.close();
                if (url.empty()) {
                    PTraceWarn("Warn: No Permission with Pilot, try run local.");
                }
            }
        }
        PTraceInfo("Pilot url: " << url);
        m_machine_type = m_machine_type == "Pilot" && !url.empty() ? "Pilot" : "CPU";
        if (m_machine_type == "Pilot")
        {
            m_pilot_machine = new(std::nothrow) PilotQVM::QPilotMachine;
            m_noise_params = new(std::nothrow) PilotQVM::PilotNoiseParams;
            auto _r = m_pilot_machine->init(url, log_cout);
            PTraceInfo("m_pilot_machine init return " << _r);
        }
        else
        {
            m_cpu_machine = new(std::nothrow) QPanda::CPUQVM;
            m_cpu_machine->init();
        }
        _start();
        login_pilot(username, pwd);
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }

    PTraceInfo("***************** on QPilotOSMachine::init ok ****************");
}

double QPilotOSMachine::pMeasureBinindex(QProg& prog, std::string index, int backendID)
{
    try
    {
        QPanda::QVec qubit_vector;
        get_all_used_qubits(prog, qubit_vector);
        if (index.size() > qubit_vector.size())
        {
            throw std::runtime_error("pMeasureBinindex parms error:" + index);
        }
        if (m_machine_type == "Pilot")
        {
            PilotQVM::ErrorCode err_code;
            complex_d result;
            if ((err_code = m_pilot_machine->execute_single_amplitude_task(
                convert_qprog_to_originir(prog, this),
                binary_to_inter(index), result, backendID))
                != PilotQVM::ErrorCode::NO_ERROR_FOUND)
            {
                throw std::runtime_error("pMeasureBinindex run error,please check the parms,error code:"
                    + std::to_string(int(err_code)));
            }
            return result.real() * result.real() + result.imag() * result.imag();
        }
        QPanda::SingleAmplitudeQVM m_single_amplitude_machine;
        m_single_amplitude_machine.init();
        m_single_amplitude_machine.run(prog, qubit_vector);
        return m_single_amplitude_machine.pMeasureBinindex(index);
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

double QPilotOSMachine::pMeasureDecindex(QProg& prog, std::string index, int backendID)
{
    try
    {
        QPanda::QVec qubit_vector;
        get_all_used_qubits(prog, qubit_vector);
        auto m_value = std::atoll(index.c_str());
        if (m_value >= std::pow(2, qubit_vector.size()) || m_value < 0)
        {
            throw std::runtime_error("pMeasureDecindex parms error:" + index);
        }
        if (m_machine_type == "Pilot")
        {
            complex_d result;
            PilotQVM::ErrorCode err_code;
            if ((err_code = m_pilot_machine->execute_single_amplitude_task(
                convert_qprog_to_originir(prog, this), index, result, backendID))
                != PilotQVM::ErrorCode::NO_ERROR_FOUND)
            {
                throw std::runtime_error("pMeasureDecindex run error,please check the parms,error code:"
                    + std::to_string(int(err_code)));
            }
            return result.real() * result.real() + result.imag() * result.imag();
        }
        QPanda::SingleAmplitudeQVM m_single_amplitude_machine;
        m_single_amplitude_machine.init();
        m_single_amplitude_machine.run(prog, qubit_vector);
        return m_single_amplitude_machine.pMeasureDecindex(index);
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::unordered_map<std::string, std::complex<double>> QPilotOSMachine::pmeasure_subset(QProg& prog,
    const std::vector<std::string>& amplitude, int backendID)
{
    try
    {
        QPanda::QVec qubit_vector;
        auto m_qubit_size = get_all_used_qubits(prog, qubit_vector);
        for (auto m_values : amplitude)
        {
            auto m_value = std::atoll(m_values.c_str());
            if (m_value >= std::pow(2, qubit_vector.size())|| m_value<0)
            {
                throw std::runtime_error("pmeasure_subset parms error:" + m_value);
            }
        }
        if (m_machine_type == "Pilot")
        {
            std::map<std::string, std::complex<double>> result;
            std::unordered_map<std::string, std::complex<double>> m_result;
            PilotQVM::ErrorCode err_code;
            if ((err_code = m_pilot_machine->execute_partial_amplitude_task(
                convert_qprog_to_originir(prog, this), amplitude, result, backendID))
                != PilotQVM::ErrorCode::NO_ERROR_FOUND)
            {
                throw std::runtime_error("pmeasure_subset run error,please check the parms,error code:"
                    + std::to_string(int(err_code)));
            }
            for (auto rest : result)
            {
                m_result[rest.first] = rest.second;
            }
            return m_result;
        }
        QPanda::PartialAmplitudeQVM m_partial_amplitude_machine;
        m_partial_amplitude_machine.init();
        m_partial_amplitude_machine.qAllocMany(m_qubit_size);
        m_partial_amplitude_machine.run(prog);
        return m_partial_amplitude_machine.pmeasure_subset(amplitude);
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::map<std::string, double> QPilotOSMachine::probRunDict(QProg& prog,
    const std::vector<uint32_t>& qubit_vec, int backendID)
{
    try
    {
        QPanda::QVec qubit_vector;
        get_all_used_qubits(prog, qubit_vector);
        for (auto m_value : qubit_vec)
        {
            if (m_value >= qubit_vector.size())
            {
                throw std::runtime_error("probRunDict parms error:" + m_value);
            }
        }
        if (m_machine_type == "Pilot")
        {
            std::map<std::string, double> result;
            PilotQVM::ErrorCode err_code;
            if ((err_code = m_pilot_machine->execute_full_amplitude_pmeasure_task(
                convert_qprog_to_originir(prog, this), qubit_vec, result, backendID))
                != PilotQVM::ErrorCode::NO_ERROR_FOUND)
            {
                throw std::runtime_error("probRunDict run error,please check the parms,error code:"
                    + std::to_string(int(err_code)));
            }
            return result;
        }

        QPanda::QVec m_qubit_measure;
        for (size_t i = 0; i < qubit_vec.size(); i++)
        {
            m_qubit_measure.push_back(qubit_vector[qubit_vec[i]]);
        }
        return m_cpu_machine->probRunDict(prog, m_qubit_measure);
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::map<std::string, size_t> QPilotOSMachine::runWithConfiguration(QProg& prog, int shots, 
    const QPanda::NoiseModel& noise_model/* = QPanda::NoiseModel()*/)
{
    return runWithConfiguration(prog, shots, ANY_CLUSTER_BACKEND, noise_model);
}

std::map<std::string, size_t> QPilotOSMachine::runWithConfiguration(QProg& prog, int shots, 
    const uint32_t& backendID, const QPanda::NoiseModel& noise_model/* = QPanda::NoiseModel()*/)
{
    PTraceInfo("On QPilotOSMachine::runWithConfiguration.");
    try
    {
        std::map<std::string, size_t> m_result;
        PTraceInfo("On QPilotOSMachine::runWithConfiguration.");
        if (m_machine_type == "Pilot")
        {
            PTraceInfo("On run for pilot.");
            PilotQVM::ErrorCode err_code;
            std::map<std::string, double> result;
            if (noise_model.enabled())
            {
                PTraceInfo("noise_model enabled");
                auto noise_model_type = noise_model.get_noise_model_type();
                std::vector<double> single_params = noise_model.get_single_params();
                std::vector<double> double_params = noise_model.get_double_params();
                if (!m_pilot_machine->build_noise_params(
                    noise_model_type, single_params, double_params, *m_noise_params))
                {
                    throw std::runtime_error("runWithConfiguration noise model error,please check the parms.");
                }
                err_code = m_pilot_machine->execute_noise_measure_task(
                    convert_qprog_to_originir(prog, this), *m_noise_params, result, backendID, shots);
            }
            else
            {
                PTraceInfo("On no-noise  full_amplitude_measure task");
                err_code = m_pilot_machine->execute_full_amplitude_measure_task(
                    convert_qprog_to_originir(prog, this), result, backendID, shots);
            }

            PTraceInfo("err_code = " << (uint32_t)err_code);
            if (err_code != PilotQVM::ErrorCode::NO_ERROR_FOUND)
            {
                throw std::runtime_error("runWithConfiguration run error,please check the parms,error code:"
                    + std::to_string((int)err_code));
            }

            PTraceInfo("result.size()=" << result.size());
            for (auto rest : result)
            {
                m_result[rest.first] = size_t(rest.second * shots);
            }
            return m_result;
        }
        QPanda::QVec qubit_vector;
        auto m_qubit_size = get_all_used_qubits(prog, qubit_vector);
        m_cpu_machine->qAllocMany(m_qubit_size);

        return m_cpu_machine->runWithConfiguration(prog, shots, noise_model);
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

double QPilotOSMachine::real_chip_expectation(const QProg &prog, const std::string &hamiltonian, const std::vector<uint32_t> &qubits,
    const int shot, const int chip_id, const bool is_amend, const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block,
    const std::string& task_describe)
{
    try
    {
        std::vector<double> result;
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            config.backend_id = chip_id;
            config.shot = shot;
            config.task_type = static_cast<int>(TaskType::REAL_EXPECTATION);
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.hamiltonian = hamiltonian;
            config.ir = convert_qprog_to_originir(const_cast<QProg &>(prog), this);
            config.task_describe = task_describe;
            config.specified_block = specified_block;
            m_pilot_machine->execute_expectation_task(config, qubits, result);
            if (result.size() > 0) {
                return result[0];
            }
            else {
                throw std::runtime_error("task execute error!");
            }
        }        
        return 0;
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_expectation(const QProg& prog, const std::string &hamiltonian, const std::vector<uint32_t> &qubits,
    const int shot, const int chip_id, const bool is_amend, const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block,
    const std::string& task_describe)
{
    try
    {
        std::vector<double> result;
        if (m_machine_type == "Pilot") {
            std::string originir = convert_qprog_to_originir(const_cast<QProg&>(prog), this);
            CalcConfig config;
            config.backend_id = chip_id;
            config.shot = shot;
            config.task_type = static_cast<int>(TaskType::REAL_EXPECTATION);
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.hamiltonian = hamiltonian;
            config.ir = originir;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
            return m_pilot_machine->async_execute_expectation_task(config, qubits, result); /* return task_id */
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::map<std::string, double> QPilotOSMachine::real_chip_measure(const QProg& prog, int shot, int chip_id, bool is_amend, bool is_mapping, 
    bool is_optimization, const std::vector<uint32_t> &specified_block, const std::string& task_describe, int point_lable)
{
    try
    {
        /* 将prog压入vector, 进行批量提交处理 */
        std::vector<QPanda::QProg> prog_vec;
        prog_vec.emplace_back(prog);
        std::vector<std::map<std::string, double>> result = real_chip_measure_vec(prog_vec, shot, chip_id, is_amend, is_mapping, is_optimization, specified_block, task_describe, point_lable);
        if (result.size() > 0) {
            return result[0];
        }
        else {
            throw std::runtime_error("task execute error!");
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch exception" << e.what());
    }
}

std::map<std::string, double> QPilotOSMachine::real_chip_measure(const std::string& ir, int shot, int chip_id, bool is_amend, bool is_mapping, 
    bool is_optimization, const std::vector<uint32_t> &specified_block, const std::string& task_describe, int point_lable)
{
    try
    {
        /* 将prog压入vector, 进行批量提交处理 */
        std::vector<std::string> ir_vec;
        ir_vec.emplace_back(ir);
        std::vector<std::map<std::string, double>> result = real_chip_measure_vec(ir_vec, shot, chip_id, is_amend, is_mapping, is_optimization, specified_block, task_describe, point_lable);
        if (result.size() > 0) {
            return result[0];
        }
        else {
            throw std::runtime_error("task execute error!");
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch exception" << e.what());
    }
}

std::string QPilotOSMachine::real_chip_measure(const std::vector<QProg>& prog, const std::string& config_str)
{
    try
    {
        JsonMsg::JsonParser jp;
        jp.load_json(config_str);
        if (!jp.has_member_uint32("ChipID")) {
            return std::string("Config field need ChipID!");
        }

        std::string result;
        if (m_machine_type == "Pilot") 
        {
            CalcConfig config;
            for (uint32_t i = 0; i < prog.size(); i++)
            {
                config.ir_vec.push_back("");
                config.ir_vec[i] = convert_qprog_to_originir(const_cast<QProg&>(prog[i]), this);
            }

            config.backend_id = jp.get_uint32("ChipID");

            if (jp.has_member_uint32("shot")) {
                config.shot = jp.get_uint32("shot");
            }

            if (jp.has_member_uint32("TaskType")) {
                config.task_type = jp.get_uint32("TaskType");
            }

            if (jp.has_member_uint32("PulsePeriod")) {
                config.pulse_period = jp.get_uint32("PulsePeriod");
            }

            if (jp.has_member_bool("amendFlag")) {
                config.is_amend = jp.get_bool("amendFlag");
            }

            if (jp.has_member_bool("mappingFlag")) {
                config.is_amend = jp.get_bool("mappingFlag");
            }

            if (jp.has_member_bool("circuitOptimization")) {
                config.is_amend = jp.get_bool("circuitOptimization");
            }

            if (jp.has_member_array("specified_block")) {
                 jp.get_array("specified_block", config.specified_block);
            }

            if (jp.has_member_string("taskDescribe")) {
                config.task_describe = jp.get_string("taskDescribe");
            }

            if (jp.has_member_uint32("PointLable")) /*标签项*/
            {
                config.point_lable = jp.get_uint32("PointLable");
            }
            

            m_pilot_machine->execute_measure_task_vec(config, result);
            return result;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::vector<std::map<std::string, double>> QPilotOSMachine::real_chip_measure_vec(const std::vector<QProg>& prog, const int shot, 
    const int chip_id, const bool is_amend, const bool is_mapping, const bool is_optimization, const std::vector<uint32_t>& specified_block,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        std::vector<std::map<std::string, double>> result;
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            for (uint32_t i = 0; i < prog.size(); i++)
            {
                config.ir_vec.push_back("");
                config.ir_vec[i] = convert_qprog_to_originir(const_cast<QProg&>(prog[i]), this);
            }
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;        
            config.specified_block = specified_block;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            m_pilot_machine->execute_measure_task_vec(config, result);
            return result;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::vector<std::map<std::string, double>> QPilotOSMachine::real_chip_measure_vec(const std::vector<std::string>& ir, const int shot, 
    const int chip_id, const bool is_amend, const bool is_mapping, const bool is_optimization, const std::vector<uint32_t>& specified_block,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        std::vector<std::map<std::string, double>> result;
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.ir_vec = ir;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            m_pilot_machine->execute_measure_task_vec(config, result);
            return result;
        }
        /*std::vector<std::map<std::string, size_t> >m_result = m_cpu_machine->runWithConfiguration(prog[0], shot);
        for (auto &rest : m_result[0])
        {
            result[0][rest.first] = double(rest.second) / shot;
        }
        return result[0];*/ /* 为尽快推出批量计算接口，暂不开发CPU计算批量任务 */
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::map<std::string, size_t> QPilotOSMachine::real_chip_measure_prob_count(const QProg& prog,
    const int shot,
    const int chip_id,
    const bool is_mapping,
    const bool is_optimization,
    const std::vector<uint32_t>& specified_block,
    const std::string& task_describe,
    const int point_lable)
{
    try
    {
        /* 将prog压入vector, 进行批量提交处理 */
        std::vector<QPanda::QProg> prog_vec;
        prog_vec.emplace_back(prog);
        std::vector<std::map<std::string, size_t>> result = real_chip_measure_prob_count(prog_vec, shot, chip_id, is_mapping, is_optimization, specified_block, task_describe, point_lable);
        if (result.size() > 0) {
            return result[0];
        }
        else {
            throw std::runtime_error("task execute error!");
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch exception" << e.what());
    }
}

std::map<std::string, size_t> QPilotOSMachine::real_chip_measure_prob_count(const std::string& ir,
    const int shot,
    const int chip_id,
    const bool is_mapping,
    const bool is_optimization,
    const std::vector<uint32_t>& specified_block,
    const std::string& task_describe,
    const int point_lable)
{
    try
    {
        /* 将prog压入vector, 进行批量提交处理 */
        std::vector<std::string> ir_vec;
        ir_vec.emplace_back(ir);
        std::vector<std::map<std::string, size_t>> result = real_chip_measure_prob_count(ir_vec, shot, chip_id, is_mapping, is_optimization, specified_block, task_describe, point_lable);
        if (result.size() > 0) {
            return result[0];
        }
        else {
            throw std::runtime_error("task execute error!");
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch exception" << e.what());
    }
}

std::vector<std::map<std::string, size_t>> QPilotOSMachine::real_chip_measure_prob_count(const std::vector<QProg>& prog,
    const int shot,
    const int chip_id,
    const bool is_mapping,
    const bool is_optimization,
    const std::vector<uint32_t>& specified_block,
    const std::string& task_describe,
    const int point_lable)
{
    try
    {
        std::vector<std::map<std::string, size_t>> result;
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            for (uint32_t i = 0; i < prog.size(); i++)
            {
                config.ir_vec.push_back("");
                config.ir_vec[i] = convert_qprog_to_originir(const_cast<QProg&>(prog[i]), this);
            }
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = false;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.specified_block = specified_block;
            config.is_prob_counts = true;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            m_pilot_machine->execute_measure_task_vec(config, result);
            return result;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::vector<std::map<std::string, size_t>> QPilotOSMachine::real_chip_measure_prob_count(const std::vector<std::string>& ir,
    const int shot,
    const int chip_id,
    const bool is_mapping,
    const bool is_optimization,
    const std::vector<uint32_t>& specified_block,
    const std::string& task_describe,
    const int point_lable)
{
    try
    {
        std::vector<std::map<std::string, size_t>> result;
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = false;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.ir_vec = ir;
            config.specified_block = specified_block;
            config.is_prob_counts = true;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            m_pilot_machine->execute_measure_task_vec(config, result);
            return result;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_measure(const QProg& prog, const int shot, const int chip_id, const bool is_amend,
    const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block, const bool is_prob_counts,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        std::vector<QPanda::QProg> prog_vec;
        prog_vec.emplace_back(prog);
        std::string task_id = async_real_chip_measure_vec(prog_vec, shot, chip_id, is_amend, is_mapping, is_optimization, specified_block, is_prob_counts, task_describe, point_lable);
        return task_id;
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_measure(const std::string& ir, const int shot, const int chip_id, const bool is_amend,
    const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block, const bool is_prob_counts,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        std::vector<std::string> ir_vec;
        ir_vec.emplace_back(ir);
        std::string task_id = async_real_chip_measure_vec(ir_vec, shot, chip_id, is_amend, is_mapping, is_optimization, specified_block, is_prob_counts, task_describe, point_lable);
        return task_id;
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_measure(const std::vector<QProg>& prog, const std::string& config_str)
{
    if (m_machine_type == "Pilot")
    {
        try
        {
            JsonMsg::JsonParser jp;
            jp.load_json(config_str);
            if (!jp.has_member_uint32("ChipID")) {
                return std::string("Config field need ChipID!");
            }

            CalcConfig config;
            for (uint32_t i = 0; i < prog.size(); i++)
            {
                config.ir_vec.push_back("");
                config.ir_vec[i] = convert_qprog_to_originir(const_cast<QProg&>(prog[i]), this);
            }

            config.backend_id = jp.get_uint32("ChipID");

            if (jp.has_member_uint32("shot")) {
                config.shot = jp.get_uint32("shot");
            }

            if (jp.has_member_uint32("TaskType")) {
                config.task_type = jp.get_uint32("TaskType");
            }

            if (jp.has_member_uint32("PulsePeriod")) {
                config.pulse_period = jp.get_uint32("PulsePeriod");
            }

            if (jp.has_member_bool("amendFlag")) {
                config.is_amend = jp.get_bool("amendFlag");
            }

            if (jp.has_member_bool("mappingFlag")) {
                config.is_amend = jp.get_bool("mappingFlag");
            }

            if (jp.has_member_bool("circuitOptimization")) {
                config.is_amend = jp.get_bool("circuitOptimization");
            }

            if (jp.has_member_array("specified_block")) {
                jp.get_array("specified_block", config.specified_block);
            }

            if (jp.has_member_string("taskDescribe")) {
                config.task_describe = jp.get_string("taskDescribe");
            }

            /*标签项*/
            if (jp.has_member_uint32("PointLabel")) {   
                config.point_lable = jp.get_uint32("PointLabel");
            }
            

            std::string task_id;
            m_pilot_machine->async_execute_measure_task_vec(config, task_id);
            return task_id;
        }
        catch (const std::exception& e)
        {
            PTraceError("Catch unknow exception" << e.what());
        }
    }
}

std::string QPilotOSMachine::async_real_chip_measure_vec(const std::vector<QProg>& prog, const int shot, const int chip_id, const bool is_amend,
    const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block, const bool is_prob_counts,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            for (uint32_t i = 0; i < prog.size(); i++)
            {
                config.ir_vec.push_back("");
                config.ir_vec[i] = convert_qprog_to_originir(const_cast<QProg&>(prog[i]), this);
            }
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;                        
            config.specified_block = specified_block;
            config.is_prob_counts = is_prob_counts;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            std::string task_id;
            m_pilot_machine->async_execute_measure_task_vec(config, task_id);
            return task_id;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_measure_vec(const std::vector<std::string>& ir, const int shot, const int chip_id, const bool is_amend,
    const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block, const bool is_prob_counts,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            config.backend_id = chip_id;
            config.shot = shot;
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.ir_vec = ir;
            config.specified_block = specified_block;
            config.is_prob_counts = is_prob_counts;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            std::string task_id;
            m_pilot_machine->async_execute_measure_task_vec(config, task_id);
            return task_id;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_QST(const QProg& prog, const int shot, const int chip_id, const bool is_amend,
    const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            config.ir_vec.push_back(convert_qprog_to_originir(const_cast<QProg&>(prog), this));
            config.backend_id = chip_id;
            config.shot = shot;
            config.task_type = static_cast<int>(TaskType::QST_TASK);
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            std::string task_id;
            m_pilot_machine->async_execute_measure_task_vec(config, task_id);
            return task_id;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_QST_density(const QProg& prog, const int shot, const int chip_id, const bool is_amend,
    const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            config.ir_vec.push_back(convert_qprog_to_originir(const_cast<QProg&>(prog), this));
            config.backend_id = chip_id;
            config.shot = shot;
            config.task_type = static_cast<int>(TaskType::FIDELITY);
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            std::string task_id;
            m_pilot_machine->async_execute_measure_task_vec(config, task_id);
            return task_id;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_real_chip_QST_fidelity(const QProg& prog, const int shot, const int chip_id, const bool is_amend,
    const bool is_mapping, const bool is_optimization, const std::vector<uint32_t> &specified_block,
    const std::string& task_describe, const int point_lable)
{
    try
    {
        if (m_machine_type == "Pilot") {
            CalcConfig config;
            config.ir_vec.push_back(convert_qprog_to_originir(const_cast<QProg&>(prog), this));
            config.backend_id = chip_id;
            config.shot = shot;
            config.task_type = static_cast<int>(TaskType::FIDELITY);
            config.is_amend = is_amend;
            config.is_mapping = is_mapping;
            config.is_optimization = is_optimization;
            config.specified_block = specified_block;
            config.task_describe = task_describe;
            config.point_lable = point_lable;
            std::string task_id;
            m_pilot_machine->async_execute_measure_task_vec(config, task_id);
            return task_id;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

bool QPilotOSMachine::get_measure_result(const std::string& task_id, std::vector<std::map<std::string, double>>& result,
    PilotQVM::ErrorCode& errCode, std::string& errInfo)
{
    try
    {
        PilotTaskQueryResult res;
        do
        {
            query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FINISHED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FAILED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::CANCELLED))));

        if (res.m_errCode != 0 || res.m_result_vec.size() == 0)
        {
            PTraceInfo("Task failed!, errInfo: " << res.m_errInfo);
            std::cout << "Task failed!, errInfo: " << res.m_errInfo;
            errCode = static_cast<PilotQVM::ErrorCode>(res.m_errCode);
            errInfo = res.m_errInfo;
            return false;
        }

        parse_task_result(res.m_result_vec, result);
        errCode = PilotQVM::ErrorCode::NO_ERROR_FOUND;
        errInfo = "";
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        std::cout << "Exception happended: " << e.what();
        return false;
    }
    return false;

}

bool QPilotOSMachine::get_measure_result(const std::string& task_id, std::vector<std::map<std::string, uint64_t>>& result,
    PilotQVM::ErrorCode& errCode, std::string& errInfo)
{
    try
    {
        PilotTaskQueryResult res;
        do
        {
            query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FINISHED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FAILED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::CANCELLED))));

        if (res.m_errCode != 0 || res.m_result_vec.size() == 0)
        {
            PTraceInfo("Task failed!, errInfo: " << res.m_errInfo);
            std::cout << "Task failed!, errInfo: " << res.m_errInfo;
            errCode = static_cast<PilotQVM::ErrorCode>(res.m_errCode);
            errInfo = res.m_errInfo;
            return false;
        }

        parse_task_result(res.m_result_vec, result);
        errCode = PilotQVM::ErrorCode::NO_ERROR_FOUND;
        errInfo = "";
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        std::cout << "Exception happended: " << e.what();
        return false;
    }
    return false;

}

bool QPilotOSMachine::get_expectation_result(const std::string& task_id, double& result, 
    PilotQVM::ErrorCode& errCode, std::string& errInfo)
{
    try
    {
        PilotTaskQueryResult res;
        do
        {
            query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FINISHED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FAILED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::CANCELLED))));

        if (res.m_errCode != 0)
        {
            PTraceInfo("Task failed!, errInfo: " << res.m_errInfo);
            std::cout << "Task failed!, errInfo: " << res.m_errInfo;
            errCode = static_cast<PilotQVM::ErrorCode>(res.m_errCode);
            errInfo = res.m_errInfo;
            return false;
        }

        result = std::stod(res.m_result_vec[0]);
        errCode = PilotQVM::ErrorCode::NO_ERROR_FOUND;
        errInfo = "";
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        std::cout << "Exception happended: " << e.what();
        return false;
    }
    return false;

}

bool QPilotOSMachine::get_qst_result(const std::string& task_id, std::vector<std::map<std::string, double>>& result,
    PilotQVM::ErrorCode& errCode, std::string& errInfo)
{
    try
    {
        PilotTaskQueryResult res;
        do
        {
            query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FINISHED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FAILED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::CANCELLED))));

        if (res.m_errCode != 0)
        {
            PTraceInfo("Task failed!, errInfo: " << res.m_errInfo);
            std::cout << "Task failed!, errInfo: " << res.m_errInfo;
            errCode = static_cast<PilotQVM::ErrorCode>(res.m_errCode);
            errInfo = res.m_errInfo;
            return false;
        }

        parse_task_result(res.m_result_vec, result);
        errCode = PilotQVM::ErrorCode::NO_ERROR_FOUND;
        errInfo = "";
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        std::cout << "Exception happended: " << e.what();
        return false;
    }
}

bool QPilotOSMachine::get_qst_density_result(const std::string & task_id, std::vector<std::map<std::string, double>>& result,
    PilotQVM::ErrorCode & errCode, std::string & errInfo)
{
    try
    {
        PilotTaskQueryResult res;
        do
        {
            query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FINISHED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FAILED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::CANCELLED))));

        if (res.m_errCode != 0)
        {
            PTraceInfo("Task failed!, errInfo: " << res.m_errInfo);
            std::cout << "Task failed!, errInfo: " << res.m_errInfo;
            errCode = static_cast<PilotQVM::ErrorCode>(res.m_errCode);
            errInfo = res.m_errInfo;
            return false;
        }

        parse_qst_density(res.m_qst_density, result);
        errCode = PilotQVM::ErrorCode::NO_ERROR_FOUND;
        errInfo = "";
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        std::cout << "Exception happended: " << e.what();
        return false;
    }
}

bool QPilotOSMachine::get_qst_fidelity_result(const std::string & task_id, double & result,
    PilotQVM::ErrorCode & errCode, std::string & errInfo)
{
    try
    {
        PilotTaskQueryResult res;
        do
        {
            query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FINISHED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::FAILED)))
            && (res.m_state != std::to_string(static_cast<int>(PilotQVM::TaskStatus::CANCELLED))));

        if (res.m_errCode != 0)
        {
            PTraceInfo("Task failed!, errInfo: " << res.m_errInfo);
            std::cout << "Task failed!, errInfo: " << res.m_errInfo;
            errCode = static_cast<PilotQVM::ErrorCode>(res.m_errCode);
            errInfo = res.m_errInfo;
            return false;
        }

        parse_qst_fidelity(res.m_qst_fidelity, result);
        errCode = PilotQVM::ErrorCode::NO_ERROR_FOUND;
        errInfo = "";
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceError("Exception happended: " << e.what());
        std::cout << "Exception happended: " << e.what();
        return false;
    }
    return false;
}

std::string QPilotOSMachine::noise_learning(const std::string& parameter_json)
{
    try
    {
        if (m_machine_type == "Pilot") 
        {
            std::string task_id;
            m_pilot_machine->execute_noise_learning_task(parameter_json, task_id);
            return task_id;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

std::string QPilotOSMachine::async_em_compute(const std::string& parameter_json)
{
    std::string task_id;
    try
    {
        if (m_machine_type == "Pilot") {
#if 0
            NoiseConfig noise_config;
            noise_config.ir = qcir;
            noise_config.noise_strength = noiseStrength;
            noise_config.loops = loops;
            noise_config.shots = shot;
            if ("" == noiseLearningResultFile)
            {
                PTraceInfo("noiseLearningResultFile isnt specified");
                std::string task_id = "noiseLearningResultFile isnt specified";
                return task_id;
            }
            noise_config.noise_learning_result_file = noiseLearningResultFile;
#endif
            m_pilot_machine->async_execute_em_compute_task(parameter_json, task_id);
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
    return task_id;
}

std::vector<double> QPilotOSMachine::em_compute(const std::string& parameter_json)
{
    try
    {
        if (m_machine_type == "Pilot") 
        {
            std::string task_id;
            std::vector<double> result;
            m_pilot_machine->execute_em_compute_task(parameter_json, task_id, result);
            return result;
            //return task_id;
        }
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
    }
}

void QPilotOSMachine::set_noise_model(NOISE_MODEL model, const std::vector<double> single_params,
    const std::vector<double> double_params)
{
    m_pilot_machine->build_noise_params(model, single_params, double_params, *m_noise_params);
}

std::map<std::string, double> QPilotOSMachine::noise_measure(QProg& prog, int shot)
{
    std::map<std::string, double> result;
    PilotQVM::ErrorCode m_error = m_pilot_machine->execute_noise_measure_task(
        convert_qprog_to_originir(prog, this),
        *m_noise_params,
        result,
         ANY_CLUSTER_BACKEND  /*200*/,
        shot);
    if (m_error != PilotQVM::ErrorCode::NO_ERROR_FOUND)
    {
        QCERR_AND_THROW(run_fail, "QPILOTOS MACHINE ERROR");
    }
    return result;
}

std::map<std::string, double> QPilotOSMachine::full_amplitude_measure(QProg&prog, int shot)
{
    std::map<std::string, double> result;
    PilotQVM::ErrorCode m_error = m_pilot_machine->execute_full_amplitude_measure_task(
        convert_qprog_to_originir(prog, this), result,  ANY_CLUSTER_BACKEND  /*200*/, shot);
    if (m_error != PilotQVM::ErrorCode::NO_ERROR_FOUND){
        QCERR_AND_THROW(run_fail, "Error: full_amplitude run error: " << (uint32_t)m_error);
    }
    return result;
}

std::map<std::string, double> QPilotOSMachine::full_amplitude_pmeasure(QProg& prog, Qnum qubit_vec)
{
    std::vector<uint32_t> m_qubit_vec;
    std::map<std::string, double> result;
    for (uint32_t i = 0; i < qubit_vec.size(); i++)
    {
        m_qubit_vec.push_back(qubit_vec[i]);
    }
    PilotQVM::ErrorCode m_error = m_pilot_machine->execute_full_amplitude_pmeasure_task(
        convert_qprog_to_originir(prog, this), m_qubit_vec, result,  ANY_CLUSTER_BACKEND  /*200*/);
    if (m_error != PilotQVM::ErrorCode::NO_ERROR_FOUND)
    {
        QCERR_AND_THROW(run_fail, "QPILOTOS MACHINE ERROR");
    }
    return result;
}

std::map<std::string, qcomplex_t> QPilotOSMachine::partial_amplitude_pmeasure(QProg& prog, std::vector<std::string> amplitude_vec)
{
    std::map<std::string, qcomplex_t> result;
    PilotQVM::ErrorCode m_error = m_pilot_machine->execute_partial_amplitude_task(
        convert_qprog_to_originir(prog, this), amplitude_vec, result,  ANY_CLUSTER_BACKEND  /*200*/);
    if (m_error != PilotQVM::ErrorCode::NO_ERROR_FOUND)
    {
        QCERR_AND_THROW(run_fail, "QPILOTOS MACHINE ERROR");
    }
    return result;
}

qcomplex_t QPilotOSMachine::single_amplitude_pmeasure(QProg& prog, std::string amplitude)
{
    qcomplex_t result;
    PilotQVM::ErrorCode m_error = m_pilot_machine->execute_single_amplitude_task(
        convert_qprog_to_originir(prog, this), amplitude, result,  ANY_CLUSTER_BACKEND  /*201*/);
    if (m_error != PilotQVM::ErrorCode::NO_ERROR_FOUND)
    {
        QCERR_AND_THROW(run_fail, "QPILOTOS MACHINE ERROR");
    }
    return result;
}

void QPilotOSMachine::real_chip_task_validation(int shots, QProg& prog)
{
    QVec qubits;
    std::vector<int> cbits;

    auto qbit_num = get_all_used_qubits(prog, cbits);
    auto cbit_num = get_all_used_class_bits(prog, cbits);

    QPANDA_ASSERT(qbit_num > 6 || cbit_num > 6, "real chip qubit num or cbit num are not less or equal to 6");
    QPANDA_ASSERT(shots > 10000 || shots < 1000, "real chip shots must be in range [1000,10000]");

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    if (!traver_param.m_can_optimize_measure)
    {
        QCERR("measure must be last");
        throw run_fail("measure must be last");
    }
}

void QPilotOSMachine::construct_real_chip_task_json(
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
    size_t chip_id)
{
    doc.insert("code", prog_str);
    doc.insert("apiKey", token);
    doc.insert("isAmend", is_amend);
    doc.insert("mappingFlag", is_mapping);
    doc.insert("circuitOptimization", is_optimization);
    doc.insert("QMachineType", qvm_type);
    doc.insert("codeLen", prog_str.size());
    doc.insert("qubitNum", qubit_num);
    doc.insert("measureType", measure_type);
    doc.insert("classicalbitNum", cbit_num);
    doc.insert("shot", (size_t)shot);
    doc.insert("chipId", (size_t)chip_id);
}

void QPilotOSMachine::construct_cluster_task_json(
    rabbit::document& doc,
    std::string prog_str,
    std::string token,
    size_t qvm_type,
    size_t qubit_num,
    size_t cbit_num,
    size_t measure_type)
{
    doc.insert("code", prog_str);
    doc.insert("apiKey", token);
    doc.insert("QMachineType", qvm_type);
    doc.insert("codeLen", prog_str.size());
    doc.insert("qubitNum", qubit_num);
    doc.insert("measureType", measure_type);
    doc.insert("classicalbitNum", cbit_num);
}

bool QPilotOSMachine::parser_submit_json(std::string& recv_json, std::string& taskid)
{
    try
    {
        rabbit::document doc;
        doc.parse(recv_json);

        auto success = doc["success"].as_bool();
        if (!success)
        {
            PTraceInfo("Recv json:\n" << recv_json);
            auto message = doc["message"].as_string();
            QCERR_AND_THROW(run_fail, message);
        }
        else
        {
            taskid = doc["obj"]["taskId"].as_string();
            return true;
        }
    }
    catch (const std::exception& e)
    {
        QCERR_AND_THROW(run_fail, e.what());
    }
}

void QPilotOSMachine::json_string_transfer_encoding(std::string& str)
{
    //delete "\r\n" from recv json, Transfer-Encoding: chunked
    int pos = 0;
    while ((pos = str.find("\n")) != -1)
    {
        str.erase(pos, 1);
    }
}

double QPilotOSMachine::get_state_fidelity(
    QProg& prog,
    int shots,
    uint32_t chip_id,
    bool is_amend,
    bool is_mapping,
    bool is_optimization)
{
    real_chip_task_validation(shots, prog);

    //convert prog to originir
    auto prog_str = convert_qprog_to_originir(prog, this);

    rabbit::document doc;
    doc.parse("{}");

    construct_real_chip_task_json(doc, prog_str, m_token, is_amend, is_mapping, is_optimization,
        (size_t)QPilotQMchineType::FIDELITY, getAllocateQubitNum(), getAllocateCMem(),
        (size_t)PilotTaskType::CLUSTER_MEASURE, shots, (size_t)chip_id);

    return m_qst_fidelity;
}

double QPilotOSMachine::get_expectation(QProg prog, const QHamiltonian& hamiltonian, const QVec& qubit_vec)
{
    double result = 0;
    std::vector<uint32_t> qubit_vec_i;
    for (const auto& q_i : qubit_vec){
        qubit_vec_i.emplace_back(q_i->get_phy_addr());
    }

    const PilotQVM::ErrorCode _error = m_pilot_machine->execute_full_amplitude_expectation(
        convert_qprog_to_originir(prog, this), hamiltonian, qubit_vec_i, result);
    if (_error != PilotQVM::ErrorCode::NO_ERROR_FOUND){
        QCERR_AND_THROW(run_fail, "Error: Failed to get expectation:" << (uint32_t)_error);
    }

    return result;
}

bool QPilotOSMachine::parser_result_json(std::string& recv_json, std::string& taskid)
{
    json_string_transfer_encoding(recv_json);

    rabbit::document recv_doc;
    recv_doc.parse(recv_json.c_str());

    if (!recv_doc["success"].as_bool())
    {
        m_error_info = recv_doc["enMessage"].as_string();
        QCERR_AND_THROW(run_fail, m_error_info);
    }

    try
    {
        auto list = recv_doc["obj"]["qcodeTaskNewVo"]["taskResultList"];
        std::string state = list[0]["taskState"].as_string();
        std::string qtype = list[0]["rQMachineType"].as_string();

        auto status = static_cast<TaskStatus>(atoi(state.c_str()));
        auto backend_type = static_cast<QPilotQMchineType>(atoi(qtype.c_str()));
        switch (status)
        {
        case TaskStatus::FINISHED:
        {
            auto result_string = list[0]["taskResult"].as_string();

            rabbit::document result_doc;
            result_doc.parse(result_string.c_str());

            switch (backend_type)
            {
            case QPilotQMchineType::REAL_CHIP:
            {
                m_measure_result.clear();
                for (auto i = 0; i < result_doc["key"].size(); ++i)
                {
                    auto key = result_doc["key"][i].as_string();
                    auto val = result_doc["value"][i].is_double() ?
                        result_doc["value"][i].as_double() : (double)result_doc["value"][i].as_int();

                    m_measure_result.insert(make_pair(key, val));
                }

                break;
            }

            case QPilotQMchineType::NOISE_QMACHINE:
            case QPilotQMchineType::Full_AMPLITUDE:
            {

                auto result_type = result_doc["ResultType"].as_int();

                if (result_type == (int)PilotResultType::EXPECTATION)
                {
                    m_expectation = result_doc["Value"].is_double() ?
                        result_doc["Value"].as_double() : (double)result_doc["Value"].as_int();
                }
                else
                {
                    m_measure_result.clear();
                    for (auto i = 0; i < result_doc["Key"].size(); ++i)
                    {
                        auto key = result_doc["Key"][i].as_string();
                        auto val = result_doc["Value"][i].is_double() ?
                            result_doc["Value"][i].as_double() : (double)result_doc["Value"][i].as_int();

                        m_measure_result.insert(make_pair(key, val));
                    }
                }

                break;
            }

            case QPilotQMchineType::PARTIAL_AMPLITUDE:
            {
                m_pmeasure_result.clear();
                for (size_t i = 0; i < result_doc["Key"].size(); ++i)
                {
                    auto key = result_doc["Key"][i].as_string();
                    auto val_real = result_doc["ValueReal"][i].is_double() ?
                        result_doc["ValueReal"][i].as_double() : (double)result_doc["ValueReal"][i].as_int();
                    auto val_imag = result_doc["ValueImag"][i].is_double() ?
                        result_doc["ValueImag"][i].as_double() : (double)result_doc["ValueImag"][i].as_int();

                    m_pmeasure_result.insert(make_pair(key, qcomplex_t(val_real, val_imag)));
                }

                break;
            }

            case QPilotQMchineType::SINGLE_AMPLITUDE:
            {
                auto val_real = result_doc["ValueReal"][0].is_double() ?
                    result_doc["ValueReal"][0].as_double() : (double)result_doc["ValueReal"][0].as_int();
                auto val_imag = result_doc["ValueImag"][0].is_double() ?
                    result_doc["ValueImag"][0].as_double() : (double)result_doc["ValueImag"][0].as_int();

                m_single_result = qcomplex_t(val_real, val_imag);
                break;
            }

            case QPilotQMchineType::QST:
            {
                rabbit::document qst_result_doc;
                qst_result_doc.parse(list[0]["qstresult"].as_string());

                m_qst_result.clear();
                int rank = (int)std::sqrt(qst_result_doc.size());

                for (auto i = 0; i < rank; ++i)
                {
                    QStat row_value;
                    for (auto j = 0; j < rank; ++j)
                    {
                        auto qst_result_real_value = qst_result_doc[i * rank + j]["r"];
                        auto qst_result_imag_value = qst_result_doc[i * rank + j]["i"];

                        auto real_val = qst_result_real_value.is_double() ?
                            qst_result_real_value.as_double() : (double)qst_result_real_value.as_int();
                        auto imag_val = qst_result_imag_value.is_double() ?
                            qst_result_imag_value.as_double() : (double)qst_result_imag_value.as_int();

                        row_value.emplace_back(qcomplex_t(real_val, imag_val));
                    }

                    m_qst_result.emplace_back(row_value);
                }

                break;
            }

            case QPilotQMchineType::FIDELITY:
            {
                std::string qst_fidelity_str = list[0]["qstfidelity"].as_string();
                m_qst_fidelity = stod(qst_fidelity_str);

                break;
            }

            default:
                QCERR_AND_THROW(run_fail, "quantum machine type error");
            }

            return false;
        }

        case TaskStatus::FAILED:
        {
            QCERR_AND_THROW(run_fail, "Task run failed");
        }

        //case TaskStatus::WAITING:
        //case TaskStatus::COMPUTING:
        //case TaskStatus::QUEUING:
        //    m_task_status = status;

        //    /* The next status only appear in real chip backend */
        //case TaskStatus::SENT_TO_BUILD_SYSTEM:
        //case TaskStatus::BUILD_SYSTEM_RUN:
        //    return true;

        //case TaskStatus::BUILD_SYSTEM_ERROR:
        //    QCERR_AND_THROW(run_fail, "build system error");

        //case TaskStatus::SEQUENCE_TOO_LONG:
        //    QCERR_AND_THROW(run_fail, "exceeding maximum timing sequence");

        default:
            m_task_status = (PilotQVM::TaskStatus)status;
            PTraceError("Error: On status: " << (uint32_t)status);
            return true;
        }
    }
    catch (const std::exception& e)
    {
        PTraceError("Error: exception for Recv json:\n" << recv_json << "\nexception:" << e.what());
        QCERR_AND_THROW(run_fail, "parse result exception error");
    }

    return false;
}

bool QPilotOSMachine::query_task_state(const std::string& task_id, PilotQVM::PilotTaskQueryResult& result)
{
    return m_pilot_machine->execute_query_task_state(task_id, result);
}

bool QPilotOSMachine::query_task_state(const std::string& task_id, PilotQVM::PilotTaskQueryResult& result, const bool save_to_file, std::string& file_path)
{
    bool success = m_pilot_machine->execute_query_task_state(task_id, result);
    if (success && save_to_file)
    {
        if (result.m_state == "3")
        {
            if (file_path.size() == 0)
            {
                file_path = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".json";
            }
            else
            {
                if (file_path.back() != '/' && !(file_path.size() == 2 && file_path.back() == ':')) {
                    file_path += '/' + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".json";
                }
                else{
                    file_path += std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".json";
                }
            }
            std::ofstream outputFile(file_path);
            if (!outputFile.is_open())
            {
                PTraceError("Can't open this file: " << file_path);
                return false;
            }
            outputFile << result.m_resultJson;
            outputFile.close();
            std::cout << "task result is saved in file: " << file_path << std::endl;
        }
    }
    return success;
}

bool QPilotOSMachine::query_compile_prog(const std::string task_id, std::string& compile_prog, bool without_compensate)
{
    return m_pilot_machine->execute_query_compile_prog(task_id, compile_prog, without_compensate);
}

bool QPilotOSMachine::login_pilot_with_api_key(const std::string& api_key)
{
    try
    {
        const PilotQVM::ErrorCode err_code = m_pilot_machine->execute_login_pilot_api(api_key);
        if (err_code != PilotQVM::ErrorCode::NO_ERROR_FOUND)
        {
            throw std::runtime_error("login pilot error,please check the parms,error code:"
                + std::to_string(int(err_code)));
        }
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
        return false;
    }
}

bool QPilotOSMachine::login_pilot(const std::string& username, const std::string& pwd)
{
    try
    {
        const PilotQVM::ErrorCode err_code = m_pilot_machine->execute_login_pilot(username, pwd);
        if (err_code != PilotQVM::ErrorCode::NO_ERROR_FOUND)
        {
            throw std::runtime_error("login pilot error,please check the parms,error code:"
                + std::to_string(int(err_code)));
        }
        return true;
    }
    catch (const std::exception& e)
    {
        PTraceInfo("Catch unknow exception" << e.what());
        return false;
    }
}

#endif