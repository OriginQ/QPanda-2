#include "QPandaConfig.h"
#include "QPanda.h"
#include "Extensions/Extensions.h"
#include "Extensions/QAlg/HHL/HHL.h"
#include "Extensions/PilotOSMachine/QPilotMachine.h"
#include "Extensions/PilotOSMachine/data_struct.h"
#include <math.h>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"
#include "Core/Utilities/UnitaryDecomposer/QSDecomposition.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA

#include "Extensions/VirtualZTransfer/VirtualZTransfer.h"

void export_extension_class(py::module& m)
{
#ifdef USE_EXTENSION
    py::class_<HHLAlg>(m, "HHLAlg", " quantum hhl algorithm class")
        .def(py::init<QuantumMachine*>())
        .def("get_hhl_circuit",
            &HHLAlg::get_hhl_circuit,
            py::arg("matrix_A"),
            py::arg("data_b"),
            py::arg("precision_cnt") = 0,
            py::return_value_policy::automatic)

        .def("check_QPE_result", &HHLAlg::check_QPE_result, "check QPE result")

        .def("get_amplification_factor", &HHLAlg::get_amplification_factor, "get_amplification_factor")

        .def("get_ancillary_qubit", &HHLAlg::get_ancillary_qubit, "get_ancillary_qubit")

        .def(
            "get_qubit_for_b",
            [](HHLAlg& self)
            {
                std::vector<Qubit*> q_vec = self.get_qubit_for_b();
                return q_vec;
            },
            "get_qubit_for_b",
            py::return_value_policy::reference)

        .def(
            "get_qubit_for_QFT",
            [](HHLAlg& self)
            {
                std::vector<Qubit*> q_vec = self.get_qubit_for_QFT();
                return q_vec;
            },
            "get_qubit_for_QFT",
             py::return_value_policy::reference)

        .def("query_uesed_qubit_num", &HHLAlg::query_uesed_qubit_num, "query_uesed_qubit_num");

//#if defined(USE_OPENSSL) && defined(USE_CURL)

    py::class_<PilotQVM::PilotNoiseParams>(m, "PilotNoiseParams", "pliot noise simulate params")
        .def_readwrite("noise_model", &PilotQVM::PilotNoiseParams::noise_model)
        .def_readwrite("single_gate_param", &PilotQVM::PilotNoiseParams::single_gate_param)
        .def_readwrite("double_gate_param", &PilotQVM::PilotNoiseParams::double_gate_param)
        .def_readwrite("single_p2", &PilotQVM::PilotNoiseParams::single_p2)
        .def_readwrite("double_p2", &PilotQVM::PilotNoiseParams::double_p2)
        .def_readwrite("single_pgate", &PilotQVM::PilotNoiseParams::single_pgate)
        .def_readwrite("double_pgate", &PilotQVM::PilotNoiseParams::double_pgate);

    py::enum_<PilotQVM::ErrorCode>(m, "ErrorCode", "pliot error code")
        .value("NO_ERROR_FOUND", PilotQVM::ErrorCode::NO_ERROR_FOUND)
        .value("DATABASE_ERROR", PilotQVM::ErrorCode::DATABASE_ERROR)
        .value("ORIGINIR_ERROR", PilotQVM::ErrorCode::ORIGINIR_ERROR)
        .value("JSON_FIELD_ERROR", PilotQVM::ErrorCode::JSON_FIELD_ERROR)
        .value("BACKEND_CALC_ERROR", PilotQVM::ErrorCode::BACKEND_CALC_ERROR)
        .value("ERR_TASK_BUF_OVERFLOW", PilotQVM::ErrorCode::ERR_TASK_BUF_OVERFLOW)
        .value("EXCEED_MAX_QUBIT", PilotQVM::ErrorCode::EXCEED_MAX_QUBIT)
        .value("ERR_UNSUPPORT_BACKEND_TYPE", PilotQVM::ErrorCode::ERR_UNSUPPORT_BACKEND_TYPE)
        .value("EXCEED_MAX_CLOCK", PilotQVM::ErrorCode::EXCEED_MAX_CLOCK)
        .value("ERR_UNKNOW_TASK_TYPE", PilotQVM::ErrorCode::ERR_UNKNOW_TASK_TYPE)
        .value("ERR_QVM_INIT_FAILED", PilotQVM::ErrorCode::ERR_QVM_INIT_FAILED)
        .value("ERR_QCOMPILER_FAILED", PilotQVM::ErrorCode::ERR_QCOMPILER_FAILED)
        .value("ERR_PRE_ESTIMATE", PilotQVM::ErrorCode::ERR_PRE_ESTIMATE)
        .value("ERR_MATE_GATE_CONFIG", PilotQVM::ErrorCode::ERR_MATE_GATE_CONFIG)
        .value("ERR_FIDELITY_MATRIX", PilotQVM::ErrorCode::ERR_FIDELITY_MATRIX)
        .value("ERR_QST_PROG", PilotQVM::ErrorCode::ERR_QST_PROG)
        .value("ERR_EMPTY_PROG", PilotQVM::ErrorCode::ERR_EMPTY_PROG)
        .value("ERR_QUBIT_SIZE", PilotQVM::ErrorCode::ERR_QUBIT_SIZE)
        .value("ERR_QUBIT_TOPO", PilotQVM::ErrorCode::ERR_QUBIT_TOPO)
        .value("ERR_QUANTUM_CHIP_PROG", PilotQVM::ErrorCode::ERR_QUANTUM_CHIP_PROG)
        .value("ERR_REPEAT_MEASURE", PilotQVM::ErrorCode::ERR_REPEAT_MEASURE)
        .value("ERR_OPERATOR_DB", PilotQVM::ErrorCode::ERR_OPERATOR_DB)
        .value("ERR_TASK_STATUS_BUF_OVERFLOW", PilotQVM::ErrorCode::ERR_TASK_STATUS_BUF_OVERFLOW)
        .value("ERR_BACKEND_CHIP_TASK_SOCKET_WRONG", PilotQVM::ErrorCode::ERR_BACKEND_CHIP_TASK_SOCKET_WRONG)
        .value("CLUSTER_SIMULATE_CALC_ERR", PilotQVM::ErrorCode::CLUSTER_SIMULATE_CALC_ERR)
        .value("ERR_SCHEDULE_CHIP_TOPOLOGY_SUPPORTED", PilotQVM::ErrorCode::ERR_SCHEDULE_CHIP_TOPOLOGY_SUPPORTED)
        .value("ERR_TASK_CONFIG", PilotQVM::ErrorCode::ERR_TASK_CONFIG)
        .value("ERR_NOT_FOUND_APP_ID", PilotQVM::ErrorCode::ERR_NOT_FOUND_APP_ID)
        .value("ERR_NOT_FOUND_TASK_ID", PilotQVM::ErrorCode::ERR_NOT_FOUND_TASK_ID)
        .value("ERR_PARSER_SUB_TASK_RESULT", PilotQVM::ErrorCode::ERR_PARSER_SUB_TASK_RESULT)
        .value("ERR_SYS_CALL_TIME_OUT", PilotQVM::ErrorCode::ERR_SYS_CALL_TIME_OUT)
        .value("ERR_TASK_TERMINATED", PilotQVM::ErrorCode::ERR_TASK_TERMINATED)
        .value("ERR_INVALID_URL", PilotQVM::ErrorCode::ERR_INVALID_URL)
        .value("ERR_PARAMETER", PilotQVM::ErrorCode::ERR_PARAMETER)
        .value("ERR_QPROG_LENGTH", PilotQVM::ErrorCode::ERR_QPROG_LENGTH)
        .value("ERR_CHIP_OFFLINE", PilotQVM::ErrorCode::ERR_CHIP_OFFLINE)
        .value("UNDEFINED_ERROR", PilotQVM::ErrorCode::UNDEFINED_ERROR)
        .value("ERR_SUB_GRAPH_OUT_OF_RANGE", PilotQVM::ErrorCode::ERR_SUB_GRAPH_OUT_OF_RANGE)
        .value("ERR_TCP_INIT_FATLT", PilotQVM::ErrorCode::ERR_TCP_INIT_FATLT)
        .value("ERR_TCP_SERVER_HALT", PilotQVM::ErrorCode::ERR_TCP_SERVER_HALT)
        .value("CLUSTER_BASE", PilotQVM::ErrorCode::CLUSTER_BASE)
        .export_values();

#ifndef  DOCKER
#ifdef USE_CURL
    py::class_<PilotQVM::QPilotMachine>(m, "QPilotMachine", "pliot machine")
        .def(py::init<>())

        .def("init", &PilotQVM::QPilotMachine::init,
            py::arg("pilot_url"),
            py::arg("log_cout") = false,
            py::return_value_policy::automatic)

        .def("init_withconfig",
            &PilotQVM::QPilotMachine::init_withconfig,
            py::arg("config_path") = "pilotmachine.conf",
            py::return_value_policy::automatic)

        .def("execute_measure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, 
            const uint32_t& chip_id, const bool& b_mapping,
            const bool& b_optimization, const uint32_t& shots,
            const std::vector<uint32_t> &specified_block )->
            std::vector<std::map<std::string, double>> {
                std::vector<std::map<std::string, double>> result;
                CalcConfig config;
                config.backend_id = chip_id;
                config.shot = shots;
                config.is_mapping = b_mapping;
                config.is_optimization = b_optimization;
                config.ir = prog_str;
                config.specified_block = specified_block;
                auto ErrInfo = self.execute_measure_task_vec(config, result);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                    throw::runtime_error("execute_measure_task run error:" + std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("b_mapping") = true,
            py::arg("b_optimization") = true,
            py::arg("shots") = 1000,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::return_value_policy::reference)

        .def("execute_callback_measure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, std::function<void(PilotQVM::ErrorCode,
                const std::map<std::string, double>&)>cb_func, const uint32_t& chip_id,
                const bool& b_mapping, const bool& b_optimization, const uint32_t& shots, 
                const std::vector<uint32_t> &specified_block = std::vector<uint32_t>{})->PilotQVM::ErrorCode {
                CalcConfig config;
                config.backend_id = chip_id;
                config.shot = shots;
                config.is_mapping = b_mapping;
                config.is_optimization = b_optimization;
                config.ir = prog_str;
                config.specified_block = specified_block;
                return self.execute_measure_task(config, cb_func);
            },
            py::arg("prog_str"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("b_mapping") = true,
            py::arg("b_optimization") = true,
            py::arg("shots") = 1000,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::return_value_policy::reference)

        .def("execute_full_amplitude_measure_task",
            [&](PilotQVM::QPilotMachine &self, const std::string& prog_str, 
            const uint32_t& chip_id, const uint32_t& shots)->
            std::map<std::string, double> {
                std::map<std::string, double> result;
                auto ErrInfo = self.execute_full_amplitude_measure_task(prog_str, result, chip_id, shots);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                    throw::runtime_error("execute_full_amplitude_measure_task run error:" + std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::arg("shots") = 1000,
            py::return_value_policy::reference)

        .def("execute_callback_full_amplitude_measure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str,
                std::function<void(PilotQVM::ErrorCode, const std::map<std::string, double>&)>cb_func,
                const uint32_t& chip_id, const uint32_t& shots)->PilotQVM::ErrorCode {
                return self.execute_full_amplitude_measure_task(prog_str, cb_func, chip_id, shots);
            },
            py::arg("prog_str"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::arg("shots") = 1000,
            py::return_value_policy::reference)

        .def("execute_full_amplitude_pmeasure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, 
                const std::vector<uint32_t>& qubit_vec, const uint32_t& chip_id)->
                std::map<std::string, double> {
                std::map<std::string, double> result;
                auto ErrInfo = self.execute_full_amplitude_pmeasure_task(prog_str, qubit_vec, result, chip_id);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                    throw::runtime_error("execute_full_amplitude_pmeasure_task run error:" + std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("qubit_vec"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,   
            py::return_value_policy::reference)

        .def("execute_callback_full_amplitude_pmeasure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::vector<uint32_t>& qubit_vec, 
                std::function<void(PilotQVM::ErrorCode, const std::map<std::string, double>&)>cb_func,
                const uint32_t& chip_id )->PilotQVM::ErrorCode {
                return self.execute_full_amplitude_pmeasure_task(prog_str, qubit_vec, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("qubit_vec"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)

        .def("build_noise_params",
            [&](PilotQVM::QPilotMachine& self, const uint32_t& nose_model_type, const std::vector<double>& single_params,
                const std::vector<double>& double_params)->PilotQVM::PilotNoiseParams {
                PilotQVM::PilotNoiseParams noise_params;
                auto ErrInfo = self.build_noise_params(nose_model_type, single_params, double_params, noise_params);
                if (!ErrInfo)
                {
                    throw::runtime_error("build_noise_params error.");
                }
                    return noise_params;
            },
            py::arg("nose_model_type"),
            py::arg("single_params"),
            py::arg("double_params"),
            py::return_value_policy::reference)

        .def("execute_noise_measure_task",
            [&](PilotQVM::QPilotMachine& self, std::string prog_str, const PilotQVM::PilotNoiseParams& noise_params,
                uint32_t chip_id, uint32_t shots)->std::map<std::string, double> {
                std::map<std::string, double> result;
                auto ErrInfo = self.execute_noise_measure_task(prog_str, noise_params, result, chip_id, shots);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                    throw::runtime_error("execute_noise_measure_task run error:" + std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("noise_params"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::arg("shots") = 1000,
            py::return_value_policy::reference)

        .def("execute_callback_noise_measure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const PilotQVM::PilotNoiseParams &noise_params, 
                std::function<void(PilotQVM::ErrorCode, std::map<std::string, double>&)>cb_func, 
                const uint32_t& chip_id, const uint32_t& shots)->PilotQVM::ErrorCode {
                return self.execute_noise_measure_task(prog_str, noise_params, cb_func, chip_id, shots);
            },
            py::arg("prog_str"),
            py::arg("noise_params"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::arg("shots") = 1000,
            py::return_value_policy::reference)

        .def("execute_partial_amplitude_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::vector<std::string>& target_amplitude_vec,
                const uint32_t& chip_id )->std::map<std::string, PilotQVM::Complex_> {
                std::map<std::string, PilotQVM::Complex_> result;
                auto ErrInfo = self.execute_partial_amplitude_task(prog_str, target_amplitude_vec, result, chip_id);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                throw::runtime_error("execute_partial_amplitude_task run error:" + std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("target_amplitude_vec"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)

        .def("execute_callback_partial_amplitude_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::vector<std::string>& target_amplitude_vec,
                std::function<void(PilotQVM::ErrorCode, const std::map<std::string, PilotQVM::Complex_>&)>cb_func, 
                const uint32_t& chip_id)->PilotQVM::ErrorCode {
                return self.execute_partial_amplitude_task(prog_str, target_amplitude_vec, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("target_amplitude_vec"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)

        .def("execute_single_amplitude_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::string& target_amplitude,
                const uint32_t& chip_id)->PilotQVM::Complex_ {
                PilotQVM::Complex_ result;
                auto ErrInfo = self.execute_single_amplitude_task(prog_str, target_amplitude, result, chip_id);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                    throw::runtime_error("execute_single_amplitude_task run error:" + std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("target_amplitude"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)

        .def("execute_callback_single_amplitude_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::string& target_amplitude,
                std::function<void(PilotQVM::ErrorCode, const PilotQVM::Complex_&)>cb_func, 
                const uint32_t& chip_id)->PilotQVM::ErrorCode {
                return self.execute_single_amplitude_task(prog_str, target_amplitude, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("target_amplitude"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)

        .def("execute_full_amplitude_expectation",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const PilotQVM::QuantumHamiltonianData& hamiltonian,
                const std::vector<uint32_t>& qubit_vec, const uint32_t& chip_id)->double {
                double result;
                auto ErrInfo = self.execute_full_amplitude_expectation(prog_str, hamiltonian, qubit_vec, result, chip_id);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                    throw::runtime_error("execute_full_amplitude_expectation run error:" + std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("hamiltonian"),
            py::arg("qubit_vec"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)

        .def("execute_callback_full_amplitude_expectation",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const PilotQVM::QuantumHamiltonianData& hamiltonian,
                const std::vector<uint32_t>& qubit_vec, std::function<void(PilotQVM::ErrorCode, double)> cb_func, 
                const uint32_t& chip_id)->PilotQVM::ErrorCode {
                return self.execute_full_amplitude_expectation(prog_str, hamiltonian, qubit_vec, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("hamiltonian"),
            py::arg("qubit_vec"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference);

#endif
#endif //DOCKER

    py::class_<QPanda::QPilotOSMachine, QPanda::QuantumMachine>(m, "QPilotOSService", "origin quantum pilot OS Machine")
        .def(py::init<std::string>(),
            py::arg("machine_type") = "CPU",
            py::return_value_policy::reference)

        .def("init",
            py::overload_cast<>(&QPanda::QPilotOSMachine::init),
            py::return_value_policy::automatic)

        .def("init_config", &QPanda::QPilotOSMachine::init_config,
            py::arg("url"),
            py::arg("log_cout"),
            py::return_value_policy::automatic)

        .def("get_token",
            &QPanda::QPilotOSMachine::get_token,
            py::arg("rep_json"),
            py::return_value_policy::automatic)

        .def("build_init_msg",
            &QPanda::QPilotOSMachine::buil_init_msg,
            py::arg("api_key"),
            py::return_value_policy::automatic)

        .def("build_task_msg",
            &QPanda::QPilotOSMachine::build_measure_task_msg,
            py::arg("prog"),
            py::arg("shot"),
            py::arg("chip_id"),
            py::arg("is_amend"),
            py::arg("is_mapping"),
            py::arg("is_optimization"),
            py::arg("specified_block"),
            py::arg("task_describe"),
            py::arg("point_lable"),
            py::arg("priority"),
            py::return_value_policy::automatic,
            "use c++ to build real chip measure task msg body.")

        .def("build_expectation_task_msg",
            &QPanda::QPilotOSMachine::build_expectation_task_msg,
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits") = std::vector<uint32_t>{},
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("task_describe") = std::string{},
            py::return_value_policy::automatic,
            "use C++ to build a expectation task body.")

        .def("build_qst_task_msg",
            [&](QPilotOSMachine& self, QProg& prog, const int shot, const int chip_id, const bool is_amend,
            const bool is_mapping, const bool is_optimization, const std::vector<uint32_t>& specified_block,
            const std::string& task_describe)
            {
                return self.build_qst_task_msg(prog, shot, chip_id, is_amend, is_mapping, is_optimization, specified_block, task_describe);
            },
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("task_describe") = std::string{},
            py::return_value_policy::automatic,
            "use C++ to build ordinary qst task msg body")

        .def("build_query_msg",
            &QPanda::QPilotOSMachine::build_query_msg,
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("tcp_recv",
            [&](QPanda::QPilotOSMachine& self, const std::string& ip, const unsigned short& port, const string& task_id)
            {
                std::string resp;
                bool is_ok = self.tcp_recv(ip, port, task_id, resp);
                py::list result_data_list;
                result_data_list.append(is_ok);
                result_data_list.append(resp);
                return result_data_list;
            },
            py::arg("ip"),
            py::arg("port"),
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("parser_sync_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& json_str)
            {
                std::vector<std::map<std::string, double>> result;
                self.parser_probability_result(json_str, result);
                /*py::list result_data_list;
                result_data_list.append(result);*/
                return result;
            },
            py::arg("json_str"),
            py::return_value_policy::automatic)

        .def("parser_expectation_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& json_str)
            {
                std::vector<double> result;
                self.parser_expectation_result(json_str, result);
                py::list result_data_list;
                result_data_list.append(result);
                return result_data_list;
            },
            py::arg("json_str"),
            py::return_value_policy::automatic,
            "deprecated, use Python's json lib.")

        .def("parse_task_result",
            [&](QPanda::QPilotOSMachine& self, const string& result_str) {
                std::map<std::string, double> result_val;
                self.parse_task_result(result_str, result_val);
                return result_val;
            },
            py::arg("result_str"),
            "Parse result str to map<string, double>\n"
            "Args:\n"
            "    result_str: Taeget result string\n"
            "\n"
            "Returns:\n"
            "    dict: map<string, double>\n"
            "Raises:\n"
            "    none\n",
            py::return_value_policy::automatic)

        .def("parse_probability_result",
            [&](QPanda::QPilotOSMachine& self, const std::vector<std::string>& result_str) {
                std::vector<std::map<std::string, double>> result_val;
                self.parse_task_result(result_str, result_val);
                return result_val;
            },
            py::arg("result_str"),
            "Parse result str to map<string, double>\n"
            "Args:\n"
            "    result_str: Taeget result string\n"
            "\n"
            "Returns:\n"
            "    array: vector<map<string, double>>\n"
            "Raises:\n"
            "    none\n",
            py::return_value_policy::automatic)

        .def("parse_prob_counts_result",
            [&](QPanda::QPilotOSMachine& self, const std::vector<std::string>& result_str) {
                std::vector<std::map<std::string, uint64_t>> result_val;
                self.parse_task_result(result_str, result_val);
                return result_val;
            },
            py::arg("result_str"),
            "Parse result str to map<string, double>\n"
            "Args:\n"
            "    result_str: Taeget result string\n"
            "\n"
            "Returns:\n"
            "    array: vector<map<string, double>>\n"
            "Raises:\n"
            "    none\n",
            py::return_value_policy::automatic)

        .def("finalize",
            &QPanda::QPilotOSMachine::finalize,
            "finalize")

        .def("qAlloc",
            &QPanda::QPilotOSMachine::allocateQubit,
            "Allocate a qubit",
            py::return_value_policy::reference)

        .def("qAlloc_many",
            [&](QPanda::QPilotOSMachine& self, size_t qubit_num)
            {
                auto qv = static_cast<std::vector<Qubit*>>(self.qAllocMany(qubit_num));
                return qv;
            },
            py::arg("qubit_num"),
            "Allocate a list of qubits",
            py::return_value_policy::reference)

        .def("cAlloc",
            py::overload_cast<>(&QPanda::QPilotOSMachine::allocateCBit),
            "Allocate a cbit",
            py::return_value_policy::reference)

        .def("cAlloc",
            py::overload_cast<size_t>(&QPanda::QPilotOSMachine::allocateCBit),
            py::arg("cbit"),
            "Allocate a cbit",
            py::return_value_policy::reference)

        .def("set_config",
            [](QPilotOSMachine &qvm, size_t max_qubit, size_t max_cbit)
            {
                Configuration config = { max_qubit, max_cbit };
                qvm.setConfig(config);
            },
            py::arg("max_qubit"),
            py::arg("max_cbit"),
            "set QVM max qubit and max cbit\n"
            "\n"
            "Args:\n"
            "    max_qubit: quantum machine max qubit num \n"
            "    max_cbit: quantum machine max cbit num \n"
            "\n"
            "Returns:\n"
            "    none\n"
            "Raises:\n"
            "    run_fail: An error occurred in set_configure\n")

        .def("cAlloc_many",
            &QPanda::QPilotOSMachine::allocateCBits,
            py::arg("cbit_num"),
            "Allocate a list of cbits",
            py::return_value_policy::reference)

        .def("qFree",
            &QPanda::QPilotOSMachine::Free_Qubit,
            py::arg("qubit"),
            "Free a qubit")

        .def("qFree_all",
            &QPanda::QPilotOSMachine::Free_Qubits,
            py::arg("qubit_list"),
            "Free a list of qubits")

        .def("qFree_all",
            py::overload_cast<QVec&>(&QPanda::QPilotOSMachine::qFreeAll),
            "Free all of qubits")

        .def("cFree",
            &QPanda::QPilotOSMachine::Free_CBit,
            "Free a cbit")

        .def("cFree_all",
            &QPanda::QPilotOSMachine::Free_CBits,
            py::arg("cbit_list"),
            "Free a list of cbits")

        .def("cFree_all",
            py::overload_cast<>(&QPanda::QPilotOSMachine::cFreeAll),
            "Free all of cbits")

#if /*defined(USE_OPENSSL) && */defined(USE_CURL)
        .def("init",
            py::overload_cast<std::string, bool, const std::string&>(&QPanda::QPilotOSMachine::init),
            py::arg("url"),
            py::arg("log_cout") = false,
            py::arg("api_key") = py::none(),
            py::return_value_policy::automatic)

        .def("init",
            py::overload_cast<std::string, bool, const std::string&, const std::string&>
            (&QPanda::QPilotOSMachine::init),
            py::arg("url"),
            py::arg("log_cout") = false,
            py::arg("username") = py::none(),
            py::arg("password") = py::none(),
            py::return_value_policy::automatic)

        .def("output_version",
            &QPanda::QPilotOSMachine::OutputVersionInfo,
            py::return_value_policy::automatic)

        .def("pMeasureBinindex",
            &QPanda::QPilotOSMachine::pMeasureBinindex,
            py::arg("prog"),
            py::arg("index"),
            py::arg("backendID") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::automatic)

        .def("pMeasureDecindex",
            &QPanda::QPilotOSMachine::pMeasureDecindex,
            py::arg("prog"),
            py::arg("index"),
            py::arg("backendID") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::automatic)

        .def("pmeasure_subset",
            &QPanda::QPilotOSMachine::pmeasure_subset,
            py::arg("prog"),
            py::arg("amplitude"),
            py::arg("backendID") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::automatic)

        .def("real_chip_expectation",
            &QPanda::QPilotOSMachine::real_chip_expectation,
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits") = std::vector<uint32_t>{},
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::return_value_policy::automatic,
            "deprecated, use request instead.")

        .def("async_real_chip_expectation",
            &QPanda::QPilotOSMachine::async_real_chip_expectation,
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits") = std::vector<uint32_t>{},
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::return_value_policy::automatic,
            "deprecated, use request instead.")

        .def("get_expectation_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& task_id) {
                double expectation;
                PilotQVM::ErrorCode err_code;
                std::string err_info;
                self.get_expectation_result(task_id, expectation, err_code, err_info);
                py::list result_data_list;
                result_data_list.append(expectation);
                result_data_list.append(err_code);
                result_data_list.append(err_info);
                return result_data_list;
            },
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("real_chip_measure",
            py::overload_cast<const QProg&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure",
            py::overload_cast<const std::string&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure",
            py::overload_cast<const std::vector<QProg>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_vec),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure",
            py::overload_cast<const std::vector<std::string>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_vec),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure",
            py::overload_cast<const std::vector<QProg>&, const std::string&>
            (&QPanda::QPilotOSMachine::real_chip_measure),
            py::arg("prog"),
            py::arg("config_str"),
            py::return_value_policy::automatic)

        .def("real_chip_measure_vec",
            py::overload_cast<const std::vector<QProg>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_vec),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure_vec",
            py::overload_cast<const std::vector<std::string>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_vec),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure_prob_count",
            py::overload_cast<const std::string&, const int, const int, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_prob_count),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure_prob_count",
            py::overload_cast<const QProg&, const int, const int, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_prob_count),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure_prob_count",
            py::overload_cast<const std::vector<std::string>&, const int, const int, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_prob_count),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("real_chip_measure_prob_count",
            py::overload_cast<const std::vector<QProg>&, const int, const int, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::real_chip_measure_prob_count),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_measure",
            py::overload_cast<const QProg&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const bool, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_measure),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("is_prob_counts") = true,
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_measure",
            py::overload_cast<const std::string&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const bool, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_measure),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("is_prob_counts") = true,
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_measure",
            py::overload_cast<const std::vector<QProg>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const bool, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_measure_vec),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("is_prob_counts") = true,
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_measure",
            py::overload_cast<const std::vector<std::string>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const bool, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_measure_vec),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("is_prob_counts") = true,
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_measure",
            py::overload_cast<const std::vector<QProg>&, const std::string&>
            (&QPanda::QPilotOSMachine::async_real_chip_measure),
            py::arg("prog"),
            py::arg("config_str"),
            py::return_value_policy::automatic)

        .def("async_real_chip_measure_vec",
            py::overload_cast<const std::vector<QProg>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const bool, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_measure_vec),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("is_prob_counts") = true,
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_measure_vec",
            py::overload_cast<const std::vector<std::string>&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const bool, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_measure_vec),
            py::arg("ir"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("is_prob_counts") = true,
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_qst",
            py::overload_cast<const QProg&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_QST),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_qst_density",
            py::overload_cast<const QProg&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_QST_density),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("async_real_chip_qst_fidelity",
            py::overload_cast<const QProg&, const int, const int, const bool, const bool, const bool, const std::vector<uint32_t>&, const std::string&, const int>
            (&QPanda::QPilotOSMachine::async_real_chip_QST_fidelity),
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("specified_block") = std::vector<uint32_t>{},
            py::arg("describe") = "",
            py::arg("point_lable") = 0,
            py::return_value_policy::automatic)

        .def("noise_learning",
            py::overload_cast<const std::string&>
            (&QPanda::QPilotOSMachine::noise_learning),
            py::arg("parameter_json") = true,
            py::return_value_policy::automatic)

        .def("async_em_compute",
            py::overload_cast<const std::string&>
            (&QPanda::QPilotOSMachine::async_em_compute),
            py::arg("parameter_json"),
            py::return_value_policy::automatic)

        .def("em_compute",
            py::overload_cast<const std::string&>
            (&QPanda::QPilotOSMachine::em_compute),
            py::arg("parameter_json"),
            py::return_value_policy::automatic)

        .def("query_task_state",
            [&](QPanda::QPilotOSMachine& self, const string& task_id) {
                PilotQVM::PilotTaskQueryResult query_result;
                self.query_task_state(task_id, query_result);

                py::list result_data_list;
                result_data_list.append(query_result.m_state);
                result_data_list.append(query_result.m_result_vec);
                result_data_list.append(query_result.m_errCode);
                result_data_list.append(query_result.m_errInfo);
                return result_data_list;
            },
            py::arg("task_id"),
            "Query Task State by task_id\n"
            "Args:\n"
            "    task_id: Taeget task id, got by async_real_chip_measure\n"
            "\n"
            "Returns:\n"
            "    string: task state: 2: Running; 3: Finished; 4: Failed\n"
            "    string: task result string\n"
            "Raises:\n"
            "    none\n",
            py::return_value_policy::automatic)

        .def("query_task_state_vec",
            [&](QPanda::QPilotOSMachine& self, const string& task_id) {
                PilotQVM::PilotTaskQueryResult query_result;
                self.query_task_state(task_id, query_result);

                py::list result_data_list;
                result_data_list.append(query_result.m_state);
                result_data_list.append(query_result.m_result_vec);
                result_data_list.append(query_result.m_errCode);
                result_data_list.append(query_result.m_errInfo);
                return result_data_list;
            },
            py::arg("task_id"),
            "Query Task State by task_id\n"
            "Args:\n"
            "    task_id: Taeget task id, got by async_real_chip_measure\n"
            "\n"
            "Returns:\n"
            "    string: task state: 2: Running; 3: Finished; 4: Failed\n"
            "    array: task result array\n"
            "Raises:\n"
            "    none\n",
            py::return_value_policy::automatic)

        .def("query_task_state",
            [&](QPanda::QPilotOSMachine& self, const string& task_id, bool is_save, std::string& file_path) {
                PilotQVM::PilotTaskQueryResult query_result;
                self.query_task_state(task_id, query_result, is_save, file_path);

                py::list result_data_list;
                result_data_list.append(query_result.m_state);
                result_data_list.append(query_result.m_result_vec);
                result_data_list.append(query_result.m_errCode);
                result_data_list.append(query_result.m_errInfo);
                return result_data_list;
            },
            py::arg("task_id"),
            py::arg("is_save"),
            py::arg("file_path") = "",
            py::return_value_policy::automatic)

        .def("get_measure_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& task_id) {
                std::vector<std::map<std::string, double>> result;
                PilotQVM::ErrorCode err_code;
                std::string err_info;
                self.get_measure_result(task_id, result, err_code, err_info);
                py::list result_data_list;
                result_data_list.append(result);
                result_data_list.append(err_code);
                result_data_list.append(err_info);
                return result_data_list;
            },
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("get_measure_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& task_id) {
                std::vector<std::map<std::string, uint64_t>> result;
                PilotQVM::ErrorCode err_code;
                std::string err_info;
                self.get_measure_result(task_id, result, err_code, err_info);
                py::list result_data_list;
                result_data_list.append(result);
                result_data_list.append(err_code);
                result_data_list.append(err_info);
                return result_data_list;
            },
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("_get_qst_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& task_id) {
                std::vector<std::map<std::string, double>> result;
                PilotQVM::ErrorCode err_code;
                std::string err_info;
                self.get_qst_result(task_id, result, err_code, err_info);
                py::list result_data_list;
                result_data_list.append(result);
                result_data_list.append(err_code);
                result_data_list.append(err_info);
                return result_data_list;
            },
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("_get_qst_density_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& task_id) {
                std::vector<std::map<std::string, double>> density;
                PilotQVM::ErrorCode err_code;
                std::string err_info;
                self.get_qst_density_result(task_id, density, err_code, err_info);
                py::list result_data_list;
                result_data_list.append(density);
                result_data_list.append(err_code);
                result_data_list.append(err_info);
                return result_data_list;
            },
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("_get_qst_fidelity_result",
            [&](QPanda::QPilotOSMachine& self, const std::string& task_id) {
                double fidelity;
                PilotQVM::ErrorCode err_code;
                std::string err_info;
                self.get_qst_fidelity_result(task_id, fidelity, err_code, err_info);
                py::list result_data_list;
                result_data_list.append(fidelity);
                result_data_list.append(err_code);
                result_data_list.append(err_info);
                return result_data_list;
            },
            py::arg("task_id"),
            py::return_value_policy::automatic)

        .def("query_compile_prog",
            [&](QPanda::QPilotOSMachine& self, const std::string task_id, bool& without_compensate) {
                std::string errCode;
                std::string errInfo;
                std::string compile_prog;
                self.query_compile_prog(task_id, compile_prog, without_compensate);

                py::list result_data_list;
                result_data_list.append(errCode);
                result_data_list.append(errInfo);
                result_data_list.append(compile_prog);
                return result_data_list;
            },
            py::arg("task_id"),
                "Query Task compile prog by task_id\n"
                "Args:\n"
                "    without_compensate: whether return the prog without angle compensate\n"
                "\n"
                "Returns:\n"
                "    bool: whether find compile prog success\n"
                "Raises:\n"
                "    none\n",
                py::arg("without_compensate") = true,
                py::return_value_policy::automatic)

        .def("probRunDict",
            &QPanda::QPilotOSMachine::probRunDict,
            py::arg("prog"),
            py::arg("qubit_vec"),
            py::arg("backendID") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::automatic)

        .def("runWithConfiguration",
            py::overload_cast<QProg&, int, const uint32_t &, const QPanda::NoiseModel&>(&QPanda::QPilotOSMachine::runWithConfiguration),
            py::arg("prog"),
            py::arg("shots") = 1000,
            py::arg("backend_id") = ANY_CLUSTER_BACKEND,
            py::arg("noise_model") = QPanda::NoiseModel(),
            py::return_value_policy::automatic)
#endif
            ;
#endif
    return;
}


void export_extension_funtion(py::module& m)
{
#ifdef USE_EXTENSION
    m.def("expand_linear_equations",
        [](QStat& A, std::vector<double>& b)
        {
            HHLAlg::expand_linear_equations(A, b);

            py::list linear_equations_data;
            linear_equations_data.append(A);
            linear_equations_data.append(b);
            return linear_equations_data;
        },
        py::arg("matrix"),
        py::arg("list"),
        "Extending linear equations to N dimension, N = 2 ^ n\n"
        "\n"
        "Args:\n"
        "    matrix: the source matrix, which will be extend to N*N, N = 2 ^ n\n"
        "\n"
        "    list: the source vector b, which will be extend to 2 ^ n",
        py::return_value_policy::automatic_reference);

    m.def("build_HHL_circuit",
        &build_HHL_circuit,
        py::arg("matrix_A"),
        py::arg("data_b"),
        py::arg("qvm"),
        py::arg("precision_cnt") = 0,
        "build the quantum circuit for HHL algorithm to solve the target linear systems of equations : Ax = b\n"
        "\n"
        "Args:\n"
        "    matrix_A: a unitary matrix or Hermitian N*N matrix with N = 2 ^ n\n"
        "\n"
        "    data_b: a given vector\n"
        "\n"
        "    qvm: quantum machine\n"
        "\n"
        "    precision_cnt: The count of digits after the decimal point,\n"
        "                   default is 0, indicates that there are only integer solutions\n"
        "\n"
        "Returns:\n"
        "    QCircuit The whole quantum circuit for HHL algorithm\n"
        "\n"
        "Notes:\n"
        "    The higher the precision is, the more qubit number and circuit - depth will be,\n"
        "    for example: 1 - bit precision, 4 additional qubits are required,\n"
        "    for 2 - bit precision, we need 7 additional qubits, and so on.\n"
        "    The final solution = (HHL result) * (normalization factor for b) * (1 << ceil(log2(pow(10, precision_cnt))))",
        py::return_value_policy::automatic);

    m.def("HHL_solve_linear_equations",
        &HHL_solve_linear_equations,
        py::arg("matrix_A"),
        py::arg("data_b"),
        py::arg("precision_cnt") = 0,
        "Use HHL algorithm to solve the target linear systems of equations : Ax = b\n"
        "\n"
        "Args:\n"
        "    matrix_A: a unitary matrix or Hermitian N*N matrix with N = 2 ^ n\n"
        "\n"
        "    data_b: a given vector\n"
        "\n"
        "    precision_cnt: The count of digits after the decimal point\n"
        "                   default is 0, indicates that there are only integer solutions.\n"
        "\n"
        "Returns:\n"
        "    QStat The solution of equation, i.e.x for Ax = b\n"
        "\n"
        "Notes:\n"
        "    The higher the precision is, the more qubit number and circuit - depth will be,\n"
        "    for example: 1 - bit precision, 4 additional qubits are required,\n"
        "    for 2 - bit precision, we need 7 additional qubits, and so on.",
        py::return_value_policy::automatic);

    m.def("sabre_mapping",
        [](QProg prog,
           QuantumMachine *quantum_machine,
           QVec& qv,
           uint32_t max_look_ahead = 20,
           uint32_t max_iterations = 10,
           const std::string& config_data = CONFIG_PATH,
           uint32_t m_max_random_mappings = 10000,
           uint32_t hops = 1)
        {
            QMappingConfig mapping_config = QMappingConfig(config_data);
            auto ret_prog = SABRE_mapping(prog, 
                quantum_machine, 
                qv, 
                max_look_ahead, 
                max_iterations, 
                mapping_config,
                m_max_random_mappings,
                hops);

            return ret_prog;
        },
        py::arg("prog"),
        py::arg("quantum_machine"),
        py::arg("init_map"),
        py::arg("max_look_ahead") = 20,
        py::arg("max_iterations") = 10,
        py::arg("config_data") = CONFIG_PATH,
        py::arg("m_max_random_mappings") = 10000,
        py::arg("hops") = 1,
        "sabre mapping\n"
        "\n"
        "Args:\n"
        "    prog: the target prog\n"
        "\n"
        "    quantum_machine: quantum machine\n"
        "\n"
        "    init_map: init map\n"
        "\n"
        "    max_look_ahead: sabre_mapping max_look_ahead\n"
        "\n"
        "    max_iterations: sabre_mapping max_iterations\n"
        "\n"
        "    config_data: config data, @See JsonConfigParam::load_config()\n"
        "    m_max_random_mappings: m_max_random_mappings data\n"
        "    hops: hops data\n"
        "\n"
        "Returns:\n"
        "    mapped quantum program",
        py::return_value_policy::automatic);

    m.def("sabre_mapping",
        [](QProg prog,
            QuantumMachine *quantum_machine,
            uint32_t max_look_ahead = 20,
            uint32_t max_iterations = 10,
            const std::string& config_data = CONFIG_PATH)
        {
            QMappingConfig mapping_config = QMappingConfig(config_data);
            QVec qv;
            auto ret_prog = SABRE_mapping(prog, quantum_machine, qv, max_look_ahead, max_iterations, mapping_config);
            return ret_prog;
        },
        py::arg("prog"),
        py::arg("quantum_machine"),
        py::arg("max_look_ahead") = 20,
        py::arg("max_iterations") = 10,
        py::arg("config_data") = CONFIG_PATH,
        "sabre mapping\n"
        "\n"
        "Args:\n"
        "    prog: the target prog\n"
        "\n"
        "    quantum_machine: quantum machine\n"
        "\n"
        "    max_look_ahead: sabre_mapping max_look_ahead\n"
        "\n"
        "    max_iterations: sabre_mapping max_iterations\n"
        "\n"
        "    config_data: config data, @See JsonConfigParam::load_config()\n"
        "\n"
        "Returns:\n"
        "    mapped quantum program",
        py::return_value_policy::automatic);

    m.def("sabre_mapping",
        [](QProg prog,
            QuantumMachine *quantum_machine,
            uint32_t max_look_ahead,
            uint32_t max_iterations,
            const dmatrix_t& arch_matrix)
        {
            QMappingConfig mapping_config = QMappingConfig(arch_matrix);
            QVec qv;
            auto ret_prog = SABRE_mapping(prog, quantum_machine, qv, max_look_ahead, max_iterations, mapping_config);
            return ret_prog;
        },
        py::arg("prog"),
        py::arg("quantum_machine"),
        py::arg("max_look_ahead"),
        py::arg("max_iterations"),
        py::arg("arch_matrix"),
        "sabre mapping\n"
        "\n"
        "Args:\n"
        "    prog: the target prog\n"
        "\n"
        "    quantum_machine: quantum machine\n"
        "\n"
        "    max_look_ahead: sabre_mapping max_look_ahead, default is 20 \n"
        "\n"
        "    max_iterations: sabre_mapping max_iterations, default is 10 \n"
        "\n"
        "    arch_matrix: arch matrix\n"
        "\n"
        "Returns:\n"
        "    mapped quantum program",
        py::return_value_policy::automatic);

    m.def("OBMT_mapping",
        [](QPanda::QProg prog,
            QPanda::QuantumMachine* quantum_machine,
            bool optimization = false,
            uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(),
            uint32_t max_children = (std::numeric_limits<uint32_t>::max)(),
            const std::string& config_data = CONFIG_PATH)
        {
            QVec qv;
            auto ret_prog = OBMT_mapping(prog, 
                quantum_machine, 
                qv, 
                optimization, 
                max_partial, 
                max_children, 
                QMappingConfig(config_data));
            return ret_prog;
        },
        py::arg("prog"),
        py::arg("quantum_machine"),
        py::arg("b_optimization") = false,
        py::arg("max_partial") = (std::numeric_limits<uint32_t>::max)(),
        py::arg("max_children") = (std::numeric_limits<uint32_t>::max)(),
        py::arg("config_data") = CONFIG_PATH,
        "OPT_BMT mapping\n"
        "\n"
        "Args:\n"
        "    prog: the target prog\n"
        "\n"
        "    quantum_machine: quantum machine\n"
        "\n"
        "    b_optimization: whether open the optimization\n"
        "\n"
        "    max_partial: Limits the max number of partial solutions per step, There is no limit by default\n"
        "\n"
        "    max_children: Limits the max number of candidate - solutions per double gate, There is no limit by default\n"
        "\n"
        "    config_data: config data, @See JsonConfigParam::load_config()\n"
        "\n"
        "Returns:\n"
        "    mapped quantum program",
        py::return_value_policy::automatic);

    m.def(
        "OBMT_mapping",
        [](QPanda::QProg prog,
            QPanda::QuantumMachine* quantum_machine,
            bool optimization,
            const dmatrix_t& arch_matrix)
        {
            QVec qv;
            QMappingConfig mapping_config = QMappingConfig(arch_matrix);
            uint32_t max_partial = (std::numeric_limits<uint32_t>::max)();
            uint32_t max_children = (std::numeric_limits<uint32_t>::max)();
            auto ret_prog = OBMT_mapping(prog, quantum_machine, qv, optimization, max_partial, max_children, mapping_config);
            return ret_prog;
        },
        py::arg("prog"),
        py::arg("quantum_machine"),
        py::arg("b_optimization"),
        py::arg("arch_matrix"),
        "OPT_BMT mapping\n"
        "\n"
        "Args:\n"
        "    prog: the target prog\n"
        "\n"
        "    quantum_machine: quantum machine\n"
        "\n"
        "    b_optimization: whether open the optimization\n"
        "\n"
        "    arch_matrix: arch graph matrix\n"
        "\n"
        "Returns:\n"
        "    mapped quantum program",
        py::return_value_policy::automatic);

    m.def("virtual_z_transform",
        [](QPanda::QProg &prog, QPanda::QuantumMachine* quantum_machine, bool b_del_rz_gate = false, const std::string& config_data = CONFIG_PATH)
        {

            virtual_z_transform(prog, quantum_machine, b_del_rz_gate, config_data);
            return prog;

        },
        py::arg("prog"),
        py::arg("quantum_machine"),
        py::arg("b_del_rz_gate") = false,
        py::arg("config_data") = CONFIG_PATH,
        "virtual z transform\n"
        "\n"
        "Args:\n"
        "    prog: the target prog\n"
        "\n"
        "    quantum_machine: quantum machine\n"
        "\n"
        "    b_del_rz_gate: whether delete the rz gate \n"
        "\n"
        "    config_data: config data, @See JsonConfigParam::load_config()\n"
        "\n"
        "Returns:\n"
        "    mapped quantum program",
        py::return_value_policy::automatic);

    m.def("matrix_decompose_paulis",
        [](QuantumMachine* qvm, EigenMatrixX& mat)
        {
            PualiOperatorLinearCombination linearcom;
            matrix_decompose_paulis(qvm, mat, linearcom);
            return linearcom;
        },
        "decompose matrix into paulis combination\n"
            "\n"
            "Args:\n"
            "    quantum_machine: quantum machine\n"
            "\n"
            "    matrix: 2^N *2^N double matrix \n"
            "\n"
            "Returns:\n"
            "    result : linearcom contains pauli circuit", 
        py::return_value_policy::automatic);

    m.def("matrix_decompose_paulis",
        [](QVec qubits, EigenMatrixX& mat)
        {
            PualiOperatorLinearCombination linearcom;
            matrix_decompose_paulis(qubits, mat, linearcom);
            return linearcom;
        },
        "decompose matrix into paulis combination\n"
            "\n"
            "Args:\n"
            "    quantum_machine: quantum machine\n"
            "    matrix: 2^N *2^N double matrix \n"
            "\n"
            "Returns:\n"
            "    result : linearcom contains pauli circuit", 
        py::return_value_policy::automatic);

    m.def("pauli_combination_replace", &pauli_combination_replace, py::return_value_policy::automatic);

    m.def("transfrom_pauli_operator_to_matrix",
        [](const PauliOperator& opt)
        {
            return transPauliOperatorToMatrix(opt);
        },
        "transfrom pauli operator to matrix\n"
            "\n"
            "Args:\n"
            "    matrix: 2^N *2^N double matrix \n"
            "\n"
            "Returns:\n"
            "    result : hamiltonian", 
        py::return_value_policy::automatic);

    m.def("matrix_decompose",
        [](QVec& qubits, QMatrixXcd& src_mat, const DecompositionMode mode, bool b_positive_seq)
        {
            QStat mat = Eigen_to_QStat(src_mat);
            switch (mode)
            {
            case DecompositionMode::QR:
                return matrix_decompose_qr(qubits, mat, b_positive_seq);
                // break;
            case DecompositionMode::HOUSEHOLDER_QR:
                return matrix_decompose_householder(qubits, mat, b_positive_seq);
            case DecompositionMode::QSD:
                return unitary_decomposer_nq(src_mat, qubits, mode, true);
                // break;
            case DecompositionMode::CSD:
                return unitary_decomposer_nq(src_mat, qubits, DecompositionMode::CSD, true);
            default:
                throw std::runtime_error("Error: DecompositionMode");
            }
        },
        py::arg("qubits"),
        py::arg("matrix"),
        py::arg_v("mode", DecompositionMode::QSD, "DecompositionMode.QSD"),
        py::arg("b_positive_seq") = true,
        "Matrix decomposition\n"
        "\n"
        "Args:\n"
        "    qubits: the used qubits\n"
        "\n"
        "    matrix: The target matrix\n"
        "\n"
        "    mode: DecompositionMode decomposition mode, default is QSD\n"
        "\n"
        "    b_positive_seq: true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is true\n"
        "\n"
        "Returns:\n"
        "    QCircuit The quantum circuit for target matrix",
        py::return_value_policy::automatic);

    m.def(
        "matrix_decompose",
        [](QVec& qubits, QStat& src_mat, const DecompositionMode mode, bool b_positive_seq)
        {
            QMatrixXcd mat = QStat_to_Eigen(src_mat);
            switch (mode)
            {
            case DecompositionMode::QR:
                return matrix_decompose_qr(qubits, src_mat, b_positive_seq);
                // break;
            case DecompositionMode::HOUSEHOLDER_QR:
                return matrix_decompose_householder(qubits, src_mat, b_positive_seq);
            case DecompositionMode::QSD:
                return unitary_decomposer_nq(mat, qubits, DecompositionMode::QSD, true);
                // break;
            case DecompositionMode::CSD:
                return unitary_decomposer_nq(mat, qubits, DecompositionMode::CSD, true);
            default:
                throw std::runtime_error("Error: DecompositionMode");
            }
        },
        py::arg("qubits"),
        py::arg("matrix"),
        py::arg_v("mode", DecompositionMode::QSD, "DecompositionMode.QSD"),
        py::arg("b_positive_seq") = true,
        "Matrix decomposition\n"
        "\n"
        "Args:\n"
        "    qubits: the used qubits\n"
        "\n"
        "    matrix: The target matrix\n"
        "\n"
        "    mode: DecompositionMode decomposition mode, default is QSD\n"
        "\n"
        "    b_positive_seq: true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is true\n"
        "\n"
        "Returns:\n"
        "    QCircuit The quantum circuit for target matrix",
        py::return_value_policy::automatic);

    m.def(
        "expand_linear_equations",
        [](QStat& A, std::vector<double>& b)
        {
            HHLAlg::expand_linear_equations(A, b);

            py::list linear_equations_data;
            linear_equations_data.append(A);
            linear_equations_data.append(b);
            return linear_equations_data;
        },
        py::arg("matrix"),
        py::arg("list"),
        "Extending linear equations to N dimension, N = 2 ^ n\n"
        "\n"
        "Args:\n"
        "    matrix: the source matrix, which will be extend to N*N, N = 2 ^ n\n"
        "\n"
        "    list: the source vector b, which will be extend to 2 ^ n",
        py::return_value_policy::automatic_reference);

#endif
}
