#include "QPandaConfig.h"
#include "QPanda.h"
#include "Extensions/Extensions.h"
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


#ifdef USE_EXTENSION
#include "Extensions/EigenTensor/EigenTensor.h"
#include "Extensions/VirtualZTransfer/VirtualZTransfer.h"
#include "Extensions/PilotOSMachine/QPilotMachine.h"
#endif

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA



void export_extension_class(py::module& m)
{
#ifdef USE_EXTENSION
    py::class_<HHLAlg>(m, "HHLAlg"," quantum hhl algorithm class")
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

#ifdef USE_CURL
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
    py::class_<PilotQVM::QPilotMachine>(m, "QPilotMachine", "pliot machine")
        .def(py::init(),
            py::return_value_policy::reference)
        .def("init",
            &PilotQVM::QPilotMachine::init,
            py::arg("pilot_url"),
            py::arg("log_cout") = false,
            py::return_value_policy::automatic)
        .def("init_withconfig",
            &PilotQVM::QPilotMachine::init_withconfig,
            py::arg("config_path")="pilotmachine.conf",
            py::return_value_policy::automatic)
        .def("execute_measure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const uint32_t& chip_id = ANY_CLUSTER_BACKEND, const bool& b_mapping=true, const bool& b_optimization=true, const uint32_t& shots = 1000)->std::map<std::string, double>{
                std::map<std::string, double> result;
                auto ErrInfo = self.execute_measure_task(prog_str, result, chip_id, b_mapping, b_optimization, shots);
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
            py::return_value_policy::reference)
        .def("execute_callback_measure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, std::function<void(PilotQVM::ErrorCode, const std::map<std::string, double>&)>cb_func, const uint32_t& chip_id = ANY_CLUSTER_BACKEND, const bool& b_mapping = true, const bool& b_optimization = true, const uint32_t& shots = 1000)->PilotQVM::ErrorCode {
                return self.execute_measure_task(prog_str, cb_func, chip_id, b_mapping, b_optimization, shots);
            },
            py::arg("prog_str"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("b_mapping") = true,
            py::arg("b_optimization") = true,
            py::arg("shots") = 1000,
            py::return_value_policy::reference)
        .def("execute_full_amplitude_measure_task",
            [&](PilotQVM::QPilotMachine &self,const std::string& prog_str, const uint32_t& chip_id = ANY_CLUSTER_BACKEND, const uint32_t& shots = 1000)->std::map<std::string, double>{
                std::map<std::string, double> result;
                auto ErrInfo = self.execute_full_amplitude_measure_task(prog_str, result, chip_id, shots);
                if (ErrInfo != PilotQVM::ErrorCode::NO_ERROR_FOUND)
                {
                    throw::runtime_error("execute_full_amplitude_measure_task run error:"+std::to_string(uint32_t(ErrInfo)));
                }
                return result;
            },
            py::arg("prog_str"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::arg("shots") = 1000,
            py::return_value_policy::reference)
        .def("execute_callback_full_amplitude_measure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, std::function<void(PilotQVM::ErrorCode, const std::map<std::string, double>&)>cb_func, const uint32_t& chip_id = ANY_CLUSTER_BACKEND, const uint32_t& shots = 1000)->PilotQVM::ErrorCode {
                return self.execute_full_amplitude_measure_task(prog_str, cb_func, chip_id, shots);
            },
            py::arg("prog_str"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::arg("shots") = 1000,
            py::return_value_policy::reference)
        .def("execute_full_amplitude_pmeasure_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::vector<uint32_t>& qubit_vec, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->std::map<std::string, double>{
                std::map<std::string, double> result;
                auto ErrInfo = self.execute_full_amplitude_pmeasure_task(prog_str, qubit_vec,result, chip_id);
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
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::vector<uint32_t>& qubit_vec, std::function<void(PilotQVM
                ::ErrorCode, const std::map<std::string, double>&)>cb_func, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->PilotQVM::ErrorCode {
                return self.execute_full_amplitude_pmeasure_task(prog_str, qubit_vec, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("qubit_vec"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)
        .def("build_noise_params",
            [&](PilotQVM::QPilotMachine& self, const uint32_t& nose_model_type,const std::vector<double>& single_params,const std::vector<double>& double_params)->PilotQVM::PilotNoiseParams{
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
            [&](PilotQVM::QPilotMachine& self, std::string prog_str, const PilotQVM::PilotNoiseParams& noise_params, uint32_t chip_id, uint32_t shots)->std::map<std::string, double> {
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
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const PilotQVM::PilotNoiseParams &noise_params, std::function<void(PilotQVM::ErrorCode, std::map<std::string, double>&)>cb_func, const uint32_t& chip_id = ANY_CLUSTER_BACKEND, const uint32_t& shots = 1000)->PilotQVM::ErrorCode {
                return self.execute_noise_measure_task(prog_str, noise_params, cb_func, chip_id, shots);
            },
            py::arg("prog_str"),
                py::arg("noise_params"),
                py::arg("cb_func"),
                py::arg("chip_id") = ANY_CLUSTER_BACKEND,
                py::arg("shots") = 1000,
                py::return_value_policy::reference)
        .def("execute_partial_amplitude_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::vector<std::string>& target_amplitude_vec, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->std::map<std::string, PilotQVM::Complex_>{
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
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::vector<std::string>& target_amplitude_vec, std::function<void(PilotQVM::ErrorCode, const std::map<std::string, PilotQVM::Complex_>&)>cb_func, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->PilotQVM::ErrorCode {
                return self.execute_partial_amplitude_task(prog_str, target_amplitude_vec, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("target_amplitude_vec"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)
        .def("execute_single_amplitude_task",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::string& target_amplitude, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->PilotQVM::Complex_{
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
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const std::string& target_amplitude, std::function<void(PilotQVM::ErrorCode, const PilotQVM::Complex_&)>cb_func, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->PilotQVM::ErrorCode {
                return self.execute_single_amplitude_task(prog_str, target_amplitude, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("target_amplitude"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference)
        .def("execute_full_amplitude_expectation",
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const PilotQVM::QuantumHamiltonianData& hamiltonian, const std::vector<uint32_t>& qubit_vec, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->double{
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
            [&](PilotQVM::QPilotMachine& self, const std::string& prog_str, const PilotQVM::QuantumHamiltonianData& hamiltonian, const std::vector<uint32_t>& qubit_vec, std::function<void(PilotQVM::ErrorCode, double)> cb_func, const uint32_t& chip_id = ANY_CLUSTER_BACKEND)->PilotQVM::ErrorCode{
                return self.execute_full_amplitude_expectation(prog_str, hamiltonian, qubit_vec, cb_func, chip_id);
            },
            py::arg("prog_str"),
            py::arg("hamiltonian"),
            py::arg("qubit_vec"),
            py::arg("cb_func"),
            py::arg("chip_id") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::reference);
#endif //DOCKER
    py::class_<QPanda::QPilotOSMachine>(m, "QPilotOSMachine","origin quantum pilot OS Machine")
        .def(py::init<std::string>(),
            py::arg("machine_type")="CPU",
            py::return_value_policy::reference)
        .def("init",
            &QPanda::QPilotOSMachine::init,
            py::arg("url") = "",
            py::arg("log_cout") = false,
            py::return_value_policy::automatic)
        .def("init_qvm", 
            &QPanda::QPilotOSMachine::init,
            py::arg("url") = "",
            py::arg("log_cout") = false,
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
        .def("real_chip_measure",
            &QPanda::QPilotOSMachine::real_chip_measure,
            py::arg("prog"),
            py::arg("shot") = 1000,
            py::arg("chip_id") = ANY_QUANTUM_CHIP,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::return_value_policy::automatic)
        .def("probRunDict",
            &QPanda::QPilotOSMachine::probRunDict,
            py::arg("prog"),
            py::arg("qubit_vec"),
            py::arg("backendID") = ANY_CLUSTER_BACKEND,
            py::return_value_policy::automatic)
        .def("runWithConfiguration",
            py::overload_cast<QProg& ,int , const uint32_t &, const QPanda::NoiseModel&>(&QPanda::QPilotOSMachine::runWithConfiguration),
            py::arg("prog"),
            py::arg("shots"),
            py::arg("backend_id"),
            py::arg("noise_model") = QPanda::NoiseModel(),
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
            "Free all of cbits");
#endif

#endif
    return;
}

void export_extension_funtion(py::module& m)
{
#ifdef USE_EXTENSION
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
        "    data_b: a given vector\n"
        "    qvm: quantum machine\n"
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
        "    data_b: a given vector\n"
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

    m.def(
        "OBMT_mapping",
        [](QPanda::QProg prog, QPanda::QuantumMachine* quantum_machine, bool optimization = false, uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(), uint32_t max_children = (std::numeric_limits<uint32_t>::max)(), const std::string& config_data = CONFIG_PATH)
        {
            QVec qv;
            auto ret_prog = OBMT_mapping(prog, quantum_machine, qv, optimization, max_partial, max_children, config_data);
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
            "    quantum_machine: quantum machine\n"
            "    b_optimization: whether open the optimization\n"
            "    max_partial: Limits the max number of partial solutions per step, There is no limit by default\n"
            "    max_children: Limits the max number of candidate - solutions per double gate, There is no limit by default\n"
            "    config_data: config data, @See JsonConfigParam::load_config()\n"
            "\n"
            "Returns:\n"
            "    mapped quantum program",
            py::return_value_policy::automatic);

    m.def("virtual_z_transform",
        py::overload_cast<QProg&, QuantumMachine*, const bool, const std::string&>(&virtual_z_transform),
        py::arg("prog"),
        py::arg("quantum_machine"),
        py::arg("b_del_rz_gate") = false,
        py::arg("config_data") = CONFIG_PATH,
        "virtual z transform\n"
        "\n"
        "Args:\n"
        "    prog: the target prog\n"
        "    quantum_machine: quantum machine\n"
        "    b_del_rz_gate: whether delete the rz gate \n"
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
        "    matrix: 2^N *2^N double matrix \n"
        "\n"
        "Returns:\n"
        "    result : linearcom contains pauli circuit", py::return_value_policy::automatic);

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
        "    result : linearcom contains pauli circuit", py::return_value_policy::automatic);

    m.def("pauli_combination_replace",&pauli_combination_replace, py::return_value_policy::automatic);

    m.def("transfrom_pauli_operator_to_matrix",
        [](const PauliOperator& opt)
        {
            return transPauliOperatorToMatrix(opt);
        },
        "transfrom pauli operator to matrix\n"
        "\n"
        "Args:\n"
        "    quantum_machine: quantum machine\n"
        "    matrix: 2^N *2^N double matrix \n"
        "\n"
        "Returns:\n"
        "    result : hamiltonian", py::return_value_policy::automatic);
    
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
            "    matrix: The target matrix\n"
            "    mode: DecompositionMode decomposition mode, default is QSD\n"
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
            "    matrix: The target matrix\n"
            "    mode: DecompositionMode decomposition mode, default is QSD\n"
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
        "    list: the source vector b, which will be extend to 2 ^ n",
        py::return_value_policy::automatic_reference);

#ifndef EIGEN_TENSOR_EXAMPLE
        //convert python.numpy.ndarray args to Eigen::Tensor 
        //convert Eigen::Tensor return values to python.numpy.ndarray 
        m.def("tensor3xd", [](pybind11::array_t<double> array)
        {
            auto tensor = to_tensor3x(array);
            std::cout << "Tensor Dims " << tensor.NumDimensions << std::endl;
            std::cout << tensor << std::endl;

            //return type : pybind11::array_t<double> array
            return to_array3x(tensor);
        }, py::return_value_policy::automatic);

        m.def("tensor4xd", [](pybind11::array_t<double> array)
        {
            auto tensor = to_tensor4x(array);
            std::cout << "Tensor Dims " << tensor.NumDimensions << std::endl;
            std::cout << tensor << std::endl;

            //return type : pybind11::array_t<double> array
            return to_array4x(tensor);
        }, py::return_value_policy::automatic);

#endif


#endif
}