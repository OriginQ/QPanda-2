#include "QPandaConfig.h"
#include "QPanda.h"
#include <math.h>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"
#include "pybind11/operators.h"


using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA

void export_qalg(py::module &m)
{
	m.def("MAJ", &MAJ, "Quantum adder MAJ module", py::return_value_policy::reference);
	m.def("UMA", &UMA, "Quantum adder UMA module", py::return_value_policy::reference);
	m.def("MAJ2", &MAJ2, "Quantum adder MAJ2 module", py::return_value_policy::reference);
	m.def("isCarry", &isCarry, "Construct a circuit to determine if there is a carry", py::return_value_policy::reference);
	m.def("QAdder",
		  py::overload_cast<QVec &, QVec &, Qubit *, Qubit *>(&QAdder),
		  "Quantum adder with carry",
		  py::return_value_policy::reference);
	m.def("QAdderIgnoreCarry",
		  py::overload_cast<QVec &, QVec &, Qubit *>(&QAdder),
		  "Quantum adder ignore carry",
		  py::return_value_policy::reference);
	m.def("QAdd", &QAdd, "Quantum adder that supports signed operations, but ignore carry", py::return_value_policy::reference);
	m.def("QComplement", &QComplement, "Convert quantum state to binary complement representation", py::return_value_policy::reference);
	m.def("QSub", &QSub, "Quantum subtraction", py::return_value_policy::reference);
	m.def("QMultiplier", &QMultiplier, "Quantum multiplication, only supports positive multiplication", py::return_value_policy::reference);
	m.def("QMul", &QMul, "Quantum multiplication", py::return_value_policy::reference);
	m.def("QDivider",
		  py::overload_cast<QVec &, QVec &, QVec &, QVec &, ClassicalCondition &>(&QDivider),
		  py::arg("a"),
		  py::arg("b"),
		  py::arg("c"),
		  py::arg("k"),
		  py::arg("t"),
		  "Quantum division, only supports positive division, and the highest position of a and b and c is sign bit",
		  py::return_value_policy::reference);
	m.def("QDiv",
		  py::overload_cast<QVec &, QVec &, QVec &, QVec &, ClassicalCondition &>(&QDiv),
		  "Quantum division",
		  py::return_value_policy::reference);
	m.def("QDividerWithAccuracy",
		  py::overload_cast<QVec &, QVec &, QVec &, QVec &, QVec &, std::vector<ClassicalCondition> &>(&QDivider),
		  py::arg("a"),
		  py::arg("b"),
		  py::arg("c"),
		  py::arg("k"),
		  py::arg("f"),
		  py::arg("s"),
		  "Quantum division with accuracy, only supports positive division, and the highest position of a and b and c is sign bit",
		  py::return_value_policy::reference);
	m.def("QDivWithAccuracy",
		  py::overload_cast<QVec &, QVec &, QVec &, QVec &, QVec &, std::vector<ClassicalCondition> &>(&QDiv),
		  "Quantum division with accuracy",
		  py::return_value_policy::reference);
	m.def("bind_data", &bind_data, "Quantum bind data", py::return_value_policy::reference);
	m.def("bind_nonnegative_data", &bind_nonnegative_data, "Quantum bind nonnegative integer", py::return_value_policy::reference);
	m.def("constModAdd", &constModAdd, "Quantum modular adder", py::return_value_policy::reference);
	m.def("constModMul", &constModMul, "Quantum modular multiplier", py::return_value_policy::reference);
	m.def("constModExp", &constModExp, "Quantum modular exponents", py::return_value_policy::reference);

	m.def("amplitude_encode",
		  py::overload_cast<QVec, std::vector<double>, const bool>(&amplitude_encode),
		  py::arg("qubit"),
		  py::arg("data"),
		  py::arg("b_need_check_normalization") = true,
		  "Encode the input double data to the amplitude of qubits",
		  py::return_value_policy::automatic);
	m.def("amplitude_encode",
		  py::overload_cast<QVec, const QStat &>(&amplitude_encode),
		  py::arg("qubit"),
		  py::arg("data"),
		  "Encode the input complex data to the amplitude of qubits",
		  py::return_value_policy::automatic);

	m.def("iterative_amplitude_estimation",
		  &iterative_amplitude_estimation,
		  "estimate the probability corresponding to the ground state |1> of the last bit",
		  py::return_value_policy::automatic);

	m.def("QFT",
		  &QFT,
		  py::arg("qubits"),
		  "Build QFT quantum circuit",
		  py::return_value_policy::automatic);

	m.def("QPE",
		  &build_QPE_circuit<QStat>,
		  py::arg("control_qubits"),
		  py::arg("target_qubits"),
		  py::arg("matrix"),
		  py::arg("b_estimate_eigenvalue") = false,
		  "Build QPE quantum circuit",
		  py::return_value_policy::automatic_reference);

	m.def("Grover",
		  &build_grover_prog,
		  py::arg("data"),
		  py::arg("Classical_condition"),
		  py::arg("QuantumMachine"),
		  py::arg("qlist"),
		  py::arg("data") = 2,
		  "Build Grover quantum circuit",
		  py::return_value_policy::automatic);

	m.def(
		"Grover_search",
		[](const std::vector<uint32_t> &data, ClassicalCondition condition, QuantumMachine *qvm, size_t repeat = 2)
		{
			std::vector<SearchDataByUInt> target_data_vec(data.begin(), data.end());
			std::vector<size_t> search_result;
			auto prog = grover_alg_search_from_vector(target_data_vec, condition, search_result, qvm, repeat);
			py::list ret_data;
			ret_data.append(prog);
			ret_data.append(search_result);
			return ret_data;
		},
		py::arg("list"),
		py::arg("Classical_condition"),
		py::arg("QuantumMachine"),
		py::arg("data") = 2,
		"use Grover algorithm to search target data, return QProg and search_result",
		py::return_value_policy::automatic);

	m.def(
		"Grover_search",
		[](const std::vector<std::string> &data, std::string search_element, QuantumMachine *qvm, size_t repeat = 2)
		{
			std::vector<size_t> search_result;
			auto prog = grover_search_alg(data, search_element, search_result, qvm, repeat);
			py::list ret_data;
			ret_data.append(prog);
			ret_data.append(search_result);
			return ret_data;
		},
		py::arg("list"),
		py::arg("Classical_condition"),
		py::arg("QuantumMachine"),
		py::arg("data") = 2,
		"use Grover algorithm to search target data, return QProg and search_result",

		py::return_value_policy::automatic);

	py::enum_<AnsatzGateType>(m, "AnsatzGateType", py::arithmetic())
		.value("AGT_NOT", AnsatzGateType::AGT_NOT)
		.value("AGT_H", AnsatzGateType::AGT_H)
		.value("AGT_RX", AnsatzGateType::AGT_RX)
		.value("AGT_RY", AnsatzGateType::AGT_RY)
		.value("AGT_RZ", AnsatzGateType::AGT_RZ)
		.export_values();

	py::class_<AnsatzGate>(m, "AnsatzGate")
		.def(py::init<>([](AnsatzGateType type, int target)
						{ return AnsatzGate(type, target); }))
		.def(py::init<>([](AnsatzGateType type, int target, double theta)
						{ return AnsatzGate(type, target, theta); }))
		.def(py::init<>([](AnsatzGateType type, int target, double theta, int control)
						{ return AnsatzGate(type, target, theta, control); }))
		.def_readwrite("type", &AnsatzGate::type)
		.def_readwrite("target", &AnsatzGate::target)
		.def_readwrite("theta", &AnsatzGate::theta)
		.def_readwrite("control", &AnsatzGate::control);

	py::enum_<QITE::UpdateMode>(m, "UpdateMode", py::arithmetic())
		.value("GD_VALUE", QITE::UpdateMode::GD_VALUE)
		.value("GD_DIRECTION", QITE::UpdateMode::GD_DIRECTION)
		.export_values();

	py::class_<QITE>(m, "QITE")
		.def(py::init<>())
		.def("set_Hamiltonian", &QITE::setHamiltonian)
		.def("set_ansatz_gate", &QITE::setAnsatzGate)
		.def("set_delta_tau", &QITE::setDeltaTau)
		.def("set_iter_num", &QITE::setIterNum)
		.def("set_para_update_mode", &QITE::setParaUpdateMode)
		.def("set_upthrow_num", &QITE::setUpthrowNum)
		.def("set_convergence_factor_Q", &QITE::setConvergenceFactorQ)
		.def("set_quantum_machine_type", &QITE::setQuantumMachineType)
		.def("set_log_file", &QITE::setLogFile)
		.def("get_arbitary_cofficient", &QITE::setArbitaryCofficient)
		.def("exec", &QITE::exec)
		.def("get_result", &QITE::getResult);

	m.def("quantum_walk_alg",
		  &build_quantum_walk_search_prog,
		  py::arg("data"),
		  py::arg("Classical_condition"),
		  py::arg("QuantumMachine"),
		  py::arg("qlist"),
		  py::arg("data") = 2,
		  "Build quantum-walk algorithm quantum circuit",
		  py::return_value_policy::automatic);

	m.def(
		"quantum_walk_search",
		[](const std::vector<uint32_t> &data, ClassicalCondition condition, QuantumMachine *qvm, size_t repeat = 2)
		{
			std::vector<SearchDataByUInt> target_data_vec(data.begin(), data.end());
			std::vector<size_t> search_result;
			auto prog = quantum_walk_alg_search_from_vector(target_data_vec, condition, qvm, search_result, repeat);
			py::list ret_data;
			ret_data.append(prog);
			ret_data.append(search_result);
			return ret_data;
		},
		py::arg("list"),
		py::arg("Classical_condition"),
		py::arg("QuantumMachine"),
		py::arg("data") = 2,
		"use quantum-walk algorithm to search target data, return QProg and search_result",
		py::return_value_policy::automatic);

	m.def("Shor_factorization", &Shor_factorization, "Shor Algorithm function", py::return_value_policy::reference);
}
