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
#include "QPandaConfig.h"
#include "QPanda.h"


using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA

auto QAdderWithCarry = [](
	QVec& adder1,
	QVec& adder2,
	Qubit* c,
	Qubit* is_carry)
{
	return QAdder(adder1, adder2, c, is_carry);
};

auto QAdderIgnoreCarry = [](
	QVec& adder1,
	QVec& adder2,
	Qubit* c)
{
	return QAdder(adder1, adder2, c);
};

auto QDividerNoAccuracy = [](
	QVec& a,
	QVec& b,
	QVec& c,
	QVec& k,
	ClassicalCondition& t)
{
	return QDivider(a, b, c, k, t);
};

auto QDividerWithAccuracy = [](
	QVec& a,
	QVec& b,
	QVec& c,
	QVec& k,
	QVec& f,
	std::vector<ClassicalCondition>& s)
{
	return QDivider(a, b, c, k, f, s);
};

auto QDivNoAccuracy = [](
	QVec& a,
	QVec& b,
	QVec& c,
	QVec& k,
	ClassicalCondition& t)
{
	return QDiv(a, b, c, k, t);
};

auto QDivWithAccuracy = [](
	QVec& a,
	QVec& b,
	QVec& c,
	QVec& k,
	QVec& f,
	std::vector<ClassicalCondition>& s)
{
	return QDiv(a, b, c, k, f, s);
};

void init_qalg(py::module & m)
{
	m.def("MAJ", &MAJ, "Quantum adder MAJ module", py::return_value_policy::reference);
	m.def("UMA", &UMA, "Quantum adder UMA module", py::return_value_policy::reference);
	m.def("MAJ2", &MAJ2, "Quantum adder MAJ2 module", py::return_value_policy::reference);
	m.def("isCarry", &isCarry, "Construct a circuit to determine if there is a carry", py::return_value_policy::reference);
	m.def("QAdder", QAdderWithCarry, "Quantum adder", py::return_value_policy::reference);
	m.def("QAdderIgnoreCarry", QAdderIgnoreCarry, "Quantum adder ignore carry", py::return_value_policy::reference);
	m.def("QAdd", &QAdd, "Quantum adder that supports signed operations, but ignore carry", py::return_value_policy::reference);
	m.def("QComplement", &QComplement, "Convert quantum state to binary complement representation", py::return_value_policy::reference);
	m.def("QSub", &QSub, "Quantum subtraction", py::return_value_policy::reference);
	m.def("QMultiplier", &QMultiplier, "Quantum multiplication, only supports positive multiplication", py::return_value_policy::reference);
	m.def("QMul", &QMul, "Quantum multiplication", py::return_value_policy::reference);
	m.def("QDivider", QDividerNoAccuracy, "Quantum division, only supports positive division, and the highest position of a and b and c is sign bit", py::return_value_policy::reference);
	m.def("QDiv", QDivNoAccuracy, "Quantum division", py::return_value_policy::reference);
	m.def("QDividerWithAccuracy", QDividerWithAccuracy, "Quantum division with accuracy, only supports positive division, and the highest position of a and b and c is sign bit", py::return_value_policy::reference);
	m.def("QDivWithAccuracy", QDivWithAccuracy, "Quantum division with accuracy", py::return_value_policy::reference);
	m.def("bind_data", &bind_data, "Quantum bind data", py::return_value_policy::reference);
	m.def("bind_nonnegative_data", &bind_nonnegative_data, "Quantum bind nonnegative integer", py::return_value_policy::reference);
	m.def("constModAdd", &constModAdd, "Quantum modular adder", py::return_value_policy::reference);
	m.def("constModMul", &constModMul, "Quantum modular multiplier", py::return_value_policy::reference);
	m.def("constModExp", &constModExp, "Quantum modular exponents", py::return_value_policy::reference);

    m.def("amplitude_encode", &amplitude_encode, "Encode the input data to the amplitude of qubits", "qlist"_a, "data"_a, "bool"_a = true,
        py::return_value_policy::automatic
        );

	m.def("QFT", &QFT, "Build QFT quantum circuit", "qlist"_a,
		py::return_value_policy::automatic
	);

	m.def("QPE", [](const QVec control_qubits, const QVec target_qubits, QStat matrix, bool b_estimate_eigenvalue = false) {
		return build_QPE_circuit(control_qubits, target_qubits, matrix, b_estimate_eigenvalue);
	}
		, "qlist"_a, "qlist"_a, "matrix"_a, "bool"_a = false,
		"Build QPE quantum circuit",
		py::return_value_policy::automatic_reference
		);

	m.def("HHL", &build_HHL_circuit, "Build HHL quantum circuit", "matrix"_a, "data"_a, "QuantumMachine"_a,
		py::return_value_policy::automatic
	);

	m.def("HHL_solve_linear_equations", &HHL_solve_linear_equations, "use HHL to solve linear equations", "matrix"_a, "data"_a,
		py::return_value_policy::automatic
	);

	m.def("Grover", [](const std::vector<int>& data, ClassicalCondition condition, 
		QuantumMachine *qvm, QVec& measure_qubits, size_t repeat = 2) {
		return build_grover_prog(data, condition, qvm, measure_qubits, repeat);
	}, "Build Grover quantum circuit", 
		"data"_a, "Classical_condition"_a, "QuantumMachine"_a, "qlist"_a, "data"_a = 2,
		py::return_value_policy::automatic
	);

	m.def("Grover_search", [](const std::vector<int>& data, ClassicalCondition condition, QuantumMachine *qvm, size_t repeat = 2) {
		std::vector<SearchDataByUInt> target_data_vec(data.begin(), data.end());
		std::vector<size_t> search_result;
		auto prog = grover_alg_search_from_vector(target_data_vec, condition, search_result, qvm, repeat);
		py::list ret_data;
		ret_data.append(prog);
		ret_data.append(search_result);
		return ret_data;
	}, "use Grover algorithm to search target data, return QProg and search_result", 
		"list"_a, "Classical_condition"_a, "QuantumMachine"_a, "data"_a = 2,
		py::return_value_policy::automatic
	);

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

	m.def("quantum_walk_alg", [](const std::vector<int>& data, ClassicalCondition condition,
		QuantumMachine *qvm, QVec& measure_qubits, size_t repeat = 2) {
		return build_quantum_walk_prog(data, condition, qvm, measure_qubits, repeat);
	}, "Build quantum-walk algorithm quantum circuit",
		"data"_a, "Classical_condition"_a, "QuantumMachine"_a, "qlist"_a, "data"_a = 2,
		py::return_value_policy::automatic
		);

	m.def("quantum_walk_search", [](const std::vector<int>& data, ClassicalCondition condition, QuantumMachine *qvm, size_t repeat = 2) {
		std::vector<SearchDataByUInt> target_data_vec(data.begin(), data.end());
		std::vector<size_t> search_result;
		auto prog = quantum_walk_alg_search_from_vector(target_data_vec, condition, qvm, search_result, repeat);
		py::list ret_data;
		ret_data.append(prog);
		ret_data.append(search_result);
		return ret_data;
	}, "use quantum-walk algorithm to search target data, return QProg and search_result",
		"list"_a, "Classical_condition"_a, "QuantumMachine"_a, "data"_a = 2,
		py::return_value_policy::automatic
		);
		
	m.def("Shor_factorization", &Shor_factorization, "Shor Algorithm function", 
		py::return_value_policy::reference);
}
