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

#ifdef USE_EXTENSION

#include "QAlg/QAlg.h"

#endif

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA

void export_qalg(py::module &m)
{

#ifdef USE_EXTENSION

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
        "\n"
        "Args:\n"
        "\n"
        "    QVec: qubits list a\n"
        "    QVec: qubits list b\n"
        "    QVec: qubits list c\n"
        "    Qubit: qubit\n"
        "\n"
        "Returns:\n"
        "\n"
        "    result : circuit \n"
        "\n"
        "Raises:\n"
        "    run_fail: An error occurred in QAdderIgnoreCarry\n",
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
        "\n"
        "Args:\n"
        "\n"
        "    QVec: qubits list a\n"
        "    QVec: qubits list b\n"
        "    QVec: qubits list c\n"
        "    QVec: qubits list k\n"
        "    QVec: qubits list f\n"
        "    QVec: qubits list s\n"
        "    list: ClassicalCondition list\n"
        "\n"
        "Returns:\n"
        "\n"
        "    result : circuit \n"
        "\n"
        "Raises:\n"
        "    run_fail: An error occurred in QDividerWithAccuracy\n",
        py::return_value_policy::reference);

    m.def("QDivWithAccuracy",
        py::overload_cast<QVec &, QVec &, QVec &, QVec &, QVec &, std::vector<ClassicalCondition> &>(&QDiv),
        "Quantum division with accuracy",
        "\n"
        "Args:\n"
        "\n"
        "    QVec: qubits list a\n"
        "    QVec: qubits list b\n"
        "    QVec: qubits list c\n"
        "    QVec: qubits list k\n"
        "    QVec: qubits list f\n"
        "    QVec: qubits list s\n"
        "    list: ClassicalCondition list\n"
        "\n"
        "Returns:\n"
        "\n"
        "    result : circuit \n"
        "\n"
        "Raises:\n"
        "    run_fail: An error occurred in QDivWithAccuracy\n",
        py::return_value_policy::reference);
    m.def("bind_data", &bind_data,
        "Quantum bind classical data to qvec",
        "\n"
        "Args:\n"
        "    int: classical data\n"
        "    QVec: qubits list\n"
        "\n"
        "Returns:\n"
        "    result : circuit \n"
        "Raises:\n"
        "    run_fail: An error occurred in bind_data\n",
        py::return_value_policy::reference);
    m.def("bind_nonnegative_data", &bind_nonnegative_data,
        "Quantum bind classical nonnegative integer",
        "\n"
        "Args:\n"
        "    int: classical data\n"
        "    QVec: qubits list\n"
        "\n"
        "Returns:\n"
        "    result : circuit \n"
        "Raises:\n"
        "    run_fail: An error occurred in bind_nonnegative_data\n",
        py::return_value_policy::reference);

    m.def("constModAdd",
        &constModAdd,
        "Quantum modular adder",
        "\n"
        "Args:\n"
        "   QVec qvec\n"
        "   int base\n"
        "   int module_Num\n"
        "   QVec qvec1\n"
        "   QVec qvec2\n"
        "\n"
        "Returns:\n"
        "    result circuit \n"
        "Raises:\n"
        "    run_fail: An error occurred in constModAdd\n",
        py::return_value_policy::reference);

    m.def("constModMul",
        &constModMul,
        "Quantum modular multiplier",
        "\n"
        "Args:\n"
        "   QVec qvec\n"
        "   int base\n"
        "   int module_Num\n"
        "   QVec qvec1\n"
        "   QVec qvec2\n"
        "\n"
        "Returns:\n"
        "    result circuit \n"
        "Raises:\n"
        "    run_fail: An error occurred in constModMul\n",
        py::return_value_policy::reference);

    m.def("constModExp",
        &constModExp,
        "Quantum modular exponents",
        "\n"
        "Args:\n"
        "   QVec qvec\n"
        "   int base\n"
        "   int module_Num\n"
        "   QVec qvec1\n"
        "   QVec qvec2\n"
        "\n"
        "Returns:\n"
        "    result circuit \n"
        "Raises:\n"
        "    run_fail: An error occurred in constModExp\n",
        py::return_value_policy::reference);

    m.def("amplitude_encode",
        py::overload_cast<QVec, std::vector<double>, const bool>(&amplitude_encode),
        py::arg("qubit"),
        py::arg("data"),
        py::arg("b_need_check_normalization") = true,
        "Encode the input double data to the amplitude of qubits\n"
        "\n"
        "Args:\n"
        "    qubit: quantum program qubits\n"
        "    data: double data list\n"
        "    b_need_check_normalization: is need to check normalization\n"
        "\n"
        "Returns:\n"
        "    result circuit \n"
        "Raises:\n"
        "    run_fail: An error occurred in amplitude_encode\n",
        py::return_value_policy::automatic);
    m.def("amplitude_encode",
        py::overload_cast<QVec, const QStat &>(&amplitude_encode),
        py::arg("qubit"),
        py::arg("data"),
        "Encode the input double data to the amplitude of qubits\n"
        "\n"
        "Args:\n"
        "    qubit: quantum program qubits\n"
        "    data: double data list\n"
        "\n"
        "Returns:\n"
        "    result circuit \n"
        "Raises:\n"
        "    run_fail: An error occurred in amplitude_encode\n",
        py::return_value_policy::automatic);

    m.def("iterative_amplitude_estimation",
        &iterative_amplitude_estimation,
        "estimate the probability corresponding to the ground state |1> of the last bit"
        "\n"
        "Args:\n"
        "    QCircuit: quantum circuit\n"
        "    qvec: qubit list\n"
        "    double: epsilon\n"
        "    double: confidence\n"
        "\n"
        "Returns:\n"
        "    result iterative amplitude\n"
        "Raises:\n"
        "    run_fail: An error occurred in iterative_amplitude_estimation\n",
        py::return_value_policy::automatic);

    m.def("QFT",
        &QFT,
        py::arg("qubits"),
        "Build QFT quantum circuit\n"
        "\n"
        "Args:\n"
        "    qvec: qubit list\n"
        "\n"
        "Returns:\n"
        "    result : qft circuit\n"
        "Raises:\n"
        "    run_fail: An error occurred in QFT\n",
        py::return_value_policy::automatic);

    m.def("QPE",
        &build_QPE_circuit<QStat>,
        py::arg("control_qubits"),
        py::arg("target_qubits"),
        py::arg("matrix"),
        py::arg("b_estimate_eigenvalue") = false,
        "Quantum phase estimation\n"
        "\n"
        "Args:\n"
        "    control_qubits: control qubit list\n"
        "    target_qubits: target qubit list\n"
        "    matrix: matrix\n"
        "\n"
        "Returns:\n"
        "    result : QPE circuit\n"
        "Raises:\n"
        "    run_fail: An error occurred in QPE\n",
        py::return_value_policy::automatic_reference);

    m.def("Grover",
        &build_grover_prog,
        py::arg("data"),
        py::arg("Classical_condition"),
        py::arg("QuantumMachine"),
        py::arg("qlist"),
        py::arg("data") = 2,
        "Quantum grover circuit\n"
        "\n"
        "Args:\n"
        "    qvec: qubit list\n"
        "    Classical_condition: quantum Classical condition\n"
        "    QuantumMachine: quantum machine\n"
        "\n"
        "Returns:\n"
        "    result : Grover circuit\n"
        "Raises:\n"
        "    run_fail: An error occurred in Grover\n",
        py::return_value_policy::automatic);

    m.def("Grover_search",
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
            py::arg("repeat") = 2,
            "Use Grover algorithm to search target data, return QProg and search_result\n"
            "\n"
            "Args:\n"
            "    list: data list\n"
            "    Classical_condition: quantum Classical condition\n"
            "    QuantumMachine: quantum machine\n"
            "    repeat: search repeat times\n"
            "\n"
            "Returns:\n"
            "    result : Grover search result\n"
            "Raises:\n"
            "    run_fail: An error occurred in Grover\n",
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

    py::enum_<QITE::UpdateMode>(m, "UpdateMode", "quantum imaginary time evolution update mode", py::arithmetic())
        .value("GD_VALUE", QITE::UpdateMode::GD_VALUE)
        .value("GD_DIRECTION", QITE::UpdateMode::GD_DIRECTION)
        .export_values();

    py::class_<QITE>(m, "QITE", "quantum imaginary time evolution")
        .def(py::init<>())
        .def("set_Hamiltonian", &QITE::setHamiltonian)
        .def("set_pauli_matrix", &QITE::setPauliMatrix)
        .def("set_ansatz_gate", &QITE::setAnsatzGate)
        .def("set_delta_tau", &QITE::setDeltaTau)
        .def("set_iter_num", &QITE::setIterNum)
        .def("set_para_update_mode", &QITE::setParaUpdateMode)
        .def("set_upthrow_num", &QITE::setUpthrowNum)
        .def("set_convergence_factor_Q", &QITE::setConvergenceFactorQ)
        .def("set_quantum_machine_type", &QITE::setQuantumMachineType)
        .def("set_log_file", &QITE::setLogFile)
        .def("get_arbitary_cofficient", &QITE::setArbitaryCofficient)
        .def("exec", &QITE::exec, py::arg("is_optimization") = true)
        .def("get_ansatz_list", &QITE::get_ansatz_list)
        .def("get_ansatz_theta_list", &QITE::get_ansatz_theta_list)
        .def("get_result", &QITE::getResult)
        .def("get_exec_result", &QITE::get_exec_result,
            py::arg("reverse") = false,
            py::arg("sort") = false)
        .def("get_all_exec_result", &QITE::get_all_exec_result,
            py::arg("reverse") = false,
            py::arg("sort") = false);

    m.def("quantum_walk_alg",
        &build_quantum_walk_search_prog,
        py::arg("data"),
        py::arg("Classical_condition"),
        py::arg("QuantumMachine"),
        py::arg("qlist"),
        py::arg("data") = 2,
        "Build quantum-walk algorithm quantum circuit",
        py::return_value_policy::automatic);

    m.def("quantum_walk_search",
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
            "Use Quantum-walk Algorithm to search target data, return QProg and search_result\n"
            "\n"
            "Args:\n"
            "    list: data list\n"
            "    Classical_condition: quantum Classical condition\n"
            "    QuantumMachine: quantum machine\n"
            "    repeat: search repeat times\n"
            "\n"
            "Returns:\n"
            "    result : Quantum-walk search result\n"
            "Raises:\n"
            "    run_fail: An error occurred in Quantum-walk\n",
            py::return_value_policy::automatic);

    m.def("Shor_factorization",
        &Shor_factorization,
        "Use Shor factorize integer num\n"
        "\n"
        "Args:\n"
        "    int: target integer num\n"
        "    result: Shor result\n"
        "\n"
        "Returns:\n"
        "    result : Shor_factorization search result\n"
        "Raises:\n"
        "    run_fail: An error occurred in Shor_factorization\n",
        py::return_value_policy::reference);

    py::class_<AnsatzGate>(m, "AnsatzGate", "ansatz gate struct")
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

    py::class_<AnsatzCircuit>(m, "Ansatz", "quantum ansatz class")
        .def(py::init<>())
        .def(py::init<QGate&>())
        .def(py::init<AnsatzGate&>())

        .def(py::init<Ansatz&, const Thetas&>(),
        py::arg("ansatz"),
        py::arg("thetas") = std::vector<double>())

        .def(py::init<AnsatzCircuit&, const Thetas&>(),
        py::arg("ansatz_circuit"),
        py::arg("thetas") = std::vector<double>())

        .def(py::init<QCircuit&, const Thetas&>(),
        py::arg("circuit"),
        py::arg("thetas") = std::vector<double>())

        .def("__lshift__", &AnsatzCircuit::operator<<<QGate&>, py::return_value_policy::reference)
        .def("__lshift__", &AnsatzCircuit::operator<<<Ansatz>, py::return_value_policy::reference)
        .def("__lshift__", &AnsatzCircuit::operator<<<QCircuit>, py::return_value_policy::reference)
        .def("__lshift__", &AnsatzCircuit::operator<<<AnsatzGate>, py::return_value_policy::reference)

        .def("insert", [](AnsatzCircuit& ansatz, QGate& gate)
            {
                return ansatz.insert(gate);
            }, py::arg("gate"), py::return_value_policy::reference)

        .def("insert", [](AnsatzCircuit& ansatz, AnsatzGate &gate)
            {
                return ansatz.insert(gate);
            }, py::arg("gate"), py::return_value_policy::reference)
        .def("insert", [](AnsatzCircuit& ansatz, Ansatz& gate)
            {
                return ansatz.insert(gate);
            }, py::arg("gate"), py::return_value_policy::reference)

        .def("insert", [](AnsatzCircuit& ansatz, QCircuit& circuit)
            {
                return ansatz.insert(circuit);
            }, py::arg("gate"), py::return_value_policy::reference)

        .def("insert", [](AnsatzCircuit& ansatz, AnsatzCircuit& circuit, const Thetas& thetas)
            {
                return ansatz.insert(circuit, thetas);
            }, py::arg("gate"), py::arg("thetas") = std::vector<double>(), py::return_value_policy::reference)

        .def("set_thetas", [](AnsatzCircuit& ansatz, const Thetas& thetas)
            {
                return ansatz.set_thetas(thetas);
            }, py::arg("thetas"), py::return_value_policy::reference)

        .def("get_ansatz_list", [](AnsatzCircuit& ansatz)
            {
                return ansatz.get_ansatz_list();
            }, py::return_value_policy::reference)

        .def("get_thetas_list", [](AnsatzCircuit& ansatz)
            {
                return ansatz.get_thetas_list();
            }, py::return_value_policy::reference)

        .def("__str__", [](AnsatzCircuit &ansatz)
            {
                auto circuit = ansatz.qcircuit();
                return draw_qprog(circuit);
            }, py::return_value_policy::reference);

    py::implicitly_convertible<QGate, AnsatzCircuit>();
    py::implicitly_convertible<Ansatz, AnsatzCircuit>();
    py::implicitly_convertible<QCircuit, AnsatzCircuit>();
    py::implicitly_convertible<AnsatzGate, AnsatzCircuit>();

#endif
}
