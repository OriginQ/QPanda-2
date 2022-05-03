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
#include "Extensions/Extensions.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA

#include "Extensions/VirtualZTransfer/VirtualZTransfer.h"

void export_extension_class(py::module &m)
{
#ifdef USE_EXTENSION
    py::class_<HHLAlg>(m, "HHLAlg")
        .def(py::init<QuantumMachine *>())
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
            [](HHLAlg &self)
            {
                std::vector<Qubit *> q_vec = self.get_qubit_for_b();
                return q_vec;
            },
            "get_qubit_for_b",
            py::return_value_policy::reference)
        .def(
            "get_qubit_for_QFT",
            [](HHLAlg &self)
            {
                std::vector<Qubit *> q_vec = self.get_qubit_for_QFT();
                return q_vec;
            },
            "get_qubit_for_QFT",
            py::return_value_policy::reference)
        .def("query_uesed_qubit_num", &HHLAlg::query_uesed_qubit_num, "query_uesed_qubit_num");
#endif
    return;
}

void export_extension_funtion(py::module &m)
{
#ifdef USE_EXTENSION
    m.def(
        "expand_linear_equations",
        [](QStat &A, std::vector<double> &b)
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
        [](QPanda::QProg prog, QPanda::QuantumMachine *quantum_machine, bool optimization = false, uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(), uint32_t max_children = (std::numeric_limits<uint32_t>::max)(), const std::string &config_data = CONFIG_PATH)
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
        "    b_optimization:\n"
        "    max_partial: Limits the max number of partial solutions per step, There is no limit by default\n"
        "    max_children: Limits the max number of candidate - solutions per double gate, There is no limit by default\n"
        "    config_data: config data, @See JsonConfigParam::load_config()\n"
        "\n"
        "Returns:\n"
        "    mapped quantum program",
        py::return_value_policy::automatic);

    m.def("virtual_z_transform",
          py::overload_cast<QProg &, QuantumMachine *, const bool, const std::string &>(&virtual_z_transform),
          py::arg("prog"),
          py::arg("quantum_machine"),
          py::arg("b_del_rz_gate") = false,
          py::arg("config_data") = CONFIG_PATH,
          py::return_value_policy::automatic);

    m.def(
        "matrix_decompose_pualis",
        [](QuantumMachine *qvm, const EigenMatrixX &mat)
        {
            PualiOperatorLinearCombination linearcom;
            matrix_decompose_pualis(qvm, mat, linearcom);
            return linearcom;
        },
        py::return_value_policy::automatic);
#endif
    m.def(
        "matrix_decompose",
        [](QVec &qubits, QStat &src_mat, const DecompositionMode mode, bool b_positive_seq)
        {
            switch (mode)
            {
            case DecompositionMode::QR:
                return matrix_decompose_qr(qubits, src_mat, b_positive_seq);
                // break;
            case DecompositionMode::HOUSEHOLDER_QR:
                return matrix_decompose_householder(qubits, src_mat, b_positive_seq);
                // break;
            default:
                throw std::runtime_error("Error: DecompositionMode");
            }
        },
        py::arg("qubits"),
        py::arg("matrix"),
        py::arg_v("mode", DecompositionMode::QR, "DecompositionMode.QR"),
        py::arg("b_positive_seq") = true,
        "Matrix decomposition\n"
        "\n"
        "Args:\n"
        "    qubits: the used qubits\n"
        "    matrix: The target matrix\n"
        "    mode: DecompositionMode decomposition mode, default is QR\n"
        "    b_positive_seq: true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is true\n"
        "\n"
        "Returns:\n"
        "    QCircuit The quantum circuit for target matrix",
        py::return_value_policy::automatic);
}
