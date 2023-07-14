#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"
#include "Components/Operator/PauliOperator.h"

USING_QPANDA
namespace py = pybind11;

void export_PauliOperator(py::module &m)
{
	py::class_<PauliOperator>(m, "PauliOperator")
		.def(py::init<>())
        .def(py::init<const complex_d &>())
        .def(py::init<EigenMatrixX &, bool>(), py::arg("matrix"), py::arg("is_reduce_duplicates") = false)
        .def(py::init<const std::string &, const complex_d &, bool>(), py::arg("key"), py::arg("value"), py::arg("is_reduce_duplicates") = false)
        .def(py::init<const PauliOperator::PauliMap&, bool>(), py::arg("pauli_map"), py::arg("is_reduce_duplicates") = false)
		.def("dagger", &PauliOperator::dagger)
		.def("data", &PauliOperator::data)
        .def("reduce_duplicates", &PauliOperator::reduceDuplicates)
        .def("error_threshold", &PauliOperator::error_threshold)
		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self * py::self)
		.def(py::self += py::self)
		.def(py::self -= py::self)
		.def(py::self *= py::self)
		.def(py::self + QPanda::complex_d())
		.def(py::self * QPanda::complex_d())
		.def(py::self - QPanda::complex_d())
		.def(QPanda::complex_d() + py::self)
		.def(QPanda::complex_d() * py::self)
		.def(QPanda::complex_d() - py::self)
		.def("__str__", &PauliOperator::toString)

		/*will delete*/
		.def("toHamiltonian", &PauliOperator::toHamiltonian)
		.def("getMaxIndex", &PauliOperator::getMaxIndex)
		.def("isEmpty", &PauliOperator::isEmpty)
		.def("isAllPauliZorI", &PauliOperator::isAllPauliZorI)
		.def("setErrorThreshold", &PauliOperator::setErrorThreshold)
		.def("remapQubitIndex", &PauliOperator::remapQubitIndex)
		.def("toString", &PauliOperator::toString)
		/*new interface*/
		.def("to_hamiltonian", &PauliOperator::toHamiltonian)
		.def("get_max_index", &PauliOperator::getMaxIndex)
		.def("is_empty", &PauliOperator::isEmpty)
		.def("is_all_pauli_z_or_i", &PauliOperator::isAllPauliZorI)
		.def("set_error_threshold", &PauliOperator::setErrorThreshold)
		.def("remap_qubit_index", &PauliOperator::remapQubitIndex)
        .def("to_string", &PauliOperator::toString)
        .def("to_matrix", &PauliOperator::to_matrix);    

    m.def("x", &x,
        py::arg("index"),
        "construct a pauli x operator\n"
        "\n"
        "Args:\n"
        "    int: pauli operate index\n"
        "\n"
        "Returns:\n"
        "    pauli operator x  \n"
        "Raises:\n"
        "    run_fail: An error occurred in construct a pauli x operator\n",
        py::return_value_policy::automatic);
    m.def("y", &y, 
        "construct a pauli y operator\n"
        "\n"
        "Args:\n"
        "    int: pauli operate index\n"
        "\n"
        "Returns:\n"
        "    pauli operator y  \n"
        "Raises:\n"
        "    run_fail: An error occurred in construct a pauli y operator\n",
        py::return_value_policy::automatic);
    m.def("z", &z,
        "construct a pauli z operator\n"
        "\n"
        "Args:\n"
        "    int: pauli operate index\n"
        "\n"
        "Returns:\n"
        "    pauli operator z  \n"
        "Raises:\n"
        "    run_fail: An error occurred in construct a pauli z operator\n",
        py::return_value_policy::automatic);
    m.def("i", &i, 
        "construct a pauli i operator\n"
        "\n"
        "Args:\n"
        "    int: pauli operate index\n"
        "\n"
        "Returns:\n"
        "    pauli operator i  \n"
        "Raises:\n"
        "    run_fail: An error occurred in construct a pauli i operator\n",
        py::return_value_policy::automatic);

    m.def("matrix_decompose_hamiltonian",
        [](EigenMatrixX& matrix)
        {
            return matrix_decompose_hamiltonian(matrix);
        },
        "decompose matrix into hamiltonian\n"
        "\n"
        "Args:\n"
        "    quantum_machine: quantum machine\n"
        "    matrix: 2^N *2^N double matrix \n"
        "\n"
        "Returns:\n"
        "    result : hamiltonian", py::return_value_policy::automatic);
    m.def("trans_vec_to_Pauli_operator", &transVecToPauliOperator<double>, "Transfrom vector to pauli operator");
    m.def("trans_Pauli_operator_to_vec", &transPauliOperatorToVec, "Transfrom Pauli operator to vector");
}
