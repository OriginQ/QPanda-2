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
		.def("dagger", &PauliOperator::dagger,
            "Returns the adjoint (dagger) of the Pauli operator.\n"
            "\n"
            "This function computes and returns the adjoint of the current operator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A new instance of PauliOperator representing the adjoint.\n")
        .def("data", &PauliOperator::data,
            "Retrieves the data representation of the Pauli operator.\n"
            "\n"
            "This function returns the internal data structure representing the operator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The data representation of the Pauli operator.\n")
            .def("reduce_duplicates", &PauliOperator::reduceDuplicates,
                "Reduces duplicates in the Pauli operator representation.\n"
                "\n"
                "This function modifies the operator to remove duplicate elements.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     None.\n")
            .def("error_threshold", &PauliOperator::error_threshold,
                "Retrieves the current error threshold for the operator.\n"
                "\n"
                "This function returns the error threshold value set for the operator.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     A double representing the error threshold.\n")
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self* py::self)
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def(py::self *= py::self)
            .def(py::self + QPanda::complex_d())
            .def(py::self* QPanda::complex_d())
            .def(py::self - QPanda::complex_d())
            .def(QPanda::complex_d() + py::self)
            .def(QPanda::complex_d()* py::self)
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
            .def("to_hamiltonian", &PauliOperator::toHamiltonian,
                "Converts the Pauli operator to its Hamiltonian representation.\n"
                "\n"
                "This function transforms the current Pauli operator into its corresponding Hamiltonian form.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     A new Hamiltonian representation of the operator.\n")
            .def("get_max_index", &PauliOperator::getMaxIndex,
                "Retrieves the maximum qubit index used in the operator.\n"
                "\n"
                "This function returns the highest index of qubits present in the Pauli operator.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     An integer representing the maximum qubit index.\n")
            .def("is_empty", &PauliOperator::isEmpty,
                "Checks if the Pauli operator is empty.\n"
                "\n"
                "This function determines whether the current operator contains any terms.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     A boolean indicating if the operator is empty (true) or not (false).\n")
            .def("is_all_pauli_z_or_i", &PauliOperator::isAllPauliZorI,
                "Checks if all terms are either Pauli Z or identity.\n"
                "\n"
                "This function evaluates whether all components of the operator are either Pauli Z or the identity operator.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     A boolean indicating if all terms are Pauli Z or identity (true) or not (false).\n")
            .def("set_error_threshold", &PauliOperator::setErrorThreshold,
                "Sets the error threshold for the operator.\n"
                "\n"
                "This function allows the user to define a new error threshold value.\n"
                "\n"
                "Args:\n"
                "     double threshold: The new error threshold value to set.\n"
                "\n"
                "Returns:\n"
                "     None.\n")

            .def("remap_qubit_index", &PauliOperator::remapQubitIndex,
                "Remaps the qubit indices in the operator.\n"
                "\n"
                "This function updates the qubit indices according to the provided mapping.\n"
                "\n"
                "Args:\n"
                "     const std::map<int, int>& index_map: A mapping of old indices to new indices.\n"
                "\n"
                "Returns:\n"
                "     None.\n")
            .def("to_string", &PauliOperator::toString,
                "Converts the Pauli operator to a string representation.\n"
                "\n"
                "This function provides a human-readable format of the Pauli operator.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     A string representing the Pauli operator.\n")
            .def("to_matrix", &PauliOperator::to_matrix,
                "Converts the Pauli operator to a matrix form.\n"
                "\n"
                "This function transforms the Pauli operator into its matrix representation.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     An EigenMatrixX representing the matrix form of the operator.\n");

    m.def("x", &x,
        py::arg("index"),
        "Construct a Pauli X operator.\n"
        "\n"
        "Args:\n"
        "     index (int): Pauli operator index.\n"
        "\n"
        "Returns:\n"
        "     Pauli operator X.\n",
        py::return_value_policy::automatic);
    m.def("y", &y,
        "Construct a Pauli Y operator.\n"
        "\n"
        "Args:\n"
        "     index (int): Pauli operator index.\n"
        "\n"
        "Returns:\n"
        "     Pauli operator Y.\n",
        py::return_value_policy::automatic);
    m.def("z", &z,
        "Construct a Pauli Z operator.\n"
        "\n"
        "Args:\n"
        "     index (int): Pauli operator index.\n"
        "\n"
        "Returns:\n"
        "     Pauli operator Z.\n",
        py::return_value_policy::automatic);
    m.def("i", &i,
        "Construct a Pauli I operator.\n"
        "\n"
        "Args:\n"
        "     index (int): Pauli operator index.\n"
        "\n"
        "Returns:\n"
        "     Pauli operator I.\n",
        py::return_value_policy::automatic);

    m.def("matrix_decompose_hamiltonian",
        [](EigenMatrixX& matrix)
        {
            return matrix_decompose_hamiltonian(matrix);
        },
        "Decompose matrix into Hamiltonian.\n"
        "\n"
        "Args:\n"
        "     matrix (EigenMatrixX): 2^N * 2^N double matrix.\n"
        "\n"
        "Returns:\n"
        "     Decomposed Hamiltonian representation.\n",
        py::return_value_policy::automatic);
    m.def("trans_vec_to_Pauli_operator", &transVecToPauliOperator<double>,
        "Transform vector to Pauli operator.\n"
        "\n"
        "Args:\n"
        "     vector: Input vector to be transformed.\n"
        "\n"
        "Returns:\n"
        "     Pauli operator equivalent of the input vector.\n");
    m.def("trans_Pauli_operator_to_vec", &transPauliOperatorToVec,
        "Transform Pauli operator to vector.\n"
        "\n"
        "Args:\n"
        "     operator: Input Pauli operator to be transformed.\n"
        "\n"
        "Returns:\n"
        "     Vector equivalent of the input Pauli operator.\n");
}
