#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "Components/Operator/VarPauliOperator.h"

namespace py = pybind11;
USING_QPANDA

void export_VarPauliOperator(py::module& m)
{
    py::class_<VarPauliOperator>(m, "VarPauliOperator")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<const complex_var&>())
        .def(py::init<const std::string&, const complex_var&>())
        .def(py::init<const VarPauliOperator::PauliMap&>())
        .def("dagger", &VarPauliOperator::dagger,
            "Return the adjoint (dagger) of the Pauli operator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A new VarPauliOperator representing the adjoint.\n")
        .def("data", &VarPauliOperator::data,
            "Get the data of the variable Pauli operator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A data structure representing the variable Pauli operator's data.\n")
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self + complex_var())
        .def(py::self * complex_var())
        .def(py::self - complex_var())
        .def(complex_var() + py::self)
        .def(complex_var() * py::self)
        .def(complex_var() - py::self)
        .def("__str__", &VarPauliOperator::toString)

        /*will delete*/
        .def("toHamiltonian", &VarPauliOperator::toHamiltonian)
        .def("getMaxIndex", &VarPauliOperator::getMaxIndex)
        .def("isEmpty", &VarPauliOperator::isEmpty)
        .def("isAllPauliZorI", &VarPauliOperator::isAllPauliZorI)
        .def("setErrorThreshold", &VarPauliOperator::setErrorThreshold)
        .def("error_threshold", &VarPauliOperator::error_threshold)
        .def("remapQubitIndex", &VarPauliOperator::remapQubitIndex)
        .def("toString", &VarPauliOperator::toString)
        /*new interface*/
        .def("to_hamiltonian", &VarPauliOperator::toHamiltonian,
            "Convert the variable Pauli operator to a Hamiltonian representation.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A Hamiltonian representation of the operator.\n")
        .def("get_maxIndex", &VarPauliOperator::getMaxIndex,
            "Retrieve the maximum index used in the Pauli operator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     An integer representing the maximum index.\n")
        .def("is_empty", &VarPauliOperator::isEmpty,
            "Check if the variable Pauli operator is empty.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A boolean indicating whether the operator is empty.\n")
        .def("is_all_pauli_z_or_i", &VarPauliOperator::isAllPauliZorI,
            "Check if the operator consists only of Pauli Z or identity.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A boolean indicating if all operators are Z or I.\n")
        .def("set_error_threshold", &VarPauliOperator::setErrorThreshold,
            "Set the error threshold for the variable Pauli operator.\n"
            "\n"
            "Args:\n"
            "     threshold (double): A double representing the new error threshold.\n"
            "\n"
            "Returns:\n"
            "     None.\n")
        .def("error_threshold", &VarPauliOperator::error_threshold,
            "Retrieve the error threshold for the variable Pauli operator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A double representing the error threshold.\n")
        .def("remap_qubit_index", &VarPauliOperator::remapQubitIndex,
            "Remap the qubit indices of the variable Pauli operator.\n"
            "\n"
            "Args:\n"
            "     A mapping of old indices to new indices.\n"
            "\n"
            "Returns:\n"
            "     None.\n")
        .def("to_string", &VarPauliOperator::toString,
            "Convert the variable Pauli operator to a string representation.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A string representing the variable Pauli operator.\n");
}
