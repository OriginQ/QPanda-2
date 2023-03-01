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
        .def(py::init<const complex_var &>())
        .def(py::init<const std::string &, const complex_var&>())
        .def(py::init<const VarPauliOperator::PauliMap &>())
        .def("dagger", &VarPauliOperator::dagger)
		.def("data", &VarPauliOperator::data)
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
		.def("to_hamiltonian", &VarPauliOperator::toHamiltonian)
		.def("get_maxIndex", &VarPauliOperator::getMaxIndex)
		.def("is_empty", &VarPauliOperator::isEmpty)
		.def("is_all_pauli_z_or_i", &VarPauliOperator::isAllPauliZorI)
		.def("set_error_threshold", &VarPauliOperator::setErrorThreshold)
		.def("error_threshold", &VarPauliOperator::error_threshold)
		.def("remap_qubit_index", &VarPauliOperator::remapQubitIndex)
		.def("to_string", &VarPauliOperator::toString);
}
