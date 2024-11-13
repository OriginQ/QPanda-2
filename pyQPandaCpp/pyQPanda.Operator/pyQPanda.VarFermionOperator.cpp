#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "Components/Operator/VarFermionOperator.h"


namespace py = pybind11;
USING_QPANDA

void export_VarFermionOperator(py::module &m)
{
	py::class_<VarFermionOperator>(m, "VarFermionOperator")
		.def(py::init<>())
		.def(py::init<double>())
		.def(py::init<const complex_var&>())
		.def(py::init<const std::string&, const complex_var&>())
		.def(py::init<const VarFermionOperator::FermionMap&>())
		.def("normal_ordered", &VarFermionOperator::normal_ordered,
			"Returns the normal ordered form of the variable fermion operator.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A new VarFermionOperator in normal ordered form.\n")
		.def("data", &VarFermionOperator::data,
			"Get the data of the variable fermion operator.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A data structure representing the variable fermion operator's data.\n")
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
		.def("__str__", &VarFermionOperator::toString)

		/*will delete*/
		.def("isEmpty", &VarFermionOperator::isEmpty)
		.def("setErrorThreshold", &VarFermionOperator::setErrorThreshold)
		.def("toString", &VarFermionOperator::toString)
		/*new interface*/
		.def("is_empty", &VarFermionOperator::isEmpty,
			"Check if the variable fermion operator is empty.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A boolean indicating whether the operator is empty.")
		.def("set_error_threshold", &VarFermionOperator::setErrorThreshold,
			"Set the error threshold for the variable fermion operator.\n"
			"\n"
			"Args:\n"
			"     threshold (double): A double representing the new error threshold.\n"
			"\n"
			"Returns:\n"
			"     None.\n")
		.def("error_threshold", &VarFermionOperator::error_threshold,
			"Retrieve the error threshold for the variable fermion operator.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A double representing the error threshold.\n")
		.def("to_string", &VarFermionOperator::toString,
			"Convert the variable fermion operator to a string representation.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A string representing the variable fermion operator.\n");
}
