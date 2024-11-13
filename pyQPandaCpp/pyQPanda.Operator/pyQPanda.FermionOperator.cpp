#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "Components/Operator/FermionOperator.h"

USING_QPANDA
namespace py = pybind11;

void export_FermionOperator(py::module &m)
{
	py::class_<FermionOperator>(m, "FermionOperator")
		.def(py::init<>())
		.def(py::init<double>())
		.def(py::init<const complex_d&>())
		.def(py::init<const std::string&, const complex_d&>())
		.def(py::init<const FermionOperator::FermionMap&>())
		.def("normal_ordered", &FermionOperator::normal_ordered,
			"Returns the normal ordered form of the fermion operator.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A new FermionOperator in normal ordered form.")
		.def("error_threshold", &FermionOperator::error_threshold,
			"Retrieve the error threshold for the fermion operator.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A double representing the error threshold.\n")
		.def("data", &FermionOperator::data,
			"Get the data of the fermion operator.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A data structure representing the fermion operator's data.\n")
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
		.def("__str__", &FermionOperator::toString)

		/*will delete*/
		.def("isEmpty", &FermionOperator::isEmpty)
		.def("setErrorThreshold", &FermionOperator::setErrorThreshold)
		.def("toString", &FermionOperator::toString)
		/*new interface*/
		.def("is_empty", &FermionOperator::isEmpty,
			"Check if the fermion operator is empty.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A boolean indicating whether the operator is empty.\n")
		.def("set_error_threshold", &FermionOperator::setErrorThreshold,
			"Set the error threshold for the fermion operator.\n"
			"\n"
			"Args:\n"
			"     threshold: A double representing the new error threshold.\n"
			"\n"
			"Returns:\n"
			"     None.\n")
		.def("to_string", &FermionOperator::toString,
			"Convert the fermion operator to a string representation.\n"
			"\n"
			"Args:\n"
			"     None\n"
			"\n"
			"Returns:\n"
			"     A string representing the fermion operator.\n");
}
