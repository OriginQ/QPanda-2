#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "Components/Operator/FermionOperator.h"

USING_QPANDA
namespace py = pybind11;


void initFermionOperator(py::module& m)
{
    m.doc() = "";

	py::class_<FermionOperator>(m, "FermionOperator")
		.def(py::init<>())
		.def(py::init<const complex_d&>())
		.def(py::init<const std::string&, const complex_d&>())
		.def(py::init<const FermionOperator::FermionMap&>())
		.def("normal_ordered", &FermionOperator::normal_ordered)
		.def("error_threshold", &FermionOperator::error_threshold)
		.def("data", &FermionOperator::data)
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
		.def("__str__", &FermionOperator::toString)

		/*will delete*/
		.def("isEmpty", &FermionOperator::isEmpty)
		.def("setErrorThreshold", &FermionOperator::setErrorThreshold)
		.def("toString", &FermionOperator::toString)
		/*new interface*/
		.def("is_empty", &FermionOperator::isEmpty)
		.def("set_error_threshold", &FermionOperator::setErrorThreshold)
		.def("to_string", &FermionOperator::toString);
}
