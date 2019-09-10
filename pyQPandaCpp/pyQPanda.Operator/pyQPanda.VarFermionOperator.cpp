#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "Variational/VarFermionOperator.h"

namespace py = pybind11;
USING_QPANDA

void initVarFermionOperator(py::module& m)
{
    py::class_<VarFermionOperator>(m, "VarFermionOperator")
        .def(py::init<>())
        .def(py::init<const complex_var&>())
        .def(py::init<const std::string&, const complex_var&>())
        .def(py::init<const VarFermionOperator::FermionMap&>())
        .def("normal_ordered", &VarFermionOperator::normal_ordered)
        .def("isEmpty", &VarFermionOperator::isEmpty)
        .def("setErrorThreshold", &VarFermionOperator::setErrorThreshold)
        .def("error_threshold", &VarFermionOperator::error_threshold)
        .def("data", &VarFermionOperator::data)
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
        .def("toString", &VarFermionOperator::toString)
        .def("__str__", &VarFermionOperator::toString);
}
