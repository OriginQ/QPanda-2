#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "Variational/VarPauliOperator.h"

namespace py = pybind11;
USING_QPANDA

void initVarPauliOperator(py::module& m)
{
    py::class_<VarPauliOperator>(m, "VarPauliOperator")
        .def(py::init<>())
        .def(py::init<>([](double& val)
            { return VarPauliOperator(val); }))
        .def(py::init<>([](const complex_var &val)
            { return VarPauliOperator(val); }))
        .def(py::init<>([](const std::string &key, const complex_var&val)
            { return VarPauliOperator(key, val); }))
        .def(py::init<>([](const VarPauliOperator::PauliMap &map)
            { return VarPauliOperator(map); }))
        .def("dagger", &VarPauliOperator::dagger)
        .def("toHamiltonian", &VarPauliOperator::toHamiltonian)
        .def("getMaxIndex", &VarPauliOperator::getMaxIndex)
        .def("isEmpty", &VarPauliOperator::isEmpty)
        .def("isAllPauliZorI", &VarPauliOperator::isAllPauliZorI)
        .def("setErrorThreshold", &VarPauliOperator::setErrorThreshold)
        .def("error_threshold", &VarPauliOperator::error_threshold)
        .def("remapQubitIndex", &VarPauliOperator::remapQubitIndex)
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
        .def("toString", &VarPauliOperator::toString)
        .def("__str__", &VarPauliOperator::toString);
}
