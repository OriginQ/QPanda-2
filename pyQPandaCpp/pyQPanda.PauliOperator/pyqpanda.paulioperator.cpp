#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "PauliOperator/PauliOperator.h"

namespace py = pybind11;
USING_QPANDA
PYBIND11_MODULE(pyQPandaPauliOperator, m)
{
    m.doc() = "";

    py::class_<QPanda::PauliOperator>(m, "PauliOperator")
        .def(py::init<>([](const QPanda::complex_d &val)
            { return QPanda::PauliOperator(val); }))
        .def(py::init<>([](const QPanda::QPauliMap &map)
            { return QPanda::PauliOperator(map); }))
        .def("dagger", &QPanda::PauliOperator::dagger)
        .def("toHamiltonian", &QPanda::PauliOperator::toHamiltonian)
        .def("getMaxIndex", &QPanda::PauliOperator::getMaxIndex)
        .def("isEmpty", &QPanda::PauliOperator::isEmpty)
        .def("isAllPauliZorI", &QPanda::PauliOperator::isAllPauliZorI)
        .def("setErrorThreshold", &QPanda::PauliOperator::setErrorThreshold)
        .def("error_threshold", &QPanda::PauliOperator::error_threshold)
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
        .def("toString", &QPanda::PauliOperator::toString)
        .def("__str__", &QPanda::PauliOperator::toString);
}
