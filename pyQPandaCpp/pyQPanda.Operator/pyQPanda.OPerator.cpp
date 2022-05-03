#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "Core/Variational/complex_var.h"
#include "Core/Variational/var.h"

namespace py = pybind11;

USING_QPANDA
namespace Var = Variational;

void export_PauliOperator(py::module &);
void export_FermionOperator(py::module &);
void export_VarPauliOperator(py::module &);
void export_VarFermionOperator(py::module &);

PYBIND11_MODULE(pyQPandaOperator, m)
{
    m.doc() = "";

    py::class_<complex_var>(m, "complex_var")
        .def(py::init<>())
        .def(py::init<Var::var>())
        .def(py::init<Var::var, Var::var>())
        .def("real", &complex_var::real)
        .def("imag", &complex_var::imag)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self);

    py::implicitly_convertible<Var::var, complex_var>();

    export_VarPauliOperator(m);
    export_VarFermionOperator(m);
    export_PauliOperator(m);
    export_FermionOperator(m);
}
