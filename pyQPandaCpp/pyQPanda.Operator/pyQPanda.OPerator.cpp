#include "pybind11/pybind11.h"

namespace py = pybind11;

void initPauliOperator(py::module&);
void initFermionOperator(py::module&);
void initVarPauliOperator(py::module&);
void initVarFermionOperator(py::module&);

PYBIND11_MODULE(pyQPandaOperator, m)
{
    m.doc() = "";

    initPauliOperator(m);
    initFermionOperator(m);
    initVarPauliOperator(m);
    initVarFermionOperator(m);
}
