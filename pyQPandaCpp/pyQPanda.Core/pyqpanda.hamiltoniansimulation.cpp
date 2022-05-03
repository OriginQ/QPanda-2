#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "Components/HamiltonianSimulation/HamiltonianSimulation.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"

USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;

void export_hamiltoniansimulation(py::module &m)
{
	m.def("expMat", &expMat, "calculate the matrix power of e", py::return_value_policy::reference);

	py::class_<QOperator>(m, "QOperator")
		.def(py::init<>())
		.def(py::init<QGate &>())
		.def(py::init<QCircuit &>())
		.def("get_matrix", &QOperator::get_matrix)
		.def("to_instruction", &QOperator::to_instruction);
}
