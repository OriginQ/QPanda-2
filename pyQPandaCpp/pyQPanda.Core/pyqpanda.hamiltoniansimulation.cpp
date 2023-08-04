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
    m.def("expMat", &expMat,
        "Calculate the matrix power of e.\n"
        "\n"
        "This function returns the power of matrix e.\n"
        "\n"
        "Args:\n"
        "     None\n"
        "\n"
        "Returns:\n"
        "     The computed matrix.\n",
        py::return_value_policy::reference
    );

    py::class_<QOperator>(m, "QOperator", "quantum operator class")
        .def(py::init<>(),
            "Initialize a new QOperator instance.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A new QOperator instance.\n"
        )
        .def(py::init<QGate &>(),
            "Initialize QOperator based on a quantum gate.\n"
            "\n"
            "Args:\n"
            "     qgate: An instance of QGate.\n"
            "\n"
            "Returns:\n"
            "     A new QOperator instance.\n"
        )
        .def(py::init<QCircuit &>(),
            "Initialize QOperator based on a quantum circuit.\n"
            "\n"
            "Args:\n"
            "     qcircuit: An instance of QCircuit.\n"
            "\n"
            "Returns:\n"
            "     A new QOperator instance.\n"
        )
        .def("get_matrix", &QOperator::get_matrix,
            "Retrieve the matrix representation of the QOperator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The matrix representation of the QOperator.\n"
        )
        .def("to_instruction", &QOperator::to_instruction,
            "Convert the QOperator to an instruction representation.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The instruction representation of the QOperator.\n"
        );
}
