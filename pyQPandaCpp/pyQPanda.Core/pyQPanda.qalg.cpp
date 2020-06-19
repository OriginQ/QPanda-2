#include <math.h>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "QPandaConfig.h"
#include "QPanda.h"


using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

//template<>
//struct py::detail::type_caster<QVec>
//    : py::detail::list_caster<QVec, Qubit*> { };

void init_qalg(py::module & m)
{
	m.def("MAJ", &MAJ, "Quantum adder MAJ module", py::return_value_policy::reference);
	m.def("UMA", &UMA, "Quantum adder UMA module", py::return_value_policy::reference);
	m.def("MAJ2", &MAJ2, "Quantum adder MAJ2 module", py::return_value_policy::reference);
	m.def("isCarry", &isCarry, "Construct a circuit to determine if there is a carry", py::return_value_policy::reference);
	m.def("QAdder", &QAdder, "Quantum adder", py::return_value_policy::reference);
	m.def("QAdderIgnoreCarry", &QAdderIgnoreCarry, "Quantum adder ignore carry", py::return_value_policy::reference);

    m.def("amplitude_encode", &amplitude_encode, "Encode the input data to the amplitude of qubits", "qlist"_a, "data"_a,
        py::return_value_policy::automatic
        );

	m.def("QFT", &QFT, "Build QFT quantum circuit", "qlist"_a,
		py::return_value_policy::automatic
	);

	m.def("QPE", [](const QVec control_qubits, const QVec target_qubits, QStat matrix, bool b_estimate_eigenvalue = false) {
		return build_QPE_circuit(control_qubits, target_qubits, matrix, b_estimate_eigenvalue);
	}
		, "qlist"_a, "qlist"_a, "matrix"_a, "bool"_a = false,
		"Build QPE quantum circuit",
		py::return_value_policy::automatic_reference
		);

	m.def("HHL", &build_HHL_circuit, "Build HHL quantum circuit", "matrix"_a, "data"_a, "QuantumMachine"_a,
		py::return_value_policy::automatic
	);

	m.def("HHL_solve_linear_equations", &HHL_solve_linear_equations, "use HHL to solve linear equations", "matrix"_a, "data"_a,
		py::return_value_policy::automatic
	);
}
