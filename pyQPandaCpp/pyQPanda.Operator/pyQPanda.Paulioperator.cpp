#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/operators.h"
#include "Components/Operator/PauliOperator.h"

USING_QPANDA
namespace py = pybind11;


void initPauliOperator(py::module& m)
{
	py::class_<PauliOperator>(m, "PauliOperator")
		.def(py::init<>())
		.def(py::init<>([](const complex_d &val)
	        { return PauliOperator(val); }))
		.def(py::init<>([](const std::string &key, const complex_d &val)
	        { return PauliOperator(key, val); }))
		.def(py::init<>([](const PauliOperator::PauliMap &map)
	        { return PauliOperator(map); }))
		.def("dagger", &PauliOperator::dagger)
		.def("data", &PauliOperator::data)
		.def("error_threshold", &PauliOperator::error_threshold)
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
		.def("__str__", &PauliOperator::toString)

		/*will delete*/
		.def("toHamiltonian", &PauliOperator::toHamiltonian)
		.def("getMaxIndex", &PauliOperator::getMaxIndex)
		.def("isEmpty", &PauliOperator::isEmpty)
		.def("isAllPauliZorI", &PauliOperator::isAllPauliZorI)
		.def("setErrorThreshold", &PauliOperator::setErrorThreshold)		
		.def("remapQubitIndex", &PauliOperator::remapQubitIndex)
		.def("toString", &PauliOperator::toString)
		/*new interface*/
		.def("to_hamiltonian", &PauliOperator::toHamiltonian)
		.def("get_max_index", &PauliOperator::getMaxIndex)
		.def("is_empty", &PauliOperator::isEmpty)
		.def("is_all_pauli_z_or_i", &PauliOperator::isAllPauliZorI)
		.def("set_error_threshold", &PauliOperator::setErrorThreshold)
		.def("remap_qubit_index", &PauliOperator::remapQubitIndex)
		.def("to_string", &PauliOperator::toString);

	m.def("trans_vec_to_Pauli_operator", &transVecToPauliOperator<double>, "Transfrom vector to pauli operator", py::return_value_policy::reference);
	m.def("trans_Pauli_operator_to_vec", &transPauliOperatorToVec, "Transfrom Pauli operator to vector", py::return_value_policy::reference);

}
