#include "Core/Core.h"
#include "QPanda.h"
#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"


USING_QPANDA

namespace py = pybind11;

void export_noise_model(py::module &m)
{
    py::class_<NoiseModel>(m, "Noise")
        .def(py::init<>())
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_type"),
             py::arg("prob"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_types"),
             py::arg("prob"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, const QVec &>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_type"),
             py::arg("prob"),
             py::arg("qubits"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, const QVec &>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_types"),
             py::arg("prob"),
             py::arg("qubits"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, const std::vector<QVec> &>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_type"),
             py::arg("prob"),
             py::arg("qubits"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_type"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("t_gate"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_types"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("t_gate"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const QVec &>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_type"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("t_gate"),
             py::arg("qubits"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double, const QVec &>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_types"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("t_gate"),
             py::arg("qubits"))
        .def("add_noise_model",
             pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const std::vector<QVec> &>(&NoiseModel::add_noise_model),
             py::arg("noise_model"),
             py::arg("gate_type"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("t_gate"),
             py::arg("qubits"))
        .def("set_measure_error",
             pybind11::overload_cast<const NOISE_MODEL &, double, const QVec &>(&NoiseModel::set_measure_error),
             py::arg("noise_model"),
             py::arg("prob"),
             py::arg("qubits") = QVec())
        .def("set_measure_error",
             pybind11::overload_cast<const NOISE_MODEL &, double, double, double, const QVec &>(&NoiseModel::set_measure_error),
             py::arg("noise_model"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("t_gate"),
             py::arg("qubits") = QVec())
        .def("add_mixed_unitary_error",
             pybind11::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &>(&NoiseModel::add_mixed_unitary_error),
             py::arg("gate_types"),
             py::arg("unitary_matrices"),
             py::arg("probs"))
        .def("add_mixed_unitary_error",
             pybind11::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &, const QVec &>(&NoiseModel::add_mixed_unitary_error),
             py::arg("gate_types"),
             py::arg("unitary_matrices"),
             py::arg("probs"),
             py::arg("qubits"))
        .def("add_mixed_unitary_error",
             pybind11::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &, const std::vector<QVec> &>(&NoiseModel::add_mixed_unitary_error),
             py::arg("gate_types"),
             py::arg("unitary_matrices"),
             py::arg("probs"),
             py::arg("qubits"))
        .def("set_reset_error",
             &NoiseModel::set_reset_error,
             py::arg("p0"),
             py::arg("p1"),
             py::arg("qubits"))
        .def("set_readout_error",
             &NoiseModel::set_readout_error,
             py::arg("prob_list"),
             py::arg("qubits") = QVec())
        .def("set_rotation_error",
             &NoiseModel::set_rotation_error,
             py::arg("error"));
}