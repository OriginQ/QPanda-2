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
    py::class_<NoiseModel>(m, "Noise", "Quantum machine for noise simulation")
        .def(py::init<>(),
            "Initialize a new NoiseModel instance.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A new NoiseModel instance.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_type"),
            py::arg("prob"),
            "Add a noise model to the noise simulation.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_type: The type of gate to which the noise model applies.\n"
            "\n"
            "     prob: The probability of the noise occurring.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_types"),
            py::arg("prob"),
            "Add a noise model to multiple gate types.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_types: A vector of gate types to which the noise model applies.\n"
            "\n"
            "     prob: The probability of the noise occurring.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, const QVec &>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_type"),
            py::arg("prob"),
            py::arg("qubits"),
            "Add a noise model to a specific gate with targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_type: The type of gate to which the noise model applies.\n"
            "\n"
            "     prob: The probability of the noise occurring.\n"
            "\n"
            "     qubits: A vector of qubit indices that the noise affects.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, const QVec &>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_types"),
            py::arg("prob"),
            py::arg("qubits"),
            "Add a noise model to multiple gate types with targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_types: A vector of gate types to which the noise model applies.\n"
            "\n"
            "     prob: The probability of the noise occurring.\n"
            "\n"
            "     qubits: A vector of qubit indices that the noise affects.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, const std::vector<QVec> &>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_type"),
            py::arg("prob"),
            py::arg("qubits"),
            "Add a noise model to a specific gate with targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_type: The type of gate to which the noise model applies.\n"
            "\n"
            "     prob: The probability of the noise occurring.\n"
            "\n"
            "     qubits: A vector of qubit indices that the noise affects.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_type"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t_gate"),
            "Add a noise model to a specific gate with time parameters.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_type: The type of gate to which the noise model applies.\n"
            "\n"
            "     t1: The time constant for relaxation (T1).\n"
            "\n"
            "     t2: The time constant for dephasing (T2).\n"
            "\n"
            "     t_gate: The duration of the gate operation.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_types"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t_gate"),
            "Add a noise model to multiple gate types with time parameters.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_types: A vector of gate types to which the noise model applies.\n"
            "\n"
            "     t1: The time constant for relaxation (T1).\n"
            "\n"
            "     t2: The time constant for dephasing (T2).\n"
            "\n"
            "     t_gate: The duration of the gate operation.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const QVec &>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_type"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t_gate"),
            py::arg("qubits"),
            "Add a noise model to a specific gate with time parameters and targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_type: The type of gate to which the noise model applies.\n"
            "\n"
            "     t1: The time constant for relaxation (T1).\n"
            "\n"
            "     t2: The time constant for dephasing (T2).\n"
            "\n"
            "     t_gate: The duration of the gate operation.\n"
            "\n"
            "     qubits: A vector of qubit indices that the noise affects.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double, const QVec &>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_types"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t_gate"),
            py::arg("qubits"),
            "Add a noise model to multiple gate types with specified time parameters and targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_types: A vector of gate types to which the noise model applies.\n"
            "\n"
            "     t1: The time constant for relaxation (T1).\n"
            "\n"
            "     t2: The time constant for dephasing (T2).\n"
            "\n"
            "     t_gate: The duration of the gate operation.\n"
            "\n"
            "     qubits: A vector of qubit indices that the noise affects.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_noise_model",
            pybind11::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const std::vector<QVec> &>(&NoiseModel::add_noise_model),
            py::arg("noise_model"),
            py::arg("gate_type"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t_gate"),
            py::arg("qubits"),
            "Add a noise model to a specific gate with specified time parameters and targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be added.\n"
            "\n"
            "     gate_type: The type of gate to which the noise model applies.\n"
            "\n"
            "     t1: The time constant for relaxation (T1).\n"
            "\n"
            "     t2: The time constant for dephasing (T2).\n"
            "\n"
            "     t_gate: The duration of the gate operation.\n"
            "\n"
            "     qubits: A vector of vectors of qubit indices that the noise affects.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("set_measure_error",
            pybind11::overload_cast<const NOISE_MODEL &, double, const QVec &>(&NoiseModel::set_measure_error),
            py::arg("noise_model"),
            py::arg("prob"),
            py::arg("qubits") = QVec(),
            "Set the measurement error for specified qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be used.\n"
            "\n"
            "     prob: The probability of measurement error.\n"
            "\n"
            "     qubits: A vector of qubit indices to which the measurement error applies. Defaults to an empty QVec.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("set_measure_error",
            pybind11::overload_cast<const NOISE_MODEL &, double, double, double, const QVec &>(&NoiseModel::set_measure_error),
            py::arg("noise_model"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t_gate"),
            py::arg("qubits") = QVec(),
            "Set the measurement error using time parameters for the specified qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: An instance of NOISE_MODEL to be used.\n"
            "\n"
            "     t1: The time constant for relaxation (T1).\n"
            "\n"
            "     t2: The time constant for dephasing (T2).\n"
            "\n"
            "     t_gate: The duration of the gate operation.\n"
            "\n"
            "     qubits: A vector of qubit indices to which the measurement error applies. Defaults to an empty QVec.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_mixed_unitary_error",
            pybind11::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &>(&NoiseModel::add_mixed_unitary_error),
            py::arg("gate_types"),
            py::arg("unitary_matrices"),
            py::arg("probs"),
            "Add mixed unitary errors to specified gate types.\n"
            "\n"
            "Args:\n"
            "     gate_types: The type of gates to which the mixed unitary errors apply.\n"
            "\n"
            "     unitary_matrices: A vector of unitary matrices representing the errors.\n"
            "\n"
            "     probs: A vector of probabilities corresponding to each unitary matrix.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_mixed_unitary_error",
            pybind11::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &, const QVec &>(&NoiseModel::add_mixed_unitary_error),
            py::arg("gate_types"),
            py::arg("unitary_matrices"),
            py::arg("probs"),
            py::arg("qubits"),
            "Add mixed unitary errors to specified gate types with targeted qubits.\n"
            "\n"
            "Args:\n"
            "     gate_types: The type of gates to which the mixed unitary errors apply.\n"
            "\n"
            "     unitary_matrices: A vector of unitary matrices representing the errors.\n"
            "\n"
            "     probs: A vector of probabilities corresponding to each unitary matrix.\n"
            "\n"
            "     qubits: A vector of qubit indices that the mixed unitary errors affect.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("add_mixed_unitary_error",
            pybind11::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &, const std::vector<QVec> &>(&NoiseModel::add_mixed_unitary_error),
            py::arg("gate_types"),
            py::arg("unitary_matrices"),
            py::arg("probs"),
            py::arg("qubits"),
            "Add mixed unitary errors to specified gate types for multiple qubits.\n"
            "\n"
            "Args:\n"
            "     gate_types: The type of gates to which the mixed unitary errors apply.\n"
            "\n"
            "     unitary_matrices: A vector of unitary matrices representing the errors.\n"
            "\n"
            "     probs: A vector of probabilities corresponding to each unitary matrix.\n"
            "\n"
            "     qubits: A vector of QVec instances indicating which qubits the errors affect.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("set_reset_error",
            &NoiseModel::set_reset_error,
            py::arg("p0"),
            py::arg("p1"),
            py::arg("qubits"),
            "Set reset errors for specified qubits.\n"
            "\n"
            "Args:\n"
            "     p0: The probability of resetting to state |0>.\n"
            "\n"
            "     p1: The probability of resetting to state |1>.\n"
            "\n"
            "     qubits: A vector of qubit indices that the reset errors apply to.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("set_readout_error",
            &NoiseModel::set_readout_error,
            py::arg("prob_list"),
            py::arg("qubits") = QVec(),
            "Set readout errors for specified qubits.\n"
            "\n"
            "Args:\n"
            "     prob_list: A list of probabilities for readout errors.\n"
            "\n"
            "     qubits: A vector of qubit indices that the readout errors apply to (default is all qubits).\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        )
        .def("set_rotation_error",
            &NoiseModel::set_rotation_error,
            py::arg("error"),
            "Set rotation error for gates.\n"
            "\n"
            "Args:\n"
            "     error: The error model for rotation operations.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
        );
}