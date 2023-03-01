#include "Core/Core.h"
#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "QPanda.h"
#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"


USING_QPANDA
namespace py = pybind11;

void export_enum(py::module &m)
{
    py::enum_<QMachineType>(m, "QMachineType", "Quantum machine type")
        .value("CPU", QMachineType::CPU)
        .value("GPU", QMachineType::GPU)
        .value("CPU_SINGLE_THREAD", QMachineType::CPU_SINGLE_THREAD)
        .value("NOISE", QMachineType::NOISE)
        .export_values();

    py::enum_<NOISE_MODEL>(m, "NoiseModel", "noise model type")
        .value("DAMPING_KRAUS_OPERATOR", NOISE_MODEL::DAMPING_KRAUS_OPERATOR)
        .value("DECOHERENCE_KRAUS_OPERATOR", NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR)
        .value("DEPHASING_KRAUS_OPERATOR", NOISE_MODEL::DEPHASING_KRAUS_OPERATOR)
        .value("PAULI_KRAUS_MAP", NOISE_MODEL::PAULI_KRAUS_MAP)
        .value("BITFLIP_KRAUS_OPERATOR", NOISE_MODEL::BITFLIP_KRAUS_OPERATOR)
        .value("DEPOLARIZING_KRAUS_OPERATOR", NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR)
        .value("BIT_PHASE_FLIP_OPRATOR", NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR)
        .value("PHASE_DAMPING_OPRATOR", NOISE_MODEL::PHASE_DAMPING_OPRATOR);

    py::enum_<GateType>(m, "GateType", "quantum gate type")
        .value("GATE_NOP", GateType::GATE_NOP)
        .value("GATE_UNDEFINED", GateType::GATE_UNDEFINED)
        .value("P0_GATE", GateType::P0_GATE)
        .value("P1_GATE", GateType::P1_GATE)
        .value("PAULI_X_GATE", GateType::PAULI_X_GATE)
        .value("PAULI_Y_GATE", GateType::PAULI_Y_GATE)
        .value("PAULI_Z_GATE", GateType::PAULI_Z_GATE)
        .value("X_HALF_PI", GateType::X_HALF_PI)
        .value("Y_HALF_PI", GateType::Y_HALF_PI)
        .value("Z_HALF_PI", GateType::Z_HALF_PI)
        .value("HADAMARD_GATE", GateType::HADAMARD_GATE)
        .value("T_GATE", GateType::T_GATE)
        .value("S_GATE", GateType::S_GATE)
        .value("P_GATE", GateType::P_GATE)
        .value("CP_GATE", GateType::CP_GATE)
        .value("RX_GATE", GateType::RX_GATE)
        .value("RY_GATE", GateType::RY_GATE)
        .value("RZ_GATE", GateType::RZ_GATE)
        .value("RXX_GATE", GateType::RXX_GATE)
        .value("RYY_GATE", GateType::RYY_GATE)
        .value("RZZ_GATE", GateType::RZZ_GATE)
        .value("RZX_GATE", GateType::RZX_GATE)
        .value("U1_GATE", GateType::U1_GATE)
        .value("U2_GATE", GateType::U2_GATE)
        .value("U3_GATE", GateType::U3_GATE)
        .value("U4_GATE", GateType::U4_GATE)
        .value("CU_GATE", GateType::CU_GATE)
        .value("CNOT_GATE", GateType::CNOT_GATE)
        .value("CZ_GATE", GateType::CZ_GATE)
        .value("CPHASE_GATE", GateType::CPHASE_GATE)
        .value("ISWAP_THETA_GATE", GateType::ISWAP_THETA_GATE)
        .value("ISWAP_GATE", GateType::ISWAP_GATE)
        .value("SQISWAP_GATE", GateType::SQISWAP_GATE)
        .value("SWAP_GATE", GateType::SWAP_GATE)
        .value("TWO_QUBIT_GATE", GateType::TWO_QUBIT_GATE)
        .value("P00_GATE", GateType::P00_GATE)
        .value("P11_GATE", GateType::P11_GATE)
        .value("TOFFOLI_GATE", GateType::TOFFOLI_GATE)
        .value("ORACLE_GATE", GateType::ORACLE_GATE)
        .value("I_GATE", GateType::I_GATE)
        .value("BARRIER_GATE", GateType::BARRIER_GATE)
        .value("RPHI_GATE", GateType::RPHI_GATE)
        .export_values();

    py::enum_<ChipID>(m, "ChipID", "origin quantum real chip type")
        .value("Simulation", ChipID::Simulation)
        .value("WUYUAN_1", ChipID::WUYUAN_1)
        .value("WUYUAN_2", ChipID::WUYUAN_2)
        .value("WUYUAN_3", ChipID::WUYUAN_3)
        .export_values();

    py::enum_<NodeType>(m, "NodeType", "quantum node type")
        .value("NODE_UNDEFINED", NodeType::NODE_UNDEFINED)
        .value("GATE_NODE", NodeType::GATE_NODE)
        .value("CIRCUIT_NODE", NodeType::CIRCUIT_NODE)
        .value("PROG_NODE", NodeType::PROG_NODE)
        .value("MEASURE_GATE", NodeType::MEASURE_GATE)
        .value("WHILE_START_NODE", NodeType::WHILE_START_NODE)
        .value("QIF_START_NODE", NodeType::QIF_START_NODE)
        .value("CLASS_COND_NODE", NodeType::CLASS_COND_NODE)
        .value("RESET_NODE", NodeType::RESET_NODE);

    py::enum_<SingleGateTransferType>(m, "SingleGateTransferType", "Quantum single gate transfer type")
        .value("SINGLE_GATE_INVALID", SINGLE_GATE_INVALID)
        .value("ARBITRARY_ROTATION", ARBITRARY_ROTATION)
        .value("DOUBLE_CONTINUOUS", DOUBLE_CONTINUOUS)
        .value("SINGLE_CONTINUOUS_DISCRETE", SINGLE_CONTINUOUS_DISCRETE)
        .value("DOUBLE_DISCRETE", DOUBLE_DISCRETE)
        .export_values();

    py::enum_<DoubleGateTransferType>(m, "DoubleGateTransferType", "Quantum double gate transfer type")
        .value("DOUBLE_GATE_INVALID", DOUBLE_GATE_INVALID)
        .value("DOUBLE_BIT_GATE", DOUBLE_BIT_GATE)
        .export_values();

    py::enum_<QError>(m, "QError", "Quantum QError Type")
        .value("UndefineError", QError::undefineError)
        .value("qErrorNone", QError::qErrorNone)
        .value("qParameterError", QError::qParameterError)
        .value("qubitError", QError::qubitError)
        .value("loadFileError", QError::loadFileError)
        .value("initStateError", QError::initStateError)
        .value("destroyStateError", QError::destroyStateError)
        .value("setComputeUnitError", QError::setComputeUnitError)
        .value("runProgramError", QError::runProgramError)
        .value("getResultError", QError::getResultError)
        .value("getQStateError", QError::getQStateError);

    py::enum_<QCircuitOPtimizerMode>(m, "QCircuitOPtimizerMode", "Quantum circuit optimize mode")
        .value("Merge_H_X", QCircuitOPtimizerMode::Merge_H_X)
        .value("Merge_U3", QCircuitOPtimizerMode::Merge_U3)
        .value("Merge_RX", QCircuitOPtimizerMode::Merge_RX)
        .value("Merge_RY", QCircuitOPtimizerMode::Merge_RY)
        .value("Merge_RZ", QCircuitOPtimizerMode::Merge_RZ)
        .def(
            "__or__",
            [](QCircuitOPtimizerMode &self, QCircuitOPtimizerMode &other)
            {
                return self | other;
            },
            "bitwise or",
            py::return_value_policy::reference);

    py::enum_<DecompositionMode>(m, "DecompositionMode", "Quantum matrix decomposition mode")
        .value("QR", DecompositionMode::QR)
        .value("HOUSEHOLDER_QR", DecompositionMode::HOUSEHOLDER_QR)
		.value("QSDecomposition", DecompositionMode::QSD);
    py::enum_<LATEX_GATE_TYPE>(m, "LATEX_GATE_TYPE", "Quantum latex gate type")
        .value("GENERAL_GATE", LATEX_GATE_TYPE::GENERAL_GATE)
        .value("CNOT_GATE", LATEX_GATE_TYPE::X)
        .value("SWAP_GATE", LATEX_GATE_TYPE::SWAP)
        .export_values();

    py::enum_<BackendType>(m, "BackendType", "Quantum machine backend type")
        .value("CPU", BackendType::CPU)
        .value("GPU", BackendType::GPU)
        .value("CPU_SINGLE_THREAD", BackendType::CPU_SINGLE_THREAD)
        .value("NOISE", BackendType::NOISE)
        .value("MPS", BackendType::MPS)
        .export_values();

    py::enum_<DAGNodeType>(m, "DAGNodeType", "Quantum dag node type")
        .value("NUKNOW_SEQ_NODE_TYPE", DAGNodeType::NUKNOW_SEQ_NODE_TYPE)
        .value("MAX_GATE_TYPE", DAGNodeType::MAX_GATE_TYPE)
        .value("MEASURE", DAGNodeType::MEASURE)
        .value("QUBIT", DAGNodeType::QUBIT)
        .value("RESET", DAGNodeType::RESET);

    py::enum_<AnsatzGateType>(m, "AnsatzGateType", "Quantum ansatz gate type")
        .value("AGT_X", AnsatzGateType::AGT_X)
        .value("AGT_H", AnsatzGateType::AGT_H)
        .value("AGT_RX", AnsatzGateType::AGT_RX)
        .value("AGT_RY", AnsatzGateType::AGT_RY)
        .value("AGT_RZ", AnsatzGateType::AGT_RZ)
        .export_values();
}