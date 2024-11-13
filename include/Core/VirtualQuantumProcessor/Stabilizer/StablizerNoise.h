#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseSimulator.h"

QPANDA_BEGIN

class StablizerNoise
{
public:

    bool enabled();
    bool enabled(GateType type, Qnum qubits);

    QProg generate_noise_prog(QProg& source_prog);

    KarusError get_karus_error(GateType type, const Qnum& qubits);
    QGate matrix_to_clifford_gate(const QStat& matrix, Qubit* qubit);

    /* bit-flip, phase-flip, bit-phase-flip, phase-damping, depolarizing*/
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const Qnum& qubits_vec);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const std::vector<Qnum>& qubits_vecs);

private:

    QStat m_i = { 1, 0, 0, 1 };
    QStat m_x = { 0, 1, 1, 0 };
    QStat m_y = { 0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0 };
    QStat m_z = { 1, 0, 0, -1 };

    //Qubits config for GateType
    std::map<GateType, Qnum> m_single_qubits;
    std::map<GateType, std::vector<DoubleQubits>> m_double_qubits;

private:

    std::vector<std::tuple<GateType, int, KarusError>> m_one_qubit_karus_error_tuple;
    std::vector<std::tuple<GateType, int, int, KarusError>> m_two_qubit_karus_error_tuple;

    void set_gate_and_qnum(GateType type, const Qnum& qubits);
    void set_gate_and_qnums(GateType type, const std::vector<Qnum>& qubits);

    void set_single_karus_error_tuple(GateType type, const KarusError &karus_error, const Qnum& qubits);
    void set_double_karus_error_tuple(GateType type, const KarusError &karus_error, const std::vector<Qnum>& qubits);

    void update_karus_error_tuple(GateType type, int tar_qubit, const KarusError& karus_error);
    void update_karus_error_tuple(GateType type, int ctr_qubit, int tar_qubit, const KarusError& karus_error);
};

QPANDA_END