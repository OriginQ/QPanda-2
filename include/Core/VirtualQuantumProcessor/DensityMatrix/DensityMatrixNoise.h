#pragma once
#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseSimulator.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/VectorMatrix.h"

QPANDA_BEGIN

class DensityMatrixNoise
{
public:

    bool enabled();
    bool enabled(GateType type, Qnum qubits);

    std::vector<KarusError> get_karus_error(GateType type, const Qnum& qubits);

    /* karus matrix error */
    void set_noise_model(const std::vector<cmatrix_t>& karus_matrices);
    void set_noise_model(const std::vector<cmatrix_t>& karus_matrices, const std::vector<GateType>& types);

    /* bit-flip, phase-flip, bit-phase-flip, phase-damping, amplitude-damping, depolarizing*/

    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const Qnum& qubits_vec);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const std::vector<Qnum>& qubits_vecs);

    /*decoherence error*/
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double time_param);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double time_param, const Qnum& qubits_vecs);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double time_param, const std::vector<Qnum>& qubits_vecs);

private:

    //Qubits config for GateType
    std::map<GateType, Qnum> m_single_qubits;
    std::map<GateType, std::vector<DoubleQubits>> m_double_qubits;

private:

    std::vector<std::tuple<GateType, int, std::vector<KarusError>>> m_one_qubit_karus_error_tuple;
    std::vector<std::tuple<GateType, int, int, std::vector<KarusError>>> m_two_qubit_karus_error_tuple;

    void set_gate_and_qnum(GateType type, const Qnum& qubits);
    void set_gate_and_qnums(GateType type, const std::vector<Qnum>& qubits);

    void set_single_karus_error_tuple(GateType type, const KarusError &karus_error, const Qnum& qubits);
    void set_double_karus_error_tuple(GateType type, const KarusError &karus_error, const std::vector<Qnum>& qubits);

    void update_karus_error_tuple(GateType type, int tar_qubit, const KarusError& karus_error);
    void update_karus_error_tuple(GateType type, int ctr_qubit, int tar_qubit, const KarusError& karus_error);
};

QPANDA_END