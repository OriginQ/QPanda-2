#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrix.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrixNoise.h"

QPANDA_BEGIN

class DensityMatrixSimulator : public QVM
{
public:

    void init(bool is_double_precision = true);

    double get_probability(QProg& node, size_t index);
    double get_probability(QProg& node, std::string bin_index);

    prob_vec get_probabilities(QProg& node);
    prob_vec get_probabilities(QProg& node, QVec qubits);
    prob_vec get_probabilities(QProg& node, Qnum qubits);
    prob_vec get_probabilities(QProg& node, std::vector<std::string> bin_indices);
    
    double get_expectation(QProg& node, const QHamiltonian&, const QVec&);
    double get_expectation(QProg& node, const QHamiltonian&, const Qnum&);

    cmatrix_t  get_density_matrix(QProg& node);

    cmatrix_t  get_reduced_density_matrix(QProg& node, const QVec&);
    cmatrix_t  get_reduced_density_matrix(QProg& node, const Qnum&);

    /* bit-flip, phase-flip, bit-phase-flip, phase-damping, amplitude-damping, depolarizing*/
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob);
    void set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const QVec& qubits);
    void set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob, const QVec& qubits);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const std::vector<QVec>& qubits);

    /*decoherence error*/
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double t_gate);
    void set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double T1, double T2, double t_gate);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double t_gate, const QVec& qubits);
    void set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double T1, double T2, double t_gate, const QVec& qubits);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double t_gate, const std::vector<QVec>& qubits);

protected:

    void run(QProg& node, bool reset_state = true);

    void apply_gate(std::shared_ptr<AbstractQGateNode> gate_node);
    void apply_gate_with_noisy(std::shared_ptr<AbstractQGateNode> gate_node);
    
private:

    DensityMatrixNoise m_noisy;
    std::shared_ptr<AbstractDensityMatrix> m_simulator = nullptr;
};

QPANDA_END