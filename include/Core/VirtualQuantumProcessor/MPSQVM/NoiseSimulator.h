#ifndef NOISE_SIMULATOR_H
#define NOISE_SIMULATOR_H

#include "QPanda.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSImplQPU.h"
#include "Core/VirtualQuantumProcessor/RandomEngine/RandomEngine.h"

using DoubleQubits = std::pair<size_t, size_t>;

enum class NoiseConfigMethod
{
    NO_CONFIG = 0,
    NORMAL_CONFIG,
    GLOBAL_CONFIG
};

class TensorNoiseModelConfig
{
public:
    void set_model(NOISE_MODEL model) { m_model = model; }
    void set_qubits(GateType, std::vector<size_t> qubits);
    void set_params(GateType, const std::vector<double> params);
    void set_golbal_params(const std::vector<double> params) { m_global_params = params; }
    std::vector<double> get_golbal_params() {return m_global_params; }

    bool is_config(GateType, QVec);
    NOISE_MODEL get_model() { return m_model; }
    std::vector<double> get_params(GateType);
    std::vector<size_t> get_single_qubits(GateType);

private:
    NOISE_MODEL m_model;
    std::vector<double> m_global_params;
    std::map<GateType, std::vector<double>> m_params;
    std::map<GateType, std::vector<size_t>> m_single_qubits;
    std::map<GateType, std::vector<DoubleQubits>> m_double_qubits;
};

class TensorNoiseGenerator : public TraversalInterface<bool>
{
public:
    TensorNoiseGenerator() :m_mps_qpu(nullptr) {}
    TensorNoiseGenerator(const TensorNoiseModelConfig& model, MPSImplQPU* mps_qpu) : m_noise_model(model), m_mps_qpu(mps_qpu) {}
  
    void handle_quantum_gate(std::shared_ptr<AbstractQGateNode> gate, bool is_dagger);

    NoiseConfigMethod get_noise_method() { return m_method; }
    
    void set_noise_model(NOISE_MODEL model, std::vector<double> params_vec);
    void set_noise_model(NOISE_MODEL model, std::string gate, Qnum qubits_vec, std::vector<double> params_vec);

    //The next 2 set_noise_model functions is only appear in DECOHERENCE_KRAUS_OPERATOR
    void set_noise_model(NOISE_MODEL model, std::vector<double> T_params_vec, std::vector<double> time_params_vec);
    void set_noise_model(NOISE_MODEL model, std::string gate, Qnum qubits_vec, std::vector<double> T_params_vec, std::vector<double> time_params_vec);

    void set_mps_qpu_and_result(std::shared_ptr<MPSImplQPU> mps_qpu, QResult* result) { m_mps_qpu = mps_qpu; m_result = result; }
    void set_mps_qresult(QResult* qresult) { m_result = qresult; }

    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumReset>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>, QCircuitConfig &config);

private:

    QResult* m_result;
    std::shared_ptr<MPSImplQPU> m_mps_qpu;
    TensorNoiseModelConfig m_noise_model;
    NoiseConfigMethod m_method = NoiseConfigMethod::NO_CONFIG;

    void handle_flip_noise_model(NOISE_MODEL model, const std::vector<double>&, QVec);
    void handle_amplitude_damping_noise_model(const std::vector<double>&, QVec);
    void handle_phase_damping_noise_model(const std::vector<double>&, QVec);
    void handle_decoherence_noise_model(const std::vector<double>&, QVec);
    
    void handle_depolarizing_noise_model(const std::vector<double>&, QVec);
    void handle_noise_gate(const std::vector<double>& params, QVec targets);
};
#endif  //!NOISE_SIMULATOR_H