#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/Clifford.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/StablizerNoise.h"

QPANDA_BEGIN

class Stabilizer : public QVM
{
public:

    void init();

    /* bit-flip, phase-flip, bit-phase-flip, phase-damping, depolarizing*/
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob);
    void set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const QVec& qubits);
    void set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob, const QVec& qubits);
    void set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const std::vector<QVec>& qubits);

    //get monte-carlo measure result with shots
    std::map<std::string, size_t> runWithConfiguration(QProg &prog, int shots);
    
    //get probs
    prob_dict probRunDict(QProg &, QVec, int select_max = -1);

protected:

    void run(QProg& node, bool reset_state = true);

    void apply_gate(std::shared_ptr<AbstractQGateNode> gate_node);
    void apply_reset(std::shared_ptr<AbstractQuantumReset> gate_node);

private:

    StablizerNoise m_noisy;
    std::shared_ptr<AbstractClifford> m_simulator = nullptr;
};

QPANDA_END