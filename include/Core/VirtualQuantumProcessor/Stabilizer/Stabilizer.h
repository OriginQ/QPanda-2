#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/Clifford.h"

QPANDA_BEGIN

class Stabilizer : public QVM
{
public:

    void init();

    //get monte-carlo measure result with shots
    std::map<std::string, size_t> runWithConfiguration(QProg &prog, int shots, const NoiseModel& = NoiseModel());
    
    //get probs
    prob_dict probRunDict(QProg &, QVec, int select_max = -1);

protected:

    void run(QProg& node, bool reset_state = true);

    void apply_gate(std::shared_ptr<AbstractQGateNode> gate_node);
    void apply_reset(std::shared_ptr<AbstractQuantumReset> gate_node);

private:

    std::shared_ptr<AbstractClifford> m_simulator = nullptr;
};

QPANDA_END