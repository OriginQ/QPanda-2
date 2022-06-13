#pragma once

#include <cmath>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/QuantumGateParameter.h"

QPANDA_BEGIN
/**
 * @brief singleton debugger for QPU virtual machine
 * get all qubits register state vector while running QProg
 */
class QPUDebugger
{
public:
    struct State{
        std::vector<std::complex<float>>* float_state{nullptr}; 
        QStat *double_state{nullptr};
    };

    static QPUDebugger &instance();
    void save_qstate_ref(std::vector<std::complex<double>> &stat);
    void save_qstate_ref(std::vector<std::complex<float>> &stat);
    const State &get_qstate() const;

private:
    QPUDebugger() = default;
    ~QPUDebugger() = default;

    State m_qstate;
};

QPANDA_END