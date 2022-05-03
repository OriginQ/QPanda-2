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
    static QPUDebugger &instance();
    void save_qstate(QStat &stat);
    const QStat &get_qtate() const;

private:
    QPUDebugger() = default;
    ~QPUDebugger() = default;

    QStat *m_qstate{nullptr};
};

QPANDA_END