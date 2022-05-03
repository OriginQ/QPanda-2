#pragma once

#include "AbstractQDebugNode.h"
#include "QPUDebugger.h"
#include "Core/QuantumCircuit/QNode.h"

QPANDA_BEGIN

class OriginDebug : public QNode, public AbstractQDebugNode
{
public:
    OriginDebug() {}
    virtual ~OriginDebug() {}
    virtual NodeType getNodeType() const
    {
        return NodeType::DEBUG_NODE;
    }
    virtual void save_qstate(QStat& stat)
    {
        QPUDebugger::instance().save_qstate(stat);
    }
};

QPANDA_END