#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumNoise/AbstractNoiseNode.h"

QPANDA_BEGIN

class OriginNoise : public QNode, public AbstractQNoiseNode
{
public:
    OriginNoise(QVec, QStat = QStat());
    virtual ~OriginNoise() {}
    virtual NodeType getNodeType() const
    {
        return NodeType::NOISE_NODE;
    }
    virtual QVec get_qvec();
    virtual QStat get_ops();

protected:
    QVec m_vec;
    QStat m_op;
};

QPANDA_END