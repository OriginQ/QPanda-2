#pragma once

#include "Core/QuantumNoise/AbstractNoiseNode.h"

QPANDA_BEGIN

class QNoise : public AbstractQNoiseNode
{
private:
    std::shared_ptr<AbstractQNoiseNode> m_noise_node;

public:
    QNoise(std::shared_ptr<AbstractQNoiseNode>);
    virtual ~QNoise(){}
    virtual QVec get_qvec();
    virtual QStat get_ops();
    std::shared_ptr<AbstractQNoiseNode> getImplementationPtr();
};

QPANDA_END
