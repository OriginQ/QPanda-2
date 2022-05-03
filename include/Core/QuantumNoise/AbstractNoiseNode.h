#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/QVec.h"

QPANDA_BEGIN

class AbstractQNoiseNode
{
public:
    virtual ~AbstractQNoiseNode() {}
    virtual QVec get_qvec() = 0;
    virtual QStat get_ops() = 0;
};

QPANDA_END