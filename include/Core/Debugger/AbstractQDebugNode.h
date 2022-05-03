#pragma once

#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

class AbstractQDebugNode
{
public:
    AbstractQDebugNode() {}
    virtual ~AbstractQDebugNode() {}
    virtual void save_qstate(QStat& stat) = 0;
};

QPANDA_END