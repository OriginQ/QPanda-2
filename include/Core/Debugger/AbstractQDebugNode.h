#pragma once

#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

class AbstractQDebugNode
{
public:
    AbstractQDebugNode() {}
    virtual ~AbstractQDebugNode() {}
    virtual void save_qstate_ref(std::vector<std::complex<double>> &stat) = 0;
    virtual void save_qstate_ref(std::vector<std::complex<float>> &stat) = 0;
};

QPANDA_END