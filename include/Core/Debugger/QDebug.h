#pragma once

#include <memory>
#include "AbstractQDebugNode.h"

QPANDA_BEGIN

class QDebug : public AbstractQDebugNode
{
private:
    std::shared_ptr<AbstractQDebugNode> m_debug_node;

public:
    QDebug(std::shared_ptr<AbstractQDebugNode> origin_debug)
        : m_debug_node(origin_debug)
    {
    }
    virtual ~QDebug() {}
    virtual void save_qstate_ref(std::vector<std::complex<double>>& stat) override
    {
        m_debug_node->save_qstate_ref(stat);
    }

    virtual void save_qstate_ref(std::vector<std::complex<float>>& stat) override
    {
        m_debug_node->save_qstate_ref(stat);
    }

    std::shared_ptr<AbstractQDebugNode> getImplementationPtr()
    {
        return m_debug_node;
    }
};

QPANDA_END