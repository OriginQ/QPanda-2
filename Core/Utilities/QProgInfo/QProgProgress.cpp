#include "Core/Utilities/QProgInfo/QProgProgress.h"

USING_QPANDA

size_t QProgProgress::get_processed_gate_num(uint64_t exec_id)
{
    if (m_prog_exec_gates.count(exec_id))
    {
        return m_prog_exec_gates.at(exec_id);
    }
    else
    {
        return 0;
    }
}

size_t QProgProgress::update_processed_gate_num(uint64_t exec_id, size_t count)
{
    if (m_prog_exec_gates.count(exec_id))
    {
        m_prog_exec_gates.at(exec_id) += count;
        return m_prog_exec_gates.at(exec_id);
    }
    else
    {
        return 0;
    }
}

void QProgProgress::prog_start(uint64_t exec_id)
{
    if (m_prog_exec_gates.count(exec_id))
    {
        m_prog_exec_gates.at(exec_id) = 0;
    }
    else
    {
        m_prog_exec_gates[exec_id] = 0;
    }
}

void QProgProgress::prog_end(uint64_t exec_id)
{
    if (m_prog_exec_gates.count(exec_id))
    {
        m_prog_exec_gates.erase(exec_id);
    }
}
