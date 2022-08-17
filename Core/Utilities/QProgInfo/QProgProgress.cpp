#include "Core/Utilities/QProgInfo/QProgProgress.h"

USING_QPANDA

size_t QProgProgress::get_processed_gate_num(uint64_t exec_id)
{
	size_t ret = 0;
	m_mutex.lock();
    if (m_prog_exec_gates.count(exec_id)){
        ret  = m_prog_exec_gates.at(exec_id);
    }
	m_mutex.unlock();
    return ret;
}

size_t QProgProgress::update_processed_gate_num(uint64_t exec_id, size_t count)
{
    size_t ret = 0;
	m_mutex.lock();
    if (m_prog_exec_gates.count(exec_id))
    {
        m_prog_exec_gates.at(exec_id) += count;
        ret  = m_prog_exec_gates.at(exec_id);
    }
	m_mutex.unlock();
    return ret;
}

void QProgProgress::prog_start(uint64_t exec_id)
{
    m_mutex.lock();
    if (m_prog_exec_gates.count(exec_id)){
        m_prog_exec_gates.at(exec_id) = 0;
    }
    else{
        m_prog_exec_gates[exec_id] = 0;
    }
    m_mutex.unlock();
}

void QProgProgress::prog_end(uint64_t exec_id)
{
    m_mutex.lock();
    if (m_prog_exec_gates.count(exec_id)){
        m_prog_exec_gates.erase(exec_id);
    }
    m_mutex.unlock();
}
