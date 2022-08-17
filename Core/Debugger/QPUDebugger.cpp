#include "Core/Debugger/QPUDebugger.h"

USING_QPANDA

QPUDebugger &QPUDebugger::instance()
{
  static QPUDebugger debugger;
  return debugger;
}

void QPUDebugger::save_qstate_ref(std::vector<std::complex<double>> &state)
{
  m_qstate.double_state = &state;
  m_qstate.float_state = nullptr;
}

void QPUDebugger::save_qstate_ref(std::vector<std::complex<float>> &state)
{
  m_qstate.float_state = &state;
  m_qstate.double_state = nullptr;
}

const QPUDebugger::State &QPUDebugger::get_qstate() const
{
  QPANDA_ASSERT(m_qstate.double_state && m_qstate.float_state, "QVM state vector saved double complex same time.");
  QPANDA_ASSERT((!m_qstate.double_state) && (!m_qstate.float_state), "QVM state vector not saved yet.");
  return m_qstate;
}