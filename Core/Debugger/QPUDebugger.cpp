#include "Core/Debugger/QPUDebugger.h"

USING_QPANDA

QPUDebugger &QPUDebugger::instance()
{
  static QPUDebugger debugger;
  return debugger;
}

void QPUDebugger::save_qstate(QStat &stat)
{
  m_qstate = &stat;
}

const QStat &QPUDebugger::get_qtate() const
{
  QPANDA_ASSERT(m_qstate == nullptr, "QVM state vector not saved yet.");
  return *m_qstate;
}