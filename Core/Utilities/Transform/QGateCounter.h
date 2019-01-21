#ifndef _QGATECOUNTER_H
#define _QGATECOUNTER_H
#include "QuantumCircuit/QProgram.h"
QPANDA_BEGIN
class QGateCounter
{
private:
    static size_t countControlFlowQGate(QNode * pNode);
public:
    QGateCounter();
    ~QGateCounter();
    static size_t countQGate(AbstractQuantumCircuit *);
    static size_t countQGate(AbstractQuantumProgram *);
    static size_t countQGate(AbstractControlFlowNode *);
};
size_t countQGateUnderQCircuit(AbstractQuantumCircuit * pQCircuit);
size_t countQGateUnderQProg(AbstractQuantumProgram * pQProg);
QPANDA_END
#endif // !_STATISTICS_QGATE_COUNT_ALGORITHM



