#ifndef  _TRAVERSALQPROG_H_
#define  _TRAVERSALQPROG_H_
#include "QPanda.h"
QPANDA_BEGIN

class TraversalQProg
{
public:
    size_t m_qubit_num;
    virtual ~TraversalQProg() {};

    virtual void traversal(AbstractQuantumProgram*) final;
    virtual void traversal(AbstractQuantumCircuit*) final;
    virtual void traversal(QNode*) final;
    virtual void handleDaggerNode(QNode*, int) final;
    virtual void handleDaggerCircuit(QNode*) final;

private:

    virtual void traversal(AbstractQGateNode *) = 0;
    virtual void traversalAll(AbstractQuantumProgram *) = 0;
};

QPANDA_END
#endif
