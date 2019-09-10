#ifndef  _TRAVERSALQPROG_H_
#define  _TRAVERSALQPROG_H_
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"

QPANDA_BEGIN

class TraversalQProg
{
public:
    size_t m_qubit_num;
    virtual ~TraversalQProg() {};

    virtual void traversalByFile(std::string file_path) final;
    virtual void traversalAll(AbstractQuantumProgram *) {};

    virtual void traversal(AbstractQuantumProgram*) final;
    virtual void traversal(AbstractQuantumMeasure*) {};
    virtual void traversal(AbstractQuantumCircuit*) final;
    virtual void traversal(QNode*) final;
    virtual void handleDaggerNode(QNode*, int) final;
    virtual void handleDaggerCircuit(QNode*) final;

private:

    virtual void traversal(AbstractQGateNode *) = 0;

};

QPANDA_END
#endif
