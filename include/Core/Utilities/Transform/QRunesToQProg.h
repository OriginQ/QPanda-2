/*! \file QRunesToQProg.h */
#ifndef  _QRUNESTOQPROG_H_
#define  _QRUNESTOQPROG_H_
#include <functional>
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Traversal.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

QPANDA_BEGIN
/**
* @namespace QPanda
* @namespace QGATE_SPACE
*/

/**
* @class QRunesToQProg
* @ingroup Utilities
* @brief Transform QRunes instruction set To Quantum program
*/
class QRunesToQProg
{
public:
    QRunesToQProg();
    ~QRunesToQProg() = default;

    void qRunesParser(std::string, QProg&, QuantumMachine*);

    QuantumMachine * qvm;
private:
    size_t traversalQRunes(size_t, QNode*);

    size_t handleSingleGate(QNode*);

    size_t handleDoubleGate(QNode*);

    size_t handleAngleGate(QNode*);

    size_t handleDoubleAngleGate(QNode*);

    size_t handleMeasureGate(QNode*);

    size_t handleDaggerCircuit(QNode*, size_t);

    size_t handleControlCircuit(QNode*, size_t);

    std::vector<std::string> m_QRunes;
    std::vector<std::string> m_QRunes_value;

    std::map<std::string, std::function<QGate(Qubit *)> > m_singleGateFunc;
    std::map<std::string, std::function<QGate(Qubit *, Qubit*)> > m_doubleGateFunc;
    std::map<std::string, std::function<QGate(Qubit *, double)> > m_angleGateFunc;
    std::map<std::string, std::function<QGate(Qubit *, Qubit*, double)> > m_doubleAngleGateFunc;
};


/**
* @brief   QRunes instruction set transform to quantum program interface
* @ingroup Utilities
* @param[in]  QProg&   empty quantum program
* @return    void
* @exception    qprog_syntax_error   quantum program syntax error
*/
void transformQRunesToQProg(std::string, QProg&, QuantumMachine *);
QPANDA_END

#endif
