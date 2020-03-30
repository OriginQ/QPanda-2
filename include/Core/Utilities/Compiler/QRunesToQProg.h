/*! \file QRunesToQProg.h */
#ifndef  _QRUNESTOQPROG_H_
#define  _QRUNESTOQPROG_H_
#include <functional>
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"


QPANDA_BEGIN

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
    std::vector<ClassicalCondition> m_cbit_vec;
private:
    size_t traversalQRunes(size_t, std::shared_ptr<QNode>);

    size_t handleSingleGate(std::shared_ptr<QNode>);

    size_t handleDoubleGate(std::shared_ptr<QNode>);

    size_t handleAngleGate(std::shared_ptr<QNode>);

    size_t handleDoubleAngleGate(std::shared_ptr<QNode>);

    size_t handleToffoliGate(std::shared_ptr<QNode>);

    size_t handleMeasureGate(std::shared_ptr<QNode>);

    size_t handleDaggerCircuit(std::shared_ptr<QNode>, size_t);

    size_t handleControlCircuit(std::shared_ptr<QNode>, size_t);

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
*/
std::vector<ClassicalCondition> transformQRunesToQProg(std::string, QProg&, QuantumMachine *);
QPANDA_END

#endif
