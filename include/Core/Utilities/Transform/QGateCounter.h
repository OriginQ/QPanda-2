/*! \file QGateCounter.h */
#ifndef _QGATECOUNTER_H
#define _QGATECOUNTER_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
QPANDA_BEGIN
/**
* @namespace QPanda
*/
/**
* @class QGateCounter
* @ingroup Utilities
* @brief Count quantum gate num in quantum program
*/
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
/**
* @brief  Count quantum gate num under quantum circuit
* @param[in]  AbstractQuantumCircuit*  Abstract quantum circuit pointer
* @return     size_t  Quantum gate num
* @exception  invalid_argument Abstract Quantum circuit pointer is a nullptr
* @see
    * @code
            init();
            auto qubits = qAllocMany(4);
            auto cbits = cAllocMany(4);

            auto circuit = CreateEmptyCircuit();
            circuit << H(qubits[0]) << X(qubits[1]) << S(qubits[2])
            << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI/4);
            auto count = countQGateUnderQCircuit(&circuit);
            std::cout << "QCircuit count: " << count << std::endl;

            finalize();
    * @endcode
*/
size_t countQGateUnderQCircuit(AbstractQuantumCircuit * pQCircuit);

/**
* @brief  Count quantum gate num under quantum program
* @param[in]  AbstractQuantumCircuit*  Abstract quantum program pointer
* @return     size_t  Quantum gate num
* @exception  invalid_argument Abstract quantum program pointer is a nullptr
* @see
    * @code
            init();
            auto qubits = qAllocMany(4);
            auto cbits = cAllocMany(4);

            auto prog = CreateEmptyQProg();
            prog << Y(qubits[0]) << CZ(qubits[2], qubits[3]) << circuit;

            count = countQGateUnderQProg(&prog);
            std::cout << "QProg count: " << count << std::endl;

            finalize();
    * @endcode
*/
size_t countQGateUnderQProg(AbstractQuantumProgram * pQProg);
QPANDA_END
#endif // !_STATISTICS_QGATE_COUNT_ALGORITHM



