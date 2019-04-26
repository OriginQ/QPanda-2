/*! \file QGateCounter.h */
#ifndef _QGATECOUNTER_H
#define _QGATECOUNTER_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/Traversal.h"

QPANDA_BEGIN

/**
* @namespace QPanda
*/
/**
* @class QGateCounter
* @ingroup Utilities
* @brief Count quantum gate num in quantum program, quantum circuit, quantum while, quantum if
*/
class QGateCounter : public TraversalInterface
{
public:
    QGateCounter();
    ~QGateCounter();

    template <typename _Ty>
    void traversal(_Ty &node)
    {
        static_assert(std::is_base_of<QNode, _Ty>::value, "Bad Node Type");
        Traversal::traversalByType(&node, &node, this);
    }
    size_t count();
private:
    virtual void execute(AbstractQGateNode * cur_node, QNode * parent_node);
    virtual void execute(AbstractQuantumMeasure * cur_node, QNode * parent_node);
    size_t m_count;
};

/**
* @brief  Count quantum gate num under quantum program, quantum circuit, quantum while, quantum if
* @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
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
            auto count = getQGateNumber(&circuit);
            std::cout << "QCircuit count: " << count << std::endl;

            finalize();
    * @endcode
*/

template <typename _Ty>
size_t getQGateNumber(_Ty &node)
{
    static_assert(std::is_base_of<QNode, _Ty>::value, "bad node type");
    QGateCounter counter;
    counter.traversal(node);
    return counter.count();
}

QPANDA_END
#endif // _QGATECOUNTER_H



