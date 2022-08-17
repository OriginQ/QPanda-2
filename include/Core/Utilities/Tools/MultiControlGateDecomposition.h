#ifndef MULTI_CONTROL_GATE_DECOMPOSITION_H
#define MULTI_CONTROL_GATE_DECOMPOSITION_H

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "QProgFlattening.h"
QPANDA_BEGIN

/**
* @brief  multi control gate decomposition for U in SU(2)
* @param[in]  QProg、QCircuit、QGate
* @return[out]   QProg
* @see form paper : (LDD)Linear-depth quantum circuits for multi-qubit controlled gates(2022)
*/

class LinearDepthDecomposition : public QNodeDeepCopy
{
public:
    LinearDepthDecomposition() {}
    ~LinearDepthDecomposition() {}

    template <typename T>
    QProg decompose(T& node)
    {
        m_ldd_prog.clear();

        execute(std::dynamic_pointer_cast<QNode>(node.getImplementationPtr()),
            std::dynamic_pointer_cast<QNode>(m_ldd_prog.getImplementationPtr()));
        return m_ldd_prog;
    }

    void execute(std::shared_ptr<QNode>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>);

private:
    QCircuit Qn(QVec qubits, QStat matrix);
    QCircuit PnU(QVec qubits, QStat matrix);
    QCircuit CnU(QVec qubits, QStat matrix);
    QCircuit PnRx(QVec qubits, QStat matrix);
    QCircuit CnRx(QVec qubits, QStat matrix);

    QProg m_ldd_prog;
};

template <typename T>
QProg ldd_decompose(T& node)
{
    QProg temp(node);
    flatten(temp);
    LinearDepthDecomposition ldd;
    return ldd.decompose(temp);
}


QPANDA_END
#endif // MULTI_CONTROL_GATE_DECOMPOSITION_H