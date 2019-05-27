/*! \file QProgToQRunes.h */
#ifndef  _PROGTOQRUNES_H_
#define  _PROGTOQRUNES_H_
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Transform/QProgTransform.h"

QPANDA_BEGIN
/**
* @namespace QPanda
*/
/**
* @class QProgToQRunes
* @ingroup Utilities
* @brief QuantumProg Transform To QRunes instruction sets.
*/
class QProgToQRunes : public QProgTransform
{
public:
    QProgToQRunes(QuantumMachine * quantum_machine);
    ~QProgToQRunes() {};

    /**
    * @brief  Transform quantum program
    * @param[in]  QProg&    quantum program
    * @return     void
    * @exception  invalid_argument
    * @code
    * @endcode
    * @note
    */
    virtual void transform(QProg &prog) {};

    template<typename _Ty>
    void traversal(_Ty &node)
    {
        static_assert(std::is_base_of<QNode, _Ty>::value, "bad node type");

        m_QRunes.emplace_back("QINIT " + std::to_string(m_quantum_machine->getAllocateQubit()));
        m_QRunes.emplace_back("CREG " + std::to_string(m_quantum_machine->getAllocateCMem()));
        transformQNode(&node);
    }

    /**
     * @brief  get QRunes insturction set
     * @return     std::string
     * @exception
     * @note
     */
    virtual std::string getInsturctions();
private:
    virtual void transformQProg(AbstractQuantumProgram*);
    virtual void transformQGate(AbstractQGateNode*);
    virtual void transformQControlFlow(AbstractControlFlowNode*);
    virtual void transformQCircuit(AbstractQuantumCircuit*);
    virtual void transformQMeasure(AbstractQuantumMeasure*);
    virtual void transformQNode(QNode *);
    
    std::vector<std::string> m_QRunes;/**< QRunes insturction vector */
    std::map<int, std::string>  m_gatetype; /**< quantum gate type map */
    QuantumMachine * m_quantum_machine;
};

/**
* @brief  Quantum Program Transform To QRunes  instruction set
* @ingroup Utilities
* @param[in]  QProg&   quantum Program
* @return     std::string    QASM instruction set
* @see
      @code
          init(QuantumMachine_type::CPU);

          auto qubit = qAllocMany(6);
          auto cbit  = cAllocMany(2);
          auto prog = CreateEmptyQProg();

          prog << CZ(qubit[0], qubit[2]) << H(qubit[1]) << CNOT(qubit[1], qubit[2])
          << RX(qubit[0],pi/2) << Measure(qubit[1],cbit[1]);

          std::cout << transformQProgToQRunes(prog) << std::endl;
          finalize();
      @endcode
* @exception
* @note
*/
template<typename _Ty>
std::string transformQProgToQRunes(_Ty &node,QuantumMachine *machine)
{
    static_assert(std::is_base_of<QNode, _Ty>::value, "bad node type");

    if (nullptr == machine)
    {
        QCERR("Quantum machine is nullptr");
        throw std::invalid_argument("Quantum machine is nullptr");
    }

    QProgToQRunes qRunesTraverse(machine);
    qRunesTraverse.traversal(node);
    return qRunesTraverse.getInsturctions();
}
QPANDA_END
#endif
