/*! \file QProgToOriginIR.h */
#ifndef  _PROGTOORIGINIR_H_
#define  _PROGTOORIGINIR_H_
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Transform/QProgTransform.h"
#include "Core/Utilities/Traversal.h"

QPANDA_BEGIN
/**
* @namespace QPanda
*/
/**
* @class QProgToOriginIR
* @ingroup Utilities
* @brief QuantumProg Transform To OriginIR instruction sets.
*/
class QProgToOriginIR : public QProgTransform, public TraversalInterface<>
{
public:
    QProgToOriginIR(QuantumMachine * quantum_machine);
    ~QProgToOriginIR() {};

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

        m_OriginIR.emplace_back("QINIT " + std::to_string(m_quantum_machine->getAllocateQubit()));
        m_OriginIR.emplace_back("CREG " + std::to_string(m_quantum_machine->getAllocateCMem()));

		transformQProgByTraversalAlg(&node);
    }

	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node);

    /**
     * @brief  get OriginIR insturction set
     * @return     std::string
     * @exception
     * @note
     */
    virtual std::string getInsturctions();
private:
    virtual void transformQGate(AbstractQGateNode*, bool is_dagger = false);
    virtual void transformQMeasure(AbstractQuantumMeasure*);
	virtual void transformClassicalProg(AbstractClassicalProg *);
	virtual void transformQProgByTraversalAlg(QNode *node);
	virtual void transformQControlFlow(AbstractControlFlowNode *){}
    std::vector<std::string> m_OriginIR;/**< OriginIR insturction vector */
    std::map<int, std::string>  m_gatetype; /**< quantum gate type map */
    QuantumMachine * m_quantum_machine;
};

/**
* @brief  Quantum Program Transform To OriginIR  instruction set
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

          std::cout << transformQProgToOriginIR(prog) << std::endl;
          finalize();
      @endcode
* @exception
* @note
*/
template<typename _Ty>
std::string transformQProgToOriginIR(_Ty &node,QuantumMachine *machine)
{
    static_assert(std::is_base_of<QNode, _Ty>::value, "bad node type");

    if (nullptr == machine)
    {
        QCERR("Quantum machine is nullptr");
        throw std::invalid_argument("Quantum machine is nullptr");
    }

    QProgToOriginIR OriginIRTraverse(machine);
	OriginIRTraverse.traversal(node);
    return OriginIRTraverse.getInsturctions();
}
QPANDA_END
#endif
