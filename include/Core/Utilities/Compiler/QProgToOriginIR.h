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
#include "Core/Utilities/Tools/Traversal.h"

QPANDA_BEGIN

/**
* @class QProgToOriginIR
* @ingroup Utilities
* @brief QuantumProg Transform To OriginIR instruction sets.
*/
class QProgToOriginIR : public TraversalInterface<>
{
public:
    QProgToOriginIR(QuantumMachine * quantum_machine);
    ~QProgToOriginIR() {};

    /**
    * @brief  Transform quantum program
    * @param[in]  QProg&    quantum program
    * @return     void
    */
    virtual void transform(QProg &prog) {};

    template<typename _Ty>
    void traversal(_Ty &node)
    {
        m_OriginIR.emplace_back("QINIT " + std::to_string(m_quantum_machine->getAllocateQubit()));
        m_OriginIR.emplace_back("CREG " + std::to_string(m_quantum_machine->getAllocateCMem()));

		execute(node.getImplementationPtr(), nullptr);
    }

	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node);
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node);
    virtual void execute(std::shared_ptr<AbstractQNoiseNode> cur_node, std::shared_ptr<QNode> parent_node);
    virtual void execute(std::shared_ptr<AbstractQDebugNode> cur_node, std::shared_ptr<QNode> parent_node);

    /**
     * @brief  get OriginIR insturction set
     * @return     std::string
     */
    virtual std::string getInsturctions();
private:
    virtual void transformQGate(AbstractQGateNode*, bool is_dagger = false);
    virtual void transformQMeasure(AbstractQuantumMeasure*);
	virtual void transformQReset(AbstractQuantumReset *pReset);
	virtual void transformClassicalProg(AbstractClassicalProg *);
	virtual void transformQControlFlow(AbstractControlFlowNode *){}
    std::vector<std::string> m_OriginIR;/**< OriginIR insturction vector */
    std::map<int, std::string>  m_gatetype; /**< quantum gate type map */
    QuantumMachine * m_quantum_machine;
};

/**

* @brief  Quantum Program Transform To OriginIR  
* @ingroup Utilities
* @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
* @return     std::string    OriginIR instruction set
* @see
      @code
          init(QuantumMachine_type::CPU);

          auto qubit = qAllocMany(6);
          auto cbit  = cAllocMany(2);
          auto prog = CreateEmptyQProg();

          prog << CZ(qubit[0], qubit[2]) << H(qubit[1]) << CNOT(qubit[1], qubit[2])
          << RX(qubit[0],pi/2) << Measure(qubit[1],cbit[1]);
		  extern QuantumMachine* global_quantum_machine;
          std::cout << transformQProgToOriginIR(prog, global_quantum_machine) << std::endl;
          finalize();
      @endcode
*/
template<typename _Ty>
std::string transformQProgToOriginIR(_Ty &node,QuantumMachine *machine)
{
    if (nullptr == machine)
    {
        QCERR("Quantum machine is nullptr");
        throw std::invalid_argument("Quantum machine is nullptr");
    }

    QProgToOriginIR OriginIRTraverse(machine);
	OriginIRTraverse.traversal(node);
    return OriginIRTraverse.getInsturctions();
}


/**
* @brief  Convert Quantum Program  To OriginIR
* @ingroup Utilities
* @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
* @param[in]  QuantumMachine* quantum machine
* @return     std::string   OriginIR instruction set
*/
template<typename _Ty>
std::string convert_qprog_to_originir(_Ty &node, QuantumMachine *machine)
{
	return transformQProgToOriginIR(node, machine);
}

/**
* @brief  write prog to originir file
* @ingroup Utilities
* @param[in] QProg&   Quantum Program
* @param[in] QuantumMachine*  quantum machine pointer
* @param[in] const std::string	originir file name
* @return
*/
void write_to_originir_file(QProg prog, QuantumMachine * qvm, const std::string file_name);

QPANDA_END
#endif
