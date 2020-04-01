/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgToQASM.h
Author: Yulei
Updated in 2019/04/09 14:39

Classes for QProgToQASM.

*/
/*! \file QProgToQASM.h */
#ifndef  _QPROGTOQASM_H_
#define  _QPROGTOQASM_H_
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
* @brief IBM currently provides the back-end runtime environment
*/
enum IBMQBackends
{
	IBMQ_QASM_SIMULATOR = 0,
	IBMQ_16_MELBOURNE,
	IBMQX2,
	IBMQX4
};


/**
* @class QProgToQASM
* @ingroup Utilities
* @brief Quantum Prog Transform To QASM instruction sets
*/
class QProgToQASM : public TraversalInterface<bool&>
{
public:
    QProgToQASM(QuantumMachine * quantum_machine, IBMQBackends ibmBackend = IBMQ_QASM_SIMULATOR);
    ~QProgToQASM() {}
    /**
    * @brief  get QASM insturction set
    * @return     std::string  
    */
    virtual std::string getInsturctions();

    /*!
    * @brief  Transform Quantum program
    * @param[in]  QProg&  quantum program
    * @return     void  
    */
    virtual void transform(QProg &);

	static std::string getIBMQBackendName(IBMQBackends typeNum);
	
	public:
	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node,bool & ) ;
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, bool &) ;
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, bool &) ;
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, bool &) ;
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, bool &);

private:
	/**
	* @brief  Transform Quantum program by Traversal algorithm, refer to class Traversal
	* @param[in]  QProg&  quantum program
	* @return     void
	*/
	virtual void transformQProgByTraversalAlg(QProg *prog);
	virtual void transformQGate(AbstractQGateNode*, bool is_dagger);

    virtual void transformQMeasure(AbstractQuantumMeasure*);
	virtual void transformQReset(AbstractQuantumReset*);

    std::map<int, std::string>  m_gatetype; /**< Quantum gatetype map   */
    std::vector<std::string> m_qasm; /**< QASM instructin vector   */
    QuantumMachine * m_quantum_machine;
	IBMQBackends _ibmBackend;
};

    /**
    * @brief  Quantum program transform to qasm instruction set
    * @ingroup Utilities
    * @param[in]  QProg&   Quantum Program 
	* @param[in]   QuantumMachine*  quantum machine pointer
	* @param[in]	 IBMQBackends	ibmBackend = IBMQ_QASM_SIMULATOR
    * @return     std::string    QASM instruction set
    * @see
        * @code
                init(QuantumMachine_type::CPU);

                auto qubit = qAllocMany(6);
                auto cbit  = cAllocMany(2);
                auto prog = CreateEmptyQProg();

                prog << CZ(qubit[0], qubit[2]) << H(qubit[1]) << CNOT(qubit[1], qubit[2])
                << RX(qubit[0],pi/2) << Measure(qubit[1],cbit[1]);

				extern QuantumMachine* global_quantum_machine;
                std::cout << transformQProgToQASM(prog, global_quantum_machine) << std::endl;
                finalize();
        * @endcode
    */
    std::string transformQProgToQASM(QProg &pQProg, QuantumMachine * quantum_machine, IBMQBackends ibmBackend = IBMQ_QASM_SIMULATOR);

/**
* @brief  Convert Quantum program  to QASM instruction set
* @ingroup Utilities
* @param[in]  QProg&   Quantum Program
* @param[in]   QuantumMachine*  quantum machine pointer
* @param[in]	 IBMQBackends	ibmBackend = IBMQ_QASM_SIMULATOR
* @return     std::string    QASM instruction set
*/
std::string convert_qprog_to_qasm(QProg &prog, QuantumMachine* qm, IBMQBackends ibmBackend = IBMQ_QASM_SIMULATOR);
	
QPANDA_END
#endif
