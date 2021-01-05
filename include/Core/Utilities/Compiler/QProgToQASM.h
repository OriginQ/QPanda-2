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
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

/**
* @class QProgToQASM
* @ingroup Utilities
* @brief Quantum Prog Transform To QASM instruction sets
*/
class QProgToQASM : public TraverseByNodeIter
{
public:
    QProgToQASM(QProg src_prog, QuantumMachine * quantum_machine);
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
    virtual void transform();
	
public:
	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) ;
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) ;
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) ;
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) ;
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);

private:
	virtual void transformQGate(AbstractQGateNode*, bool is_dagger);

    virtual void transformQMeasure(AbstractQuantumMeasure*);
	virtual void transformQReset(AbstractQuantumReset*);
	std::string double_to_string(const double d, const int precision = 17);

	QProg m_src_prog;
    std::map<int, std::string>  m_gatetype; /**< Quantum gatetype map   */
    std::vector<std::string> m_qasm; /**< QASM instructin vector   */
    QuantumMachine * m_quantum_machine;
};

/**
* @brief  Convert Quantum program  to QASM instruction set
* @ingroup Utilities
* @param[in]  QProg&   Quantum Program
* @param[in]  QuantumMachine*  quantum machine pointer
* @param[in] IBMQBackends	ibmBackend = IBMQ_QASM_SIMULATOR
* @return     std::string    QASM instruction set
*/
std::string convert_qprog_to_qasm(QProg &prog, QuantumMachine* qm);

/**
* @brief  write prog to qasm file
* @ingroup Utilities
* @param[in] QProg&   Quantum Program
* @param[in] QuantumMachine*  quantum machine pointer
* @param[in] const std::string	qasm file name
* @return
*/
void write_to_qasm_file(QProg prog, QuantumMachine * qvm, const std::string file_name);
	
QPANDA_END
#endif
