/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
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
QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @class QProgToQASM
* @ingroup Utilities
* @brief Quantum Prog Transform To QASM instruction sets
*/
class QProgToQASM 
{
public:
    QProgToQASM(QuantumMachine * quantum_machine);
    ~QProgToQASM() {}
    /**
    * @brief  get QASM insturction set
    * @return     std::string  
    * @exception
    * @note
    */
    virtual std::string getInsturctions();

    /*!
    * @brief  Transform Quantum program
    * @param[in]  QProg&  quantum program
    * @return     void  
    * @exception
    * @note
    */
    virtual void transform(QProg &);

private:
    virtual void transformQProg(AbstractQuantumProgram*);
    virtual void transformQGate(AbstractQGateNode*);
    virtual void transformQControlFlow(AbstractControlFlowNode*);
    virtual void transformQCircuit(AbstractQuantumCircuit*);
    virtual void transformQMeasure(AbstractQuantumMeasure*);
    virtual void transformQNode(QNode*);

    void handleDaggerNode(QNode*, int);
    void handleDaggerCir(QNode*);
    void handleIfWhileQNode(AbstractControlFlowNode*, std::string);
    std::map<int, std::string>  m_gatetype; /**< Quantum gatetype map   */
    std::vector<std::string> m_qasm; /**< QASM instructin vector   */
    QuantumMachine * m_quantum_machine;
};

    /**
    * @brief  Quantum program transform to qasm instruction set
    * @ingroup Utilities
    * @param[in]  QProg&   Quantum Program 
    * @return     std::string    QASM instruction set
    * @see
        * @code
                init(QuantumMachine_type::CPU);

                auto qubit = qAllocMany(6);
                auto cbit  = cAllocMany(2);
                auto prog = CreateEmptyQProg();

                prog << CZ(qubit[0], qubit[2]) << H(qubit[1]) << CNOT(qubit[1], qubit[2])
                << RX(qubit[0],pi/2) << Measure(qubit[1],cbit[1]);

                std::cout << transformQProgToQASM(prog) << std::endl;
                finalize();
        * @endcode
    * @exception
    * @note
    */
    std::string transformQProgToQASM(QProg &pQProg, QuantumMachine * quantum_machine);

QPANDA_END
#endif
