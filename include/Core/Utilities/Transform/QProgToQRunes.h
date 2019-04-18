/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgToQRunes.h
Author: Yulei
Updated in 2019/04/09 14:37

Classes for QProgToQRunes.

*/

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
   ~QProgToQRunes();

    /**
    * @brief  Transform quantum program
    * @param[in]  QProg&    quantum program
    * @return     void
    * @exception  invalid_argument
    * @code
    * @endcode
    * @note
    */
    virtual void transform(QProg &prog);

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
* @param[in]  QProg&   Quantum Program
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
std::string transformQProgToQRunes(QProg &prog, QuantumMachine * quantum_machine);
QPANDA_END
#endif
