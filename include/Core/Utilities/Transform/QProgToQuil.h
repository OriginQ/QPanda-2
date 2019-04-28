/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgToQuil.h
Author: Wangjing
Updated in 2019/04/09 14:48

Classes for QProgToQuil.
*/
/*! \file QProgToQuil.h */
#ifndef  _QPROG_TO_QUIL_
#define  _QPROG_TO_QUIL_

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Transform/QProgTransform.h"

#include <map>
#include <string>

QPANDA_BEGIN
/**
* @namespace QPanda
* @namespace QGATE_SPACE
*/
/**
* @class QProgToQuil
* @ingroup Utilities
* @brief QuantumProg Transform To Quil instruction sets.
*/
class QProgToQuil : public QProgTransform
{
public:
    QProgToQuil(QuantumMachine * quantum_machine);
    ~QProgToQuil();

    /**
    * @brief  transform quantum program
    * @param[in]  QProg&  quantum program
    * @return     void  
    * @exception    qprog_syntax_error   quantum program syntax error
    */
    virtual void transform(QProg & prog);

    /**
    * @brief  get Quil insturction set
    * @return     std::string
    * @exception
    * @note
    */
    virtual std::string getInsturctions();
protected:
    virtual void transformQProg(AbstractQuantumProgram *);
    virtual void transformQGate(AbstractQGateNode*);
    virtual void transformQCircuit(AbstractQuantumCircuit*);
    virtual void transformQMeasure(AbstractQuantumMeasure*);
    virtual void transformQNode(QNode*);
    virtual void transformQControlFlow(AbstractControlFlowNode *);

    void dealWithQuilGate(AbstractQGateNode*);
    QCircuit transformQPandaBaseGateToQuilBaseGate(AbstractQGateNode*);
private:
    std::map<int, std::string> m_gate_type_map;
    std::vector<std::string>  m_instructs;
    QuantumMachine * m_quantum_machine;
};

/**
* @brief  Quantum program transform to quil instruction set interface
* @ingroup Utilities
* @param[in]  QProg&   quantum program
* @return     std::string   instruction set
* @see
      @code
          init();
          QProg prog;
          auto qvec = qAllocMany(4);
          auto cvec = cAllocMany(4);

          prog << X(qvec[0])
          << Y(qvec[1])
          << H(qvec[0])
          << RX(qvec[0], 3.14)
          << Measure(qvec[1], cvec[0])
          ;
          load(prog);

          auto quil = qProgToQuil(prog);
          std::cout << quil << std::endl;
          finalize();
      @endcode
* @exception    qprog_syntax_error   quantum program syntax error
* @note
*/
std::string transformQProgToQuil(QProg&, QuantumMachine * quantum_machine);
QPANDA_END
#endif // ! _QPROG_TO_QUIL_
