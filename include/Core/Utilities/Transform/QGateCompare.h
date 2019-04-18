/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGateCompare.h
Author: Wangjing
Updated in 2019/04/09 15:05

Classes for QGateCompare.

*/
/*! \file QGateCompare.h */
#ifndef  QGATE_COMPARE_H_
#define  QGATE_COMPARE_H_

#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include <map>
#include "Core/QuantumCircuit/QGlobalVariable.h"
QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @defgroup Utilities
* @brief QPanda2  base  Utilities  classes and  interface
*/

/**
* @class QGateCompare  
* @ingroup Utilities
* @brief Qunatum Gate Compare
*/
class QGateCompare {
public:
    QGateCompare();
    virtual ~QGateCompare();
    
    /**
    * @brief  Count 1uantumprogram unsupported gatenum
    * @param[in]  AbstractQuantumProgram*    Abstract Quantum program pointer
    * @param[in]  const std::vector<std::vector<std::string>>&    Instructions  
    * @return     size_t     Unsupported QGate num
    * @exception    invalid_argument    Quantum program pointer is a nullptr
    * @note
    */
    static size_t countQGateNotSupport(AbstractQuantumProgram *,
                                       const std::vector<std::vector<std::string>> &);

    /**
    * @brief  Count 1uantumprogram unsupported gatenum
    * @param[in]  AbstractQGateNode*   Abstract QGate Node pointer
    * @param[in]  const std::vector<std::vector<std::string>>&    Instructions
    * @return     size_t   Unsupported QGate num
    * @exception    invalid_argument    Quantum gate pointer is a nullptr
    * @note
    */

    static size_t countQGateNotSupport(AbstractQGateNode *,
                                       const std::vector<std::vector<std::string>> &);

    /**
    * @brief  Count 1uantumprogram unsupported gatenum
    * @param[in]  AbstractControlFlowNode*    Abstract Control Flow Node pointer
    * @param[in]  const std::vector<std::vector<std::string>>&    Instructions
    * @return     size_t    Unsupported QGate num
    * @exception    invalid_argument    Quantum controlflow pointer is a nullptr
    * @note
    */
    static size_t countQGateNotSupport(AbstractControlFlowNode *,
                                       const std::vector<std::vector<std::string>> &);
    /**
    * @brief  Count 1uantumprogram unsupported gatenum
    * @param[in]  AbstractQuantumCircuit*    Abstract Quantum Circuit Node pointer
    * @param[in]  const std::vector<std::vector<std::string>>&    Instructions
    * @return     size_t    Unsupported QGate num
    * @exception    invalid_argument   Quantum circuit pointer is a nullptr
    * @note
    */
    static size_t countQGateNotSupport(AbstractQuantumCircuit *,
                                       const std::vector<std::vector<std::string>> &);
protected:

    static size_t countQGateNotSupport(QNode *, 
                                       const std::vector<std::vector<std::string>> &);

    static bool isItemExist(const std::string &,
                            const std::vector<std::vector<std::string>> &);
private:
};
QPANDA_END


#endif
