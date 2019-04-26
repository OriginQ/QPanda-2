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
#include "Core/Utilities/Transform/QGateCounter.h"

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
class QGateCompare : public TraversalInterface
{
public:
    QGateCompare(const std::vector<std::vector<std::string>> &);

    /**
    * @brief  traversal quantum program, quantum circuit, quantum while or quantum if
    * @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
    * @return     void
    * @note
    */
    template <typename _Ty>
    void traversal(_Ty &node)
    {
        static_assert(std::is_base_of<QNode, _Ty>::value, "bad node type");
        Traversal::traversalByType(&node, &node, this);
    }

    /**
    * @brief  get unsupported gate numner
    * @return     size_t     Unsupported QGate number
    * @note
    */
    size_t count();
private:
    virtual void execute(AbstractQGateNode * cur_node, QNode * parent_node);
    size_t m_count;
    std::vector<std::vector<std::string>> m_gates;
};

/**
* @brief  Count quantum program unsupported gate numner
* @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
* @param[in]  const std::vector<std::vector<std::string>>&    support gates
* @return     size_t     Unsupported QGate number
* @note
*/
template <typename _Ty>
size_t getUnSupportQGateNumber(_Ty &node, const std::vector<std::vector<std::string>> &gates)
{
    static_assert(std::is_base_of<QNode, _Ty>::value, "bad node type");
    QGateCompare compare(gates);
    compare.traversal(node);
    return compare.count();
}

QPANDA_END


#endif
