/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file QProgCheck.h */

#ifndef _QPROG_CHECK_H_
#define _QPROG_CHECK_H_


#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QReset.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/Tools/Traversal.h"
#include <map>
#include <type_traits>
#include <memory>



QPANDA_BEGIN


/**
* @brief Qunatum QProgCheck
* @ingroup QuantumMachine
*/
class QProgCheck
{
public:

    QProgCheck();

    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @param[in]  TraversalConfig  traversal config
    * @param[in]  QPUImpl*  virtual quantum processor
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQGateNode> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param);

    /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @param[in]  TraversalConfig   traversal config
    * @param[in]  QPUImpl*  virtual quantum processor
    * @return     void
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param);

    /*!
    * @brief  Execution traversal reset node
    * @param[in,out]  AbstractQuantumReset*  reset node
    * @param[in]  QNode*  parent node
    * @param[in]  TraversalConfig   traversal config
    * @param[in]  QPUImpl*  virtual quantum processor
    * @return     void
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param);

    /*!
    * @brief  Execution traversal control flow node
    * @param[in,out]  AbstractControlFlowNode*  control flow node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @param[in]  TraversalConfig  traversal config
    * @param[in]  QPUImpl*  virtual quantum processor
    * @return     void
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param);


    /*!
    * @brief  Execution traversal qcircuit
    * @param[in,out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @param[in]  TraversalConfig  traversal config
    * @param[in]  QPUImpl*  virtual quantum processor
    * @return     void
    * @note
    */
    virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param);
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractQuantumProgram*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @param[in]  TraversalConfig  traversal config
    * @param[in]  QPUImpl*  virtual quantum processor
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumProgram> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & paramu);
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractClassicalProg*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @param[in]  TraversalConfig  traversal config
    * @param[in]  QPUImpl*  virtual quantum processor
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractClassicalProg> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param)
    {
        cur_node->get_val();
    }

protected:
    void is_can_optimize_measure(const QVec &controls, const QVec &targets, TraversalConfig &param);
private:

};



QPANDA_END





#endif // _QPROG_CHECK_H_
