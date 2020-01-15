#ifndef _QPROG_TO_QCIRCUIT_H
#define _QPROG_TO_QCIRCUIT_H

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

QPANDA_BEGIN

/**
* @brief cast QProg to QCircuit
* @ingroup Utilities
*/
class QProgToQCircuit : public TraversalInterface<QCircuit &>
{
public:
    QProgToQCircuit() {};
    ~QProgToQCircuit() {};


    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QCircuit & circuit)
    {
        circuit.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
    }

    /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuit & circuit)
    {
        QCERR("node error");
        throw run_fail("node error");
    }

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuit & circuit)
	{
		QCERR("node type error, can NOT be reset node here.");
		throw run_fail("node type error, can NOT be reset node here.");
	}

    /*!
    * @brief  Execution traversal control flow node
    * @param[in,out]  AbstractControlFlowNode*  control flow node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuit & circuit)
    {
        QCERR("node error");
        throw run_fail("node error");
    }


    /*!
    * @brief  Execution traversal qcircuit
    * @param[in,out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuit & circuit)
    {
        circuit.pushBackNode(std::dynamic_pointer_cast<QNode>(cur_node));
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractQuantumProgram*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuit & circuit)
    {
        Traversal::traversal(cur_node, *this, circuit);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractClassicalProg*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
        std::shared_ptr<QNode> parent_node, QCircuit & circuit)
    {
        QCERR("node error");
        throw run_fail("node error");
    }

};

/**
* @brief Cast Quantum Program To Quantum Circuit
* @ingroup Utilities
* @param[in]  QProg	  quantum program
* @param[out]  QCircuit&	  quantum progCircuitram
* @return     bool
*/
static bool cast_qprog_qcircuit(QProg prog, QCircuit& circuit)
{
    QProgToQCircuit traversal_class;
    bool result = true;
    try
    {
        traversal_class.execute(prog.getImplementationPtr(), nullptr, circuit);
    }
    catch (const std::exception&)
    {
        result = false;
    }

    return result;
}

QPANDA_END
#endif