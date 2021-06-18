/*! \file QNodeDeepCopy.h */
#ifndef  _QNODEDEEPCOPY_H_
#define  _QNODEDEEPCOPY_H_

#include "Core/Utilities/Tools/Traversal.h"
#include <memory>
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"

QPANDA_BEGIN


/**
* @class QNodeDeepCopy
* @ingroup QuantumCircuit
* @brief Deep copy interface for classess based on QNode
* @note
*/
class QNodeDeepCopy : public TraversalInterface<>
{
public:
    QNodeDeepCopy() {};
    ~QNodeDeepCopy() {};

   
	/**
	* @brief Execute QNode Node
	* @param[in]  QNode*
	* @return     std::shared_ptr<QPanda::QNode> new Node
	*/
	std::shared_ptr<QNode> executeQNode(std::shared_ptr<QNode> node);

    /**
    * @brief  Execute Quantum Gate Node
    * @param[in]  AbstractQGateNode* Quantum Gate Node
    * @return     std::shared_ptr<QPanda::QNode>  new QNode
    */
    QGate copy_node(std::shared_ptr<AbstractQGateNode>);

    /**
    * @brief  Execute Quantum QProg Node
    * @param[in]  AbstractQuantumProgram* Quantum QProg Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    QProg copy_node(std::shared_ptr<AbstractQuantumProgram>);

    /**
    * @brief  Execute Quantum Measure Node
    * @param[in]  AbstractQuantumMeasure* Quantum Measure Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    QMeasure copy_node( std::shared_ptr<AbstractQuantumMeasure>);

	/**
	* @brief  Execute Quantum Reset Node
	* @param[in]  AbstractQuantumReset* Quantum Reset Node
	* @return     std::shared_ptr<QPanda::QNode> new Node
	*/
	QReset copy_node(std::shared_ptr<AbstractQuantumReset>);

    /**
    * @brief  Execute Quantum Circuit Node
    * @param[in]  AbstractQuantumCircuit* Quantum Circuit Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    QCircuit copy_node(std::shared_ptr<AbstractQuantumCircuit>);


    /**
    * @brief  Execute ControlFlow Node
    * @param[in]  AbstractControlFlowNode* ControlFlow Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    std::shared_ptr<AbstractControlFlowNode> copy_node(std::shared_ptr<AbstractControlFlowNode>);
    /**
    * @brief  Execute ClassicalProg Node
    * @param[in]  AbstractClassicalProg* ClassicalProg Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    ClassicalProg copy_node(std::shared_ptr<AbstractClassicalProg>);


    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>);
	void execute(std::shared_ptr<AbstractQuantumReset>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>);
private:
    void insert(std::shared_ptr<QNode>, std::shared_ptr<QNode>);
};


/**
* @brief  deep copy interface for classess base on QNode
* @ingroup QuantumCircuit
* @param[in]  _Ty & node
* @return     _Ty
*/
template <typename _Ty>
_Ty deepCopy(_Ty &node)
{
	QNodeDeepCopy reproduction;
    auto  pNode = reproduction.copy_node(node.getImplementationPtr());
    return pNode;
}

std::shared_ptr<QNode> deepCopyQNode(std::shared_ptr<QNode> src_node);

QPANDA_END

#endif // QNODEDEEPCOPY_H