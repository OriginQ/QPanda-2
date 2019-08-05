/*! \file QNodeDeepCopy.h */
#ifndef  _QNODEDEEPCOPY_H_
#define  _QNODEDEEPCOPY_H_
#include <memory>
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/Utilities/Traversal.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"

QPANDA_BEGIN
/**
* @namespace QPanda
*/


/**
* @class QNodeDeepCopy
* @brief Deep copy interface for classess based on QNode
* @note
*/
class QNodeDeepCopy :public TraversalInterface
{
public:
    QNodeDeepCopy() {};
    ~QNodeDeepCopy() {};

    /**
    * @brief Execute QNode Node
    * @param[in]  QNode *
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    std::shared_ptr<QNode> executeQNode(QNode *);

    /**
    * @brief  Execute Quantum Gate Node
    * @param[in]  AbstractQGateNode* Quantum Gate Node
    * @return     std::shared_ptr<QPanda::QNode>  new QNode
    */
    std::shared_ptr<QNode> executeQGate(AbstractQGateNode *);

    /**
    * @brief  Execute Quantum QProg Node
    * @param[in]  AbstractQuantumProgram* Quantum QProg Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    std::shared_ptr<QNode> executeQProg(AbstractQuantumProgram *);

    /**
    * @brief  Execute Quantum Measure Node
    * @param[in]  AbstractQuantumMeasure* Quantum Measure Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    std::shared_ptr<QNode> executeQMeasure(AbstractQuantumMeasure *);


    /**
    * @brief  Execute Quantum Circuit Node
    * @param[in]  AbstractQuantumCircuit* Quantum Circuit Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    std::shared_ptr<QNode> executeQCircuit(AbstractQuantumCircuit *);


    /**
    * @brief  Execute ControlFlow Node
    * @param[in]  AbstractControlFlowNode* ControlFlow Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    std::shared_ptr<QNode> executeControlFlow(AbstractControlFlowNode *);
    /**
    * @brief  Execute ClassicalProg Node
    * @param[in]  AbstractClassicalProg* ClassicalProg Node
    * @return     std::shared_ptr<QPanda::QNode> new Node
    */
    std::shared_ptr<QNode> executeClassicalProg(AbstractClassicalProg *);

private:
    void execute(AbstractQGateNode *, QNode *);
    void execute(AbstractClassicalProg *, QNode *);
    void execute(AbstractQuantumCircuit *, QNode *);
    void execute(AbstractQuantumMeasure *, QNode *);
    void execute(AbstractControlFlowNode *, QNode *);
    void execute(AbstractQuantumProgram *, QNode *);

    void insert(std::shared_ptr<QNode>, QNode *);
};


/**
* @brief  deep copy interface for classess base on QNode
* @ingroup Utilities
* @param[in]  _Ty & node
* @return     _Ty
*/
template <typename _Ty>
_Ty deepCopy(_Ty &node)
{
    static_assert(std::is_base_of<QNode, _Ty>::value, "bad node type");
    QNodeDeepCopy reproduction;

    std::shared_ptr<QNode>  pNode = reproduction.executeQNode(node.getImplementationPtr().get());
    auto tar_node = std::dynamic_pointer_cast<_Ty >(pNode);
    return *tar_node;
}

QPANDA_END

#endif // QNODEDEEPCOPY_H