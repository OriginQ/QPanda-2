/*! \file QGateCounter.h */
#ifndef _QGATECOUNTER_H
#define _QGATECOUNTER_H
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/Tools/Traversal.h"

QPANDA_BEGIN


/**
* @class QGateCounter
* @ingroup Utilities
* @brief Count quantum gate num in quantum program, quantum circuit, quantum while, quantum if
*/
class QGateCounter : public TraversalInterface<>
{
public:
    QGateCounter();
    ~QGateCounter();

    template <typename _Ty>
    void traversal(_Ty &node)
    {
        execute(node.getImplementationPtr(), nullptr);
    }
    size_t count();
    const std::map<GateType, size_t> getGateMap();
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);
    
    /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) {}

	/*!
   * @brief  Execution traversal reset node
   * @param[in,out]  AbstractQuantumReset*  reset node
   * @param[in]  AbstractQGateNode*  quantum gate
   * @return     void
   */
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node) {}

    /*!
    * @brief  Execution traversal control flow node
    * @param[in,out]  AbstractControlFlowNode*  control flow node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node) 
    {
        Traversal::traversal(cur_node,*this);
    }


    /*!
    * @brief  Execution traversal qcircuit
    * @param[in,out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
    {
        Traversal::traversal(cur_node,false,*this);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractQuantumProgram*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
    {
        Traversal::traversal(cur_node,*this);
    }
    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractClassicalProg*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
        std::shared_ptr<QNode> parent_node)
        {}

private:
    size_t m_count;
    std::map<GateType, size_t> m_qgate_num_map;
};

/*will delete*/
template <typename _Ty>
size_t getQGateNumber(_Ty &node)
{
    QGateCounter counter;
    counter.traversal(node);
    return counter.count();
}

/* new interface */

/**
* @brief  Count quantum gate num under quantum program, quantum circuit, quantum while, quantum if
* @ingroup Utilities
* @param[in]  _Ty& quantum program, quantum circuit, quantum while or quantum if
* @param[in]  const GateType gtype
* @return     size_t  Quantum gate num
*/

template <typename _Ty>
size_t count_qgate_num(_Ty& node, const GateType gtype = GATE_UNDEFINED) {

    QGateCounter counter;
    counter.traversal(node);
    if (GATE_UNDEFINED == gtype || GATE_NOP == gtype) {
        return counter.count();
    }
    else {
        auto gatemap = counter.getGateMap();
        if (gatemap.find(gtype) != gatemap.end()) {
            return gatemap.at(gtype);
        }
    }
    return 0;
}

template <typename _Ty>
size_t getQGateNum(_Ty &node)
{
	QGateCounter counter;
	counter.traversal(node);
	return counter.count();
}

QPANDA_END
#endif // _QGATECOUNTER_H



