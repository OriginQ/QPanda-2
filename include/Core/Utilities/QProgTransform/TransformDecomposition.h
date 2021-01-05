/******************************************************************
Filename     : TransformDecomposition.h
Creator      : Menghan Dou��cheng xue
Create time  : 2018-07-04
Description  : Quantum program adaptation metadata instruction set
*******************************************************************/
#ifndef _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#define _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#include <functional>
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/GraphDijkstra.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"

QPANDA_BEGIN
using QGATE_SPACE::AbstractAngleParameter;

#define SingleGateMatrixSize 4
#define DoubleGateMatrixSize 16
#define ZeroJudgement 1e-10
#define KMETADATA_GATE_TYPE_COUNT 2

struct axis
{
    double nx;
    double ny;
    double nz;
};
struct QGatesTransform
{
    axis n1;
    axis n2;
};

/**
* @class DecomposeDoubleQGate
* @ingroup Utilities
* @brief Decomposing double gates in qprog
*/
class DecomposeDoubleQGate : public TraversalInterface<>
{
public:
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
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

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

	DecomposeDoubleQGate(std::vector<std::vector<std::string>> valid_qgate_matrix) :m_valid_qgate_matrix(valid_qgate_matrix)
	{}
private:
    void generateMatrixOfTwoLevelSystem(QStat &NewMatrix, QStat &OldMatrix, size_t Row, size_t Column);
    void matrixMultiplicationOfDoubleQGate(QStat &LeftMatrix, QStat &RightMatrix);
	std::vector<std::vector<std::string>> m_valid_qgate_matrix;
};

/**
* @class DecomposeMultipleControlQGate
* @ingroup Utilities
* @brief Decomposing multiple control qgate in qprog
*/
class DecomposeMultipleControlQGate : public TraversalInterface<>
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);

       /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

    /*!
    * @brief  Execution traversal qcircuit
    * @param[in,out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    void execute(std::shared_ptr<AbstractQuantumCircuit>  cur_node, std::shared_ptr<QNode> parent_node);

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
    void QGateExponentArithmetic(AbstractQGateNode *pNode, double Exponent, QStat &QMatrix);
    void transformAxisToMatrix(axis &Axis, double Angle, QStat &QMatrix);
    QCircuit decomposeTwoControlSingleQGate(AbstractQGateNode* pNode);
    QCircuit decomposeToffoliQGate(Qubit* TargetQubit, std::vector<Qubit*> ControlQubits);
    QCircuit firstStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, Qubit *AncillaQubit);
    QCircuit secondStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, std::vector<Qubit*> AncillaQubitVector);
    QCircuit tempStepOfMultipleControlQGateDecomposition(std::vector<Qubit*> ControlQubits, std::vector<Qubit*> AncillaQubits);
	QCircuit decompose_multiple_control_qgate(AbstractQGateNode* cur_node);
};

/**
* @class DecomposeControlUnitarySingleQGate
* @ingroup Utilities
* @brief Decomposing control unitary single qgate in qprog
*/
class DecomposeControlUnitarySingleQGate : public TraversalInterface<>
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);


       /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

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

	DecomposeControlUnitarySingleQGate(std::vector<std::vector<std::string>> valid_qgate_matrix)
		:m_valid_qgate_matrix(valid_qgate_matrix) {}
private:
	std::vector<std::vector<std::string>> m_valid_qgate_matrix;
};

/**
* @class DecomposeDoubleQGate
* @ingroup Utilities
* @brief Decomposing control unitary single qgate in qprog
*/
class DecomposeControlSingleQGateIntoMetadataDoubleQGate : public TraversalInterface<>
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);


       /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

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


    DecomposeControlSingleQGateIntoMetadataDoubleQGate(QuantumMachine * quantum_machine,
        std::vector<std::vector<std::string>> valid_qgate_matrix)
    {
        m_quantum_machine = quantum_machine;
        m_valid_qgate_matrix = valid_qgate_matrix;
    }
private:
    DecomposeControlSingleQGateIntoMetadataDoubleQGate();
    DecomposeControlSingleQGateIntoMetadataDoubleQGate(
        const DecomposeControlSingleQGateIntoMetadataDoubleQGate &old
    );
    DecomposeControlSingleQGateIntoMetadataDoubleQGate &operator = (
        const DecomposeControlSingleQGateIntoMetadataDoubleQGate &old
        ) = delete;
    QCircuit swapQGate(std::vector<int> shortest_way, std::string metadata_qgate);
    std::vector<std::vector<std::string>> m_valid_qgate_matrix;
    QuantumMachine * m_quantum_machine;
};

/**
* @class DecomposeUnitarySingleQGateIntoMetadataSingleQGate
* @ingroup Utilities
* @brief Decomposing unitary single qgate into metadata single qgate in qprog
*/
class DecomposeUnitarySingleQGateIntoMetadataSingleQGate : public TraversalInterface<>
{
public:
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate(std::vector<std::vector<std::string>> qgate_matrix,
        std::vector<std::vector<std::string>> &valid_qgate_matrix);

    /*!
    * @brief  Execution traversal qgatenode
    * @param[in,out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);


       /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

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
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate();
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate(
        const DecomposeUnitarySingleQGateIntoMetadataSingleQGate &old
    );
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate &operator = (
        const DecomposeUnitarySingleQGateIntoMetadataSingleQGate &old
        )=delete;
    std::vector<std::vector<std::string>> m_qgate_matrix;
    std::vector<std::vector<std::string>> m_valid_qgate_matrix;
    QGatesTransform base;
    void getDecompositionAngle(QStat &Qmatrix, std::vector<double> &vdAngle);
    void rotateAxis(QStat &QMatrix, axis &OriginAxis, axis &NewAxis);
};

/**
* @class DeleteUnitQnode
* @ingroup Utilities
* @brief Decomposing unit qnode in qprog
*/
 class DeleteUnitQNode : public TraversalInterface<>
 {
 public:
     /*!
     * @brief  Execution traversal qgatenode
     * @param[in,out]  AbstractQGateNode*  quantum gate
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     */
     void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node);

     
       /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

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
    * @param[in]      AbstractQGateNode*  quantum gate
    * @return         void
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
 };

 /**
 * @class CancelControlQubitVector
 * @ingroup Utilities
 * @brief Cancel control qubit vector in qprog
 */
 class CancelControlQubitVector : public TraversalInterface<>
 {
 public:
     /*!
     * @brief  Execution traversal qgatenode
     * @param[in,out]  AbstractQGateNode*  quantum gate
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     */
     void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
     {}

     /*!
     * @brief  Execution traversal qcircuit
     * @param[in,out]  AbstractQuantumCircuit*  quantum circuit
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     */
     inline void execute(std::shared_ptr<AbstractQuantumCircuit>  cur_node, std::shared_ptr<QNode> parent_node)
     {
         if (nullptr == cur_node)
         {
             QCERR("node is nullptr");
             throw std::invalid_argument("node is nullptr");
         }

         cur_node->clearControl();
         Traversal::traversal(cur_node, false, *this);
     }

       /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

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


 };

 /**
 * @class MergeSingleGate
 * @ingroup Utilities
 * @brief Merge single gate in qprog
 */
 class MergeSingleGate : public TraversalInterface<>
 {
 public:

      /*!
     * @brief  Execution traversal qgatenode
     * @param[in,out]  AbstractQGateNode*  quantum gate
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     */
     void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
     {}

            /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) 
    {}

	/*!
	* @brief  Execution traversal reset node
	* @param[in,out]  AbstractQuantumReset*  reset node
	* @param[in]  AbstractQGateNode*  quantum gate
	* @return     void
	*/
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node)
	{}

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
     void execute(std::shared_ptr<AbstractQuantumCircuit>  cur_node, std::shared_ptr<QNode> parent_node);

     /*!
     * @brief  Execution traversal qprog
     * @param[in,out]  AbstractQuantumProgram*  quantum prog
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     */
     void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node);

    /*!
    * @brief  Execution traversal qprog
    * @param[in,out]  AbstractClassicalProg*  quantum prog
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
        std::shared_ptr<QNode> parent_node)
        {}
 };

 /**
 * @class TransformDecomposition
 * @ingroup Utilities
 * @brief Transform and decompose qprog
 */
class TransformDecomposition
{
public:
    TransformDecomposition(std::vector<std::vector<std::string>> &ValidQGateMatrix,
        std::vector<std::vector<std::string>> &QGateMatrix,
        QuantumMachine * quantum_machine);
    ~TransformDecomposition();

    void TraversalOptimizationMerge(QProg & prog);

	/**
	* @brief merge continue single gate to u3 gate
	* @ingroup Utilities
	* @param[in,out]  QProg& the source prog
	* @return
	* @note
	*/
	void merge_continue_single_gate_to_u3(QProg& prog);

	void decompose_double_qgate(QProg & prog, bool b_decompose_multiple_gate = true);
	void meta_gate_transform(QProg& prog);

private:
    TransformDecomposition();
    TransformDecomposition(
        const TransformDecomposition &old
    );
    TransformDecomposition &operator = (
        const TransformDecomposition &old
        );

    DecomposeDoubleQGate m_decompose_double_gate;
    DecomposeMultipleControlQGate m_decompose_multiple_control_qgate;
    DecomposeControlUnitarySingleQGate m_decompose_control_unitary_single_qgate;
    DecomposeControlSingleQGateIntoMetadataDoubleQGate
        m_control_single_qgate_to_metadata_double_qgate;
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate
        m_unitary_single_qgate_to_metadata_single_qgate;
    DeleteUnitQNode m_delete_unit_qnode;
    CancelControlQubitVector m_cancel_control_qubit_vector;
    MergeSingleGate m_merge_single_gate;

	QuantumMachine * m_quantum_machine;
	std::vector<std::vector<std::string>>& m_valid_qgate_matrix;
};

/**
* @brief Decompose multiple control QGate
* @ingroup Utilities
* @param[in]  QProg&   Quantum Program
* @param[in]  QuantumMachine*  quantum machine pointer
* @param[in] const std::string& It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
* @return
*/
void decompose_multiple_control_qgate(QProg& prog, QuantumMachine *quantum_machine, const std::string& config_data = CONFIG_PATH);
void decompose_multiple_control_qgate(QCircuit& cir, QuantumMachine *quantum_machine, const std::string& config_data = CONFIG_PATH);

/**
* @brief Basic quantum-gate conversion
* @ingroup Utilities
* @param[in]  QProg&   Quantum Program
* @param[in]  QuantumMachine*  quantum machine pointer
* @param[in] const std::string& It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
* @return
* @note Quantum circuits or programs cannot contain multiple-control gates
*/
void transform_to_base_qgate(QProg& prog, QuantumMachine *quantum_machine, const std::string& config_data = CONFIG_PATH);
void transform_to_base_qgate(QCircuit& cir, QuantumMachine *quantum_machine, const std::string& config_data = CONFIG_PATH);

QPANDA_END
#endif // 

