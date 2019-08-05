/******************************************************************
Filename     : TransformDecomposition.h
Creator      : Menghan Dou¡¢cheng xue
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
#include "Core/Utilities/QStatMatrix.h"
#include "Core/Utilities/GraphDijkstra.h"
#include "Core/Utilities/Traversal.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN
using QGATE_SPACE::angleParameter;

#define SingleGateMatrixSize 4
#define DoubleGateMatrixSize 16
#define ZeroJudgement 1e-10

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
class DecomposeDoubleQGate : public TraversalInterface
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in|out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    void execute(AbstractQGateNode * pNode, QNode * pParentNode);
private:
    void generateMatrixOfTwoLevelSystem(QStat &NewMatrix, QStat &OldMatrix, size_t Row, size_t Column);
    void matrixMultiplicationOfDoubleQGate(QStat &LeftMatrix, QStat &RightMatrix);
};

/**
* @class DecomposeMultipleControlQGate
* @ingroup Utilities
* @brief Decomposing multiple control qgate in qprog
*/
class DecomposeMultipleControlQGate : public TraversalInterface
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in|out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    void execute(AbstractQGateNode * node, QNode * parent_node);

    /*!
    * @brief  Execution traversal qcircuit
    * @param[in|out]  AbstractQuantumCircuit*  quantum circuit
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    void execute(AbstractQuantumCircuit * node, QNode * parent_node);
private:
    void QGateExponentArithmetic(AbstractQGateNode *pNode, double Exponent, QStat &QMatrix);
    void transformAxisToMatrix(axis &Axis, double Angle, QStat &QMatrix);
    QCircuit decomposeTwoControlSingleQGate(AbstractQGateNode* pNode);
    QCircuit decomposeToffoliQGate(Qubit* TargetQubit, std::vector<Qubit*> ControlQubits);
    QCircuit firstStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, Qubit *AncillaQubit);
    QCircuit secondStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, std::vector<Qubit*> AncillaQubitVector);
    QCircuit tempStepOfMultipleControlQGateDecomposition(std::vector<Qubit*> ControlQubits, std::vector<Qubit*> AncillaQubits);
};

/**
* @class DecomposeControlUnitarySingleQGate
* @ingroup Utilities
* @brief Decomposing control unitary single qgate in qprog
*/
class DecomposeControlUnitarySingleQGate : public TraversalInterface
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in|out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    void execute(AbstractQGateNode * node, QNode * parent_node);
};

/**
* @class DecomposeDoubleQGate
* @ingroup Utilities
* @brief Decomposing control unitary single qgate in qprog
*/
class DecomposeControlSingleQGateIntoMetadataDoubleQGate : public TraversalInterface
{
public:
    /*!
    * @brief  Execution traversal qgatenode
    * @param[in|out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    void execute(AbstractQGateNode * node, QNode * parent_node);

    DecomposeControlSingleQGateIntoMetadataDoubleQGate(QuantumMachine * quantum_machine,
        std::vector<std::vector<std::string>> valid_qgate_matrix,
        std::vector<std::vector<int> > adjacent_matrix)
    {
        m_quantum_machine = quantum_machine;
        m_valid_qgate_matrix = valid_qgate_matrix;
        m_adjacent_matrix = adjacent_matrix;
    }
private:
    DecomposeControlSingleQGateIntoMetadataDoubleQGate();
    DecomposeControlSingleQGateIntoMetadataDoubleQGate(
        const DecomposeControlSingleQGateIntoMetadataDoubleQGate &old
    );
    DecomposeControlSingleQGateIntoMetadataDoubleQGate &operator = (
        const DecomposeControlSingleQGateIntoMetadataDoubleQGate &old
        );
    QCircuit swapQGate(std::vector<int> shortest_way, std::string metadata_qgate);
    std::vector<std::vector<std::string>> m_valid_qgate_matrix;
    std::vector<std::vector<int> > m_adjacent_matrix;
    QuantumMachine * m_quantum_machine;
};

/**
* @class DecomposeUnitarySingleQGateIntoMetadataSingleQGate
* @ingroup Utilities
* @brief Decomposing unitary single qgate into metadata single qgate in qprog
*/
class DecomposeUnitarySingleQGateIntoMetadataSingleQGate : public TraversalInterface
{
public:
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate(std::vector<std::vector<std::string>> qgate_matrix,
        std::vector<std::vector<std::string>> &valid_qgate_matrix);

    /*!
    * @brief  Execution traversal qgatenode
    * @param[in|out]  AbstractQGateNode*  quantum gate
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    * @exception invalid_argument
    * @note
    */
    void execute(AbstractQGateNode * node, QNode * parent_node);
private:
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate();
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate(
        const DecomposeUnitarySingleQGateIntoMetadataSingleQGate &old
    );
    DecomposeUnitarySingleQGateIntoMetadataSingleQGate &operator = (
        const DecomposeUnitarySingleQGateIntoMetadataSingleQGate &old
        );
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
 class DeleteUnitQNode : public TraversalInterface
 {
 public:
     /*!
     * @brief  Execution traversal qgatenode
     * @param[in|out]  AbstractQGateNode*  quantum gate
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     * @exception invalid_argument
     * @note
     */
     void execute(AbstractQGateNode * node, QNode * parent_node);
 };

 /**
 * @class CancelControlQubitVector
 * @ingroup Utilities
 * @brief Cancel control qubit vector in qprog
 */
 class CancelControlQubitVector :public TraversalInterface
 {
 public:
     /*!
     * @brief  Execution traversal qcircuit
     * @param[in|out]  AbstractQuantumCircuit*  quantum circuit
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     * @exception invalid_argument
     * @note
     */
     inline void execute(AbstractQuantumCircuit * node, QNode * parent_node)
     {
         if (nullptr == node)
         {
             QCERR("node is nullptr");
             throw std::invalid_argument("node is nullptr");
         }

         node->clearControl();
         Traversal::traversal(node, this, false);
     }
 };

 /**
 * @class MergeSingleGate
 * @ingroup Utilities
 * @brief Merge single gate in qprog
 */
 class MergeSingleGate :public TraversalInterface
 {
 public:
     /*!
     * @brief  Execution traversal qcircuit
     * @param[in|out]  AbstractQuantumCircuit*  quantum circuit
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     * @exception invalid_argument
     * @note
     */
     void execute(AbstractQuantumCircuit * node, QNode * parent_node);

     /*!
     * @brief  Execution traversal qprog
     * @param[in|out]  AbstractQuantumCircuit*  quantum prog
     * @param[in]  AbstractQGateNode*  quantum gate
     * @return     void
     * @exception invalid_argument
     * @note
     */
     void execute(AbstractQuantumProgram * node, QNode * parent_node);
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
        std::vector<std::vector<int> > &vAdjacentMatrix,
        QuantumMachine * quantum_machine);
    ~TransformDecomposition();

    void TraversalOptimizationMerge(QProg & prog);
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
        m_unitary_single_qgate_to_metadata_double_qgate;
    DeleteUnitQNode m_delete_unit_qnode;
    CancelControlQubitVector m_cancel_control_qubit_vector;
    MergeSingleGate m_merge_single_gate;
};

QPANDA_END
#endif // 

