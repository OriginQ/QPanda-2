/******************************************************************
Filename     : TransformDecomposition.h
Creator      : Menghan Dou¡¢cheng xue
Create time  : 2018-07-04
Description  : Quantum program adaptation metadata instruction set
*******************************************************************/
#ifndef _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#define _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#include "QuantumCircuit/QProgram.h"
#include "QPanda.h"
#include <functional>
#include "Utilities/QStatMatrix.h"
#include "Utilities/GraphDijkstra.h"
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

class TransformDecomposition;
typedef std::function<void(AbstractQGateNode *,
    QNode *,
    TransformDecomposition *)> TraversalDecompositionFunction;
QCircuit swapQGate(std::vector<int> ShortestWay, std::string MetadataQGate);
void decomposeDoubleQGate(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition *);
void decomposeMultipleControlQGate(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition *);
void decomposeControlUnitarySingleQGate(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition *);
void decomposeControlSingleQGateIntoMetadataDoubleQGate(AbstractQGateNode * pNode,
        QNode * pParentNode, TransformDecomposition *);
void decomposeUnitarySingleQGateIntoMetadataSingleQGate(AbstractQGateNode * pNode,
        QNode * pParentNode, TransformDecomposition *);
 void deleteUnitQnode(AbstractQGateNode * pNode,
        QNode * pParentNode, TransformDecomposition *);

/******************************************************************
Name£ºTransformDecomposition
Description£ºQuantum program adaptation metadata instruction set
******************************************************************/
class TransformDecomposition
{
public:
    TransformDecomposition(std::vector<std::vector<std::string>> &ValidQGateMatrix,
        std::vector<std::vector<std::string>> &QGateMatrix,
        std::vector<std::vector<int> > &vAdjacentMatrix);
    ~TransformDecomposition();

    void TraversalOptimizationMerge(QNode * node);
    QCircuit decomposeToffoliQGate(Qubit* TargetQubit, std::vector<Qubit*> ControlQubits);
    QCircuit decomposeTwoControlSingleQGate(AbstractQGateNode* pNode);
private:
    std::vector<std::vector<std::string>> m_sValidQGateMatrix;
    std::vector<std::vector<std::string>> m_sQGateMatrix;
    std::vector<std::vector<int> > m_iAdjacentMatrix;
    QGatesTransform base;

    TransformDecomposition() {};

    void cancelControlQubitVector(QNode * pNode);

    void Traversal(AbstractControlFlowNode *, TraversalDecompositionFunction, int iType);
    void Traversal(AbstractQuantumCircuit *, TraversalDecompositionFunction, int iType);
    void Traversal(AbstractQuantumProgram *, TraversalDecompositionFunction, int iType);
    void TraversalByType(QNode * pNode, QNode *, TraversalDecompositionFunction, int iType);

    void mergeSingleGate(QNode * pNode);
    void mergeControlFlowSingleGate(AbstractControlFlowNode * pNode, int);
    template<typename T>
    void mergeCircuitandProgSingleGate(T pNode);

    friend QCircuit swapQGate(std::vector<int> ShortestWay, std::string MetadataQGate);
    friend void decomposeDoubleQGate(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition *);
    friend void decomposeMultipleControlQGate(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition *);
    friend void decomposeControlUnitarySingleQGate(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition *);
    friend void decomposeControlSingleQGateIntoMetadataDoubleQGate(AbstractQGateNode * pNode,
        QNode * pParentNode, TransformDecomposition *);
    friend void decomposeUnitarySingleQGateIntoMetadataSingleQGate(AbstractQGateNode * pNode,
        QNode * pParentNode, TransformDecomposition *);
    friend void deleteUnitQnode(AbstractQGateNode * pNode,
        QNode * pParentNode, TransformDecomposition *);
    void checkControlFlowBranch(QNode *pNode);
    void insertQCircuit(AbstractQGateNode *pGateNode, QCircuit &, QNode *pParentNode);
    void getDecompositionAngle(QStat &Qmatrix, std::vector<double> &vdAngle);
    void matrixMultiplicationOfSingleQGate(QStat &LeftMatrix, QStat &RightMatrix);
    void rotateAxis(QStat &QMatrix, axis &OriginAxis, axis &NewAxis);
    void matrixMultiplicationOfDoubleQGate(QStat &LeftMatrix, QStat &RightMatrix);
    void generateMatrixOfTwoLevelSystem(QStat &NewMatrix, QStat &OldMatrix, size_t Row, size_t Column);
    inline double getArgument(qcomplex_t cNumber)
    {
        if (cNumber.imag() >= 0)
        {
            return acos(cNumber.real() / sqrt(cNumber.real()*cNumber.real() + cNumber.imag()*cNumber.imag()));
        }
        else
        {
            return -acos(cNumber.real() / sqrt(cNumber.real()*cNumber.real() + cNumber.imag()*cNumber.imag()));
        }
    }

    double transformMatrixToAxis(QStat &QMatrix, axis &Axis);
    void QGateExponentArithmetic(AbstractQGateNode *pNode, double Exponent, QStat &QMatrix);
    void transformAxisToMatrix(axis &Axis, double Angle, QStat &QMatrix);

    QCircuit firstStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, Qubit *AncillaQubit);
    QCircuit secondStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, std::vector<Qubit*> AncillaQubitVector);
    QCircuit tempStepOfMultipleControlQGateDecomposition(std::vector<Qubit*> ControlQubits, std::vector<Qubit*> AncillaQubits);
};

/******************************************************************
Name        : mergeCircuitandProgSingleGate
Description : Merge quantum  circuit and quantum prog single gate
argin       : pNode    Target Node pointer
argout      : pNode    Target Node pointer
Return      :
******************************************************************/
extern QGate U4(QStat & matrix, Qubit *qubit);
template<typename T>
void TransformDecomposition::mergeCircuitandProgSingleGate(T  pNode)
{
    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }
    auto aiter = pNode->getFirstNodeIter();

    /*
    * Traversal PNode's children node
    */
    for (; aiter != pNode->getEndNodeIter(); ++aiter)
    {
        int iNodeType = (*aiter)->getNodeType();

        /*
        * If it is not a gate type, the algorithm goes deeper
        */
        if (GATE_NODE != iNodeType)
        {
            mergeSingleGate(*aiter);
            continue;
        }

        AbstractQGateNode * pCurGateNode = dynamic_cast<AbstractQGateNode *>(*aiter);

        if (pCurGateNode->getQuBitNum() == 2)
            continue;

        auto nextIter = aiter.getNextIter();

        AbstractQGateNode * pNextGateNode = nullptr;

        /*
        * Loop through the nodes behind the target node
        * and execute the merge algorithm
        */
        while (nextIter != pNode->getEndNodeIter())
        {
            int iNextNodeType = (*nextIter)->getNodeType();

            if (GATE_NODE != iNextNodeType)
                break;

            pNextGateNode = dynamic_cast<AbstractQGateNode *>(*nextIter);

            if (pNextGateNode->getQuBitNum() == 1)
            {
                std::vector<Qubit *> CurQubitVector;
                pCurGateNode->getQuBitVector(CurQubitVector);

                std::vector<Qubit *> NextQubitVector;
                pNextGateNode->getQuBitVector(NextQubitVector);

                auto pCurPhyQubit = CurQubitVector[0]->getPhysicalQubitPtr();
                auto pNextPhyQubit = NextQubitVector[0]->getPhysicalQubitPtr();

                if ((nullptr == pCurPhyQubit) || (nullptr == pNextPhyQubit))
                {
                    QCERR("Unknown internal error");
                    throw std::runtime_error("Unknown internal error");
                }

                /*
                * Determine if it is the same qubit
                */
                if (pCurPhyQubit->getQubitAddr() == pNextPhyQubit->getQubitAddr())
                {
                    auto pCurQGate = pCurGateNode->getQGate();
                    auto pNextQGate = pNextGateNode->getQGate();

                    if ((nullptr == pCurQGate) || (nullptr == pNextQGate))
                    {
                        QCERR("Unknown internal error");
                        throw std::runtime_error("Unknown internal error");
                    }

                    /*
                    * Merge node
                    */
                    QStat CurMatrix, NextMatrix;
                    pCurQGate->getMatrix(CurMatrix);
                    pNextQGate->getMatrix(NextMatrix);
                    QStat newMatrix = CurMatrix * NextMatrix;
                    auto temp = U4(newMatrix, CurQubitVector[0]);
                    auto pCurtItem = aiter.getPCur();
                    pCurtItem->setNode(&temp);
                    pCurGateNode = dynamic_cast<AbstractQGateNode *>(pCurtItem->getNode());
                    nextIter = pNode->deleteQNode(nextIter);
                }
            }
            nextIter = nextIter.getNextIter();
        }
    }
}
QPANDA_END
#endif // !_TRAVERSAL_DECOMPOSITION_ALGORITHM_H

