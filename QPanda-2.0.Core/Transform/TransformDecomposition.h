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
#include "QPanda/QStatMatrix.h"
#include "QPanda/GraphDijkstra.h"

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
typedef function<void(AbstractQGateNode *,
    QNode *,
    TransformDecomposition *)> TraversalDecompositionFunction;

/******************************************************************
Name£ºTransformDecomposition
Description£ºQuantum program adaptation metadata instruction set
******************************************************************/
class TransformDecomposition
{
public:
    TransformDecomposition(vector<vector<string>> &ValidQGateMatrix,
        vector<vector<string>> &QGateMatrix,
        vector<vector<int> > &vAdjacentMatrix);
    ~TransformDecomposition();

    void TraversalOptimizationMerge(QNode * node);
    QCircuit decomposeToffoliQGate(Qubit* TargetQubit, vector<Qubit*> ControlQubits);
    QCircuit decomposeTwoControlSingleQGate(AbstractQGateNode* pNode);
private:
    vector<vector<string>> m_sValidQGateMatrix;
    vector<vector<string>> m_sQGateMatrix;
    vector<vector<int> > m_iAdjacentMatrix;
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

    friend QCircuit swapQGate(vector<int> ShortestWay, string MetadataQGate);
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
    void getDecompositionAngle(QStat &Qmatrix, vector<double> &vdAngle);
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
    QCircuit secondStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, vector<Qubit*> AncillaQubitVector);
    QCircuit tempStepOfMultipleControlQGateDecomposition(vector<Qubit*> ControlQubits, vector<Qubit*> AncillaQubits);
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
        stringstream ssErrMsg;
        getssErrMsg(ssErrMsg, "Unknown error");
        throw QPandaException(ssErrMsg.str(), false);
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
                vector<Qubit *> CurQubitVector;
                pCurGateNode->getQuBitVector(CurQubitVector);

                vector<Qubit *> NextQubitVector;
                pNextGateNode->getQuBitVector(NextQubitVector);

                auto pCurPhyQubit = CurQubitVector[0]->getPhysicalQubitPtr();
                auto pNextPhyQubit = NextQubitVector[0]->getPhysicalQubitPtr();

                if ((nullptr == pCurPhyQubit) || (nullptr == pNextPhyQubit))
                {
                    stringstream ssErrMsg;
                    getssErrMsg(ssErrMsg, "Unknown error");
                    throw QPandaException(ssErrMsg.str(), false);
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
                        stringstream ssErrMsg;
                        getssErrMsg(ssErrMsg, "Unknown error");
                        throw QPandaException(ssErrMsg.str(), false);
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

#endif // !_TRAVERSAL_DECOMPOSITION_ALGORITHM_H

