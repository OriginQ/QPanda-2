#ifndef _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#define _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#include "QProgram.h"
class TraversalDecompositionAlgorithm
{
    //typedef function<QCircuit(AbstractQGateNode *)> TransferAlgorithm;
public:
    TraversalDecompositionAlgorithm();
    ~TraversalDecompositionAlgorithm();

    template<typename T>
    static QNode * traversalDecomposition(T pNode);
private:
    template<typename PNODE>
    void doubleGateToCU(AbstractQGateNode *, PNODE);
    template<typename PNODE>
    void multipleControlGateToQCircuit(AbstractQGateNode *, PNODE);
    void CUToControlSingleGate(AbstractQGateNode *);
    template<typename T>
    void Traversal(AbstractControlFlowNode *, T);
    template<typename T>
    void Traversal(AbstractQuantumCircuit *, T);
    template<typename T>
    void Traversal(AbstractQuantumProgram *, T);
    template<typename T>
    void TraversalByType(QNode * pNode, T);


};

template<typename T>
inline QNode * TraversalDecompositionAlgorithm::traversalDecomposition(T pNode)
{
    QNode * pNode = dynamic_cast<QNode *> (pNode);
    if (nullptr == pNode)
        throw param_error_exception("this param is not QNode", false);
    
    int iNodeType = pNode->getNodeType();
    if((GATE_NODE == iNodeType) || (MEASURE_GATE == iNodeType))
        throw param_error_exception("the param cannot be a QGate or Measure", false);
    Traversal(pNode, doubleGateToCU);
    Traversal(pNode, multipleControlGateToQCircuit);
    Traversal(pNode, CUToControlSingleGate);
}

template<typename T>
inline void TraversalDecompositionAlgorithm::Traversal(AbstractControlFlowNode * pControlNode, T function)
{
    if(nullptr == pControlNode)
        throw param_error_exception("param error", false);
    auto pNode = dynamic_cast<QNode *>(pControlNode);
    if (nullptr == pNode)
        throw exception();
    auto iNodeType = pNode->getNodeType();
    if (WHILE_START_NODE == iNodeType)
    {
        int TrueBranchNodeType = pControlNode->getTrueBranch()->getNodeType();
        TraversalByType(TrueBranchNodeType, function);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        TraversalByType(pControlNode->getTrueBranch(), function);
        TraversalByType(pControlNode->getFalseBranch(), function);
    }
}

template<typename T>
inline void TraversalDecompositionAlgorithm::Traversal(AbstractQuantumCircuit * pQCircuit, T function)
{
    if (nullptr == pProg)
        throw param_error_exception("param error", false);

    auto aiter = pProg->getFirstNodeIter();
    if (aiter == pProg->getEndNodeIter())
        return;
    for (; aiter != pProg->getEndNodeIter(); ++aiter)
    {
        int iNodeType = (*aiter)->getNodeType();
        switch (iNodeType)
        {
        case GATE_NODE:
            function(*aiter, pControlNode);
            break;
        case CIRCUIT_NODE:
            Traversal(dynamic_cast<AbstractQuantumCircuit *>(*aiter), function);
            break;
        default:
            throw exception();
        }
    }
} 

template<typename T>
inline void TraversalDecompositionAlgorithm::Traversal(AbstractQuantumProgram * pProg, T function)
{
    if (nullptr == pProg)
        throw param_error_exception("param error", false);
    auto aiter = pProg->getFirstNodeIter();
    if (aiter == pProg->getEndNodeIter())
        return;
    for (; aiter != pProg->getEndNodeIter(); ++aiter)
    {
        TraversalByType((*aiter), function);
    }
}

template<typename T>
inline void TraversalDecompositionAlgorithm::TraversalByType(QNode * pNode, T function)
{
    int iNodeType = pNode->getNodeType();
    if (-1 == iNodeType)
        throw param_error_exception("param error", false);
    switch (iNodeType)
    {
    case GATE_NODE:
        function(*aiter, pControlNode);
        break;
    case CIRCUIT_NODE:
        Traversal(dynamic_cast<AbstractQuantumCircuit *>(*aiter), function);
        break;
    case PROG_NODE:
        Traversal(dynamic_cast<AbstractQuantumProgram *>(*aiter), function);
        break;
    case WHILE_START_NODE:
    case QIF_START_NODE:
        Traversal(dynamic_cast<AbstractControlFlowNode>(*aiter), function);
        break;
    case MEASURE_GATE:
        break;
    default:
        throw exception();
    }
}

#endif // !_TRAVERSAL_DECOMPOSITION_ALGORITHM_H

