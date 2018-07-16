#ifndef _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#define _TRAVERSAL_DECOMPOSITION_ALGORITHM_H
#include "QProgram.h"
class TraversalDecompositionAlgorithm
{
public:
    TraversalDecompositionAlgorithm(vector<vector<string>> &ValidQGateMatrix);
    ~TraversalDecompositionAlgorithm();

    template<typename T>
    QNode * traversalDecomposition(T pNode);
private:
    vector<vector<string>> m_ValidQGateMatrix;

    TraversalDecompositionAlgorithm() {};

    void doubleGateToCNOTAndSingleGate(AbstractQGateNode *, QNode *) {};

    void multipleControlGateToQCircuit(AbstractQGateNode *, QNode *) {};

    void CNOTToControlSingleGate(AbstractQGateNode *, QNode *);

    void controlSingleGateToMetadataDoubleGate(AbstractQGateNode *, QNode *) {};

    void singGateToMetadataSingleGate(AbstractQGateNode *, QNode *);

    template<typename T>
    void Traversal(AbstractControlFlowNode *, T);
    template<typename T>
    void Traversal(AbstractQuantumCircuit *, T);
    template<typename T>
    void Traversal(AbstractQuantumProgram *, T);
    template<typename T>
    void TraversalByType(QNode * pNode, QNode *, T);
};

template<typename T>
inline void TraversalDecompositionAlgorithm::traversalDecomposition(T node)
{
    QNode * pNode = dynamic_cast<QNode *> (node);
    if (nullptr == pNode)
        throw param_error_exception("this param is not QNode", false);
    
    int iNodeType = pNode->getNodeType();
    if((GATE_NODE == iNodeType) || (MEASURE_GATE == iNodeType))
        throw param_error_exception("the param cannot be a QGate or Measure", false);
    Traversal(node, doubleGateToCNOTAndSingleGate);
    Traversal(node, CNOTToControlSingleGate);
    Traversal(node, multipleControlGateToQCircuit);
    Traversal(node, controlSingleGateToMetadataDoubleGate);
    Traversal(node, singGateToMetadataSingleGate);
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
        TraversalByType(pControlNode->getTrueBranch(), pControlNode, function);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        TraversalByType(pControlNode->getTrueBranch(), pControlNode,function);
        TraversalByType(pControlNode->getFalseBranch(), pControlNode,function);
    }
}

template<typename T>
inline void TraversalDecompositionAlgorithm::Traversal(AbstractQuantumCircuit * pQCircuit, T function)
{
    if (nullptr == pQCircuit)
        throw param_error_exception("param error", false);

    auto aiter = pQCircuit->getFirstNodeIter();
    if (aiter == pQCircuit->getEndNodeIter())
        return;
    for (; aiter != pQCircuit->getEndNodeIter(); ++aiter)
    {
        int iNodeType = (*aiter)->getNodeType();
        switch (iNodeType)
        {
        case GATE_NODE:
            function(*aiter, pQCircuit);
            break;
        case CIRCUIT_NODE:
            Traversal(dynamic_cast<AbstractQuantumCircuit *>(*aiter), pQCircuit, function);
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
        TraversalByType((*aiter), pProg, function);
    }
}

template<typename T >
inline void TraversalDecompositionAlgorithm::TraversalByType(QNode * pNode, QNode * fatherNode, T function)
{
    int iNodeType = pNode->getNodeType();
    if (-1 == iNodeType)
        throw param_error_exception("param error", false);
    switch (iNodeType)
    {
    case GATE_NODE:
        function(pNode, fatherNode);
        break;
    case CIRCUIT_NODE:
        Traversal(dynamic_cast<AbstractQuantumCircuit *>(pNode), function);
        break;
    case PROG_NODE:
        Traversal(dynamic_cast<AbstractQuantumProgram *>(pNode), function);
        break;
    case WHILE_START_NODE:
    case QIF_START_NODE:
        Traversal(dynamic_cast<AbstractControlFlowNode>(pNode), function);
        break;
    case MEASURE_GATE:
        break;
    default:
        throw exception();
    }
}


#endif // !_TRAVERSAL_DECOMPOSITION_ALGORITHM_H

