#include "QGateCounter.h"
using namespace std;
USING_QPANDA
QGateCounter::QGateCounter()
{
}


QGateCounter::~QGateCounter()
{
}

size_t QGateCounter::countQGate(AbstractQuantumCircuit * pQCircuit)
{
    if (nullptr == pQCircuit)
    {
        QCERR("QCircuit is null");
        throw invalid_argument("QCircuit is null");
    }

    int iQGateCount = 0;
    auto aiter = pQCircuit->getFirstNodeIter();
    if (aiter == pQCircuit->getEndNodeIter())
    {
        return iQGateCount;
    }


    for (; aiter != pQCircuit->getEndNodeIter(); ++aiter)
    {
        QNode * pNode = (*aiter).get();

        if (pNode->getNodeType() == GATE_NODE)
        {
            iQGateCount++;
        }
        else if (MEASURE_GATE == pNode->getNodeType())
        {
            iQGateCount++;
        }
        else if (CIRCUIT_NODE == pNode->getNodeType())
        {
            auto temp = dynamic_cast<AbstractQuantumCircuit*>(pNode);
            iQGateCount += (int)countQGate(temp);
        }
        else
        {
            QCERR("node type error");
            throw runtime_error("node type error");
        }
    }

    return iQGateCount;

}

size_t QGateCounter::countQGate(AbstractQuantumProgram * pQProg)
{
    if (nullptr == pQProg)
    {
        QCERR("QCircuit is a nullptr");
        throw invalid_argument("QCircuit is a nullptr");
    }

    int iQGateCount = 0;
    auto aiter = pQProg->getFirstNodeIter();
    if (aiter == pQProg->getEndNodeIter())
    {
        return iQGateCount;
    }

    for (; aiter != pQProg->getEndNodeIter(); ++aiter)
    {
        QNode * pNode = (*aiter).get();

        int iNodeType = pNode->getNodeType();
        if ((GATE_NODE == iNodeType) || (MEASURE_GATE == iNodeType))
        {
            iQGateCount++;
        }
        else if (CIRCUIT_NODE == iNodeType)
        {
            auto temp = dynamic_cast<AbstractQuantumCircuit*>(pNode);
            iQGateCount += (int)countQGate(temp);
        }
        else if(PROG_NODE == iNodeType)
        {
            auto temp = dynamic_cast<AbstractQuantumProgram*>(pNode);
            iQGateCount += (int)countQGate(temp);
        }
        else if ((WHILE_START_NODE == iNodeType) || (QIF_START_NODE == iNodeType))
        {
            auto temp = dynamic_cast<AbstractControlFlowNode*>(pNode);
            iQGateCount += (int)countQGate(temp);
        }
        else
        {
            QCERR("node type error");
            throw runtime_error("node type error");
        }
    }

    return iQGateCount;
}

size_t QGateCounter::countQGate(AbstractControlFlowNode * pControlFlow)
{
    if (nullptr == pControlFlow)
    {
        QCERR("pControlFlow is a nullptr");
        throw invalid_argument("pControlFlow is a nullptr");
    }

    auto pNode = dynamic_cast<QNode*>(pControlFlow);

    if (nullptr == pNode)
    {
        QCERR("pControlFlow type error");
        throw runtime_error("pControlFlow type error");
    }

    int iNodeType = pNode->getNodeType();
    size_t iQGateCount = 0;

    if (WHILE_START_NODE == iNodeType)
    {
        auto pTrueBranch = pControlFlow->getTrueBranch();
        iQGateCount += countControlFlowQGate(pTrueBranch);
        return iQGateCount;
    }
    else if(QIF_START_NODE == iNodeType)
    {
        QNode * pBranch = pControlFlow->getTrueBranch();
        iQGateCount += countControlFlowQGate(pBranch);

        pBranch = pControlFlow->getFalseBranch();
        if (nullptr != pBranch)
        {
            iQGateCount += countControlFlowQGate(pBranch);
        }
        return iQGateCount;
    }
    else
    {
        QCERR("node type error");
        throw runtime_error("node type error");
    }
}

size_t QGateCounter::countControlFlowQGate(QNode * pNode)
{
    size_t iQGateCount = 0;

    int iNodeType = pNode->getNodeType();

    if ((GATE_NODE == iNodeType) || (MEASURE_GATE == iNodeType))
    {
        iQGateCount++;
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        auto temp = dynamic_cast<AbstractQuantumCircuit*>(pNode);
        iQGateCount += countQGate(temp);
    }
    else if (PROG_NODE == iNodeType)
    {
        auto temp = dynamic_cast<AbstractQuantumProgram*>(pNode);
        iQGateCount += countQGate(temp);
    }
    else if ((WHILE_START_NODE == iNodeType) || (QIF_START_NODE == iNodeType))
    {
        auto temp = dynamic_cast<AbstractControlFlowNode*>(pNode);
        iQGateCount += countQGate(temp);
    }

    return iQGateCount;
}


size_t QPanda::countQGateUnderQCircuit(AbstractQuantumCircuit * pQCircuit)
{
    if (nullptr == pQCircuit)
    {
        QCERR("pQCircuit is a nullptr");
        throw invalid_argument("pQCircuit is a nullptr");
    }

    return QGateCounter::countQGate(pQCircuit);
}

size_t QPanda::countQGateUnderQProg(AbstractQuantumProgram * pQProg)
{
    if (nullptr == pQProg)
    {
        QCERR("pQCircuit is a nullptr");
        throw invalid_argument("pQCircuit is a nullptr");
    }

    return QGateCounter::countQGate(pQProg);
}