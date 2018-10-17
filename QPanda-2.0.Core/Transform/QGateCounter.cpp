#include "QGateCounter.h"
#include "QPanda/QPandaException.h"


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
        throw param_error_exception("QCircuit is null",false);
    }

    int iQGateCount = 0;
    auto aiter = pQCircuit->getFirstNodeIter();
    if (aiter == pQCircuit->getEndNodeIter())
    {
        return iQGateCount;
    }


    for (; aiter != pQCircuit->getEndNodeIter(); ++aiter)
    {
        QNode * pNode = *aiter;

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
            throw exception();
        }
    }

    return iQGateCount;

}

size_t QGateCounter::countQGate(AbstractQuantumProgram * pQProg)
{
    if (nullptr == pQProg)
    {
        throw param_error_exception("QCircuit is null", false);
    }

    int iQGateCount = 0;
    auto aiter = pQProg->getFirstNodeIter();
    if (aiter == pQProg->getEndNodeIter())
    {
        return iQGateCount;
    }

    for (; aiter != pQProg->getEndNodeIter(); ++aiter)
    {
        QNode * pNode = *aiter;

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
            stringstream ssErrMsg;
            getssErrMsg(ssErrMsg, "Unknown error");
            throw QPandaException(ssErrMsg.str(), false);
        }
    }

    return iQGateCount;
}

size_t QGateCounter::countQGate(AbstractControlFlowNode * pControlFlow)
{
    if (nullptr == pControlFlow)
    {
        stringstream ssErrMsg;
        getssErrMsg(ssErrMsg, "param is null");
        throw param_error_exception(ssErrMsg.str(), false);
    }

    auto pNode = dynamic_cast<QNode *>(pControlFlow);

    if (nullptr == pNode)
    {
        stringstream ssErrMsg;
        getssErrMsg(ssErrMsg, "Unknown error");
        throw QPandaException(ssErrMsg.str(), false);
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
        stringstream ssErrMsg;
        getssErrMsg(ssErrMsg, "Unknown error");
        throw QPandaException(ssErrMsg.str(), false);
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


size_t countQGateUnderQCircuit(AbstractQuantumCircuit * pQCircuit)
{
    if (nullptr == pQCircuit)
    {
        throw param_error_exception("QCircuit is null", false);
    }

    return QGateCounter::countQGate(pQCircuit);
}

size_t countQGateUnderQProg(AbstractQuantumProgram * pQProg)
{
    if (nullptr == pQProg)
    {
        throw param_error_exception("QCircuit is null", false);
    }

    return QGateCounter::countQGate(pQProg);
}