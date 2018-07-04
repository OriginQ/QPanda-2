#include "StatisticsQGateCountAlgorithm.h"
#include "QPanda/QPandaException.h"


StatisticsQGateCountAlgorithm::StatisticsQGateCountAlgorithm()
{
}


StatisticsQGateCountAlgorithm::~StatisticsQGateCountAlgorithm()
{
}

size_t StatisticsQGateCountAlgorithm::countQGate(AbstractQuantumCircuit * pQCircuit)
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
            iQGateCount += countQGate(temp);
        }
        else
        {
            throw exception();
        }
    }

    return iQGateCount;

}


size_t countQGateUnderQCircuit(AbstractQuantumCircuit * pQCircuit)
{
    auto temp = dynamic_cast<AbstractQuantumCircuit *>(pQCircuit);
    if (nullptr == temp)
    {
        throw param_error_exception("QCircuit is null", false);
    }

    return StatisticsQGateCountAlgorithm::countQGate(pQCircuit);
}