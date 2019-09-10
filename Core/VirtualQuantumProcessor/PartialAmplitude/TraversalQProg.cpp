#include "Core/VirtualQuantumProcessor/PartialAmplitude/TraversalQProg.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Transform/QRunesToQProg.h"
using namespace std;
USING_QPANDA

void TraversalQProg::traversal(AbstractQuantumProgram *pQProg)
{
    if (nullptr == pQProg)
    {
        QCERR("pQProg is null");
        throw invalid_argument("pQProg is null");
    }
    for (auto iter = pQProg->getFirstNodeIter(); iter != pQProg->getEndNodeIter(); ++iter)
    {
        QNode *pNode = (*iter).get();
        TraversalQProg::traversal(pNode);
    }
}

void TraversalQProg::traversal(QNode *pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }

    switch (pNode->getNodeType())
    {
    case NodeType::GATE_NODE:
        traversal(dynamic_cast<AbstractQGateNode *>(pNode));
        break;

    case NodeType::CIRCUIT_NODE:
        TraversalQProg::traversal(dynamic_cast<AbstractQuantumCircuit *>(pNode));
        break;

    case NodeType::PROG_NODE:
        TraversalQProg::traversal(dynamic_cast<AbstractQuantumProgram *>(pNode));
        break;
        
    case NodeType::MEASURE_GATE:
        traversal(dynamic_cast<AbstractQuantumMeasure *>(pNode));
        break;

    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
    case NodeType::NODE_UNDEFINED:
    default:
        QCERR("UnSupported Node");
        throw undefine_error("QNode");
        break;
    }
}

void TraversalQProg::handleDaggerNode(QNode *pNode,int nodetype)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }
    if (NodeType::GATE_NODE == nodetype)
    {
        AbstractQGateNode *pGATE = dynamic_cast<AbstractQGateNode *>(pNode);
        pGATE->setDagger(!pGATE->isDagger());
        traversal(pGATE);
    }
    else if (CIRCUIT_NODE == nodetype)
    {
        AbstractQuantumCircuit *qCircuit = dynamic_cast<AbstractQuantumCircuit *>(pNode);
        qCircuit->setDagger(!qCircuit->isDagger());
        traversal(qCircuit);
    }
    else
    {
        QCERR("node type error");
        throw invalid_argument("node type error");
    }
}

void TraversalQProg::handleDaggerCircuit(QNode *pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }
    switch (pNode->getNodeType())
    {
    case NodeType::GATE_NODE:
        TraversalQProg::handleDaggerNode(pNode, NodeType::GATE_NODE);
        break;

    case NodeType::CIRCUIT_NODE: 
        TraversalQProg::handleDaggerNode(pNode, NodeType::CIRCUIT_NODE);
        break;

    case NodeType::PROG_NODE:
        TraversalQProg::traversal(dynamic_cast<AbstractQuantumProgram *>(pNode));
        break;

    case NodeType::MEASURE_GATE:
    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
    case NodeType::NODE_UNDEFINED:
    default:
        QCERR("UnSupported Node");
        throw undefine_error("QNode");
        break;
    }
}

void TraversalQProg::traversal(AbstractQuantumCircuit *pCircuit)
{
    if (nullptr == pCircuit)
    {
        QCERR("pCircuit is null");
        throw invalid_argument("pCircuit is null");
    }
    if (pCircuit->isDagger())
    {
        for (auto iter = pCircuit->getLastNodeIter(); iter != pCircuit->getHeadNodeIter(); --iter)
        {
            QNode *pNode = (*iter).get();
            handleDaggerCircuit(pNode);
        }
    }
    else
    {
        for (auto iter = pCircuit->getFirstNodeIter(); iter != pCircuit->getEndNodeIter(); ++iter)
        {
            QNode *pNode = (*iter).get();
            traversal(pNode);
        }
    }
}

void TraversalQProg::traversalByFile(std::string file_path)
{
    auto prog = QProg();
    auto qvm = new CPUQVM();
    Configuration config;
    config.maxCMem = config.maxQubit = 49;
    qvm->setConfig(config);
    qvm->init();
    transformQRunesToQProg(file_path, prog, qvm);
    traversal(dynamic_cast<AbstractQuantumProgram *>
        (prog.getImplementationPtr().get()));
    qvm->finalize();
    delete qvm;
}
