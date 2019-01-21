/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "./QCircuitParse.h"
#include "QuantumCircuit/QGlobalVariable.h"
using namespace std;
USING_QPANDA
QGATE_FUN_MAP QGateParseMap::m_QGateFunctionMap = {};

QCirCuitParse::QCirCuitParse(AbstractQuantumCircuit * pNode,
    QuantumGateParam * pParam, 
    QuantumGates* pGates,
    bool isDagger, 
    vector<Qubit *> controlQbitVector) :
    m_pNode(pNode), m_pGates(pGates), m_pParam(pParam), m_bDagger(isDagger)
{
    for (auto aiter : controlQbitVector)
    {
        m_controlQubitVector.push_back(aiter);
    }
}

QNodeAgency * QCirCuitParse::getAgency(QNode * pNode)
{
    AbstractQuantumCircuit *temp = (dynamic_cast<AbstractQuantumCircuit *>(m_pNode));

    if (nullptr == temp)
    {
        QCERR("pNode type error");
        throw runtime_error("pNode type error");
    }

    QNodeAgency * childAgency = nullptr;
    vector<Qubit *> controlQubitVector;
    temp->getControlVector(controlQubitVector);

    for (auto aQubit : m_controlQubitVector)
    {
        controlQubitVector.push_back(aQubit);
    }

    if (pNode->getNodeType() == GATE_NODE)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQGateNode*>(pNode),
                                      m_pGates,
                                      m_pNode->isDagger()^ m_bDagger,
                                      controlQubitVector);
    }
    else if (MEASURE_GATE == pNode->getNodeType())
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumMeasure*>(pNode),
                                      m_pParam,
                                      m_pGates);
    }
    else if (CIRCUIT_NODE == pNode->getNodeType())
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumCircuit*>(pNode),
                                      m_pParam,
                                      m_pGates, 
                                      m_pNode->isDagger() ^ m_bDagger,
                                      controlQubitVector);
    }
    else
    {
        QCERR("pNode type error");
        throw runtime_error("pNode type error");
    }

    return childAgency;
}

bool QCirCuitParse::executeAction()
{
    AbstractQuantumCircuit *temp = (dynamic_cast<AbstractQuantumCircuit *>(m_pNode));
    if (temp == nullptr)
    {
        return false;
    }

    bool isDagger = temp->isDagger() ^ m_bDagger;

    if (isDagger)
    {
        auto aiter = temp->getLastNodeIter();
        if (nullptr == *aiter)
        {
            return true;
        }
        for (; aiter != temp->getHeadNodeIter(); --aiter)
        {
            QNode * pNode = *aiter;
            QNodeAgency *pQNodeAgency = getAgency(pNode);;

            if (false == pQNodeAgency->executeAction())
            {
                delete pQNodeAgency;
                return false;
            }
            delete pQNodeAgency;
        }

    }
    else
    {
        auto aiter = temp->getFirstNodeIter();
        if (nullptr == *aiter)
        {
            return true;
        }
        for (; aiter != temp->getEndNodeIter(); ++aiter)
        {
            QNode * pNode = *aiter;
            vector<Qubit *> controlQubitVector;
            temp->getControlVector(controlQubitVector);

            for (auto aQubit : m_controlQubitVector)
            {
                controlQubitVector.push_back(aQubit);
            }

            QNodeAgency *pQNodeAgency = getAgency(pNode);

            if (false == pQNodeAgency->executeAction())
            {
                delete pQNodeAgency;
                return false;
            }
            delete pQNodeAgency;
        }
    }
    return true;
}

bool QCirCuitParse::verify()
{
    AbstractQuantumCircuit *temp = (dynamic_cast<AbstractQuantumCircuit *>(m_pNode));
    if (nullptr == temp)
    {
        return false;
    }

    bool isDagger = temp->isDagger() ^ m_bDagger;
    if (isDagger)
    {
        auto aiter = temp->getLastNodeIter();
        if (nullptr == *aiter)
        {
            return true;
        }
        for (; aiter != temp->getHeadNodeIter(); --aiter)
        {
            QNode * pNode = *aiter;
            QNodeAgency *pQNodeAgency = getAgency(pNode);
            
            if (false == pQNodeAgency->verify())
            {
                delete pQNodeAgency;
                return false;
            }
            delete pQNodeAgency;
        }
    }
    else
    {
        auto aiter = temp->getFirstNodeIter();
        if (nullptr == *aiter)
        {
            return true;
        }
        for (; aiter != temp->getEndNodeIter(); ++aiter)
        {
            QNode * pNode = *aiter;
            QNodeAgency *pQNodeAgency = getAgency(pNode);

            if (false == pQNodeAgency->verify())
            {
                delete pQNodeAgency;
                return false;
            }
            delete pQNodeAgency;
        }
    }
    return true;
}


QGateParse::QGateParse(AbstractQGateNode * pNode, QuantumGates * pGates, bool isDagger, vector<Qubit*>& controlQubitVector) : m_isDagger(isDagger)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is null");
        throw invalid_argument("pNode is null");
    }
    else
        m_pNode = pNode;

    m_pGates = pGates;

    for (auto aiter : controlQubitVector)
    {
        m_controlQubitVector.push_back(aiter);
    }
}
bool compareQubit(Qubit * a, Qubit * b)
{
    return a->getPhysicalQubitPtr()->getQubitAddr() < b->getPhysicalQubitPtr()->getQubitAddr();
}

bool Qubitequal(Qubit * a, Qubit * b)
{
    return a->getPhysicalQubitPtr()->getQubitAddr() == b->getPhysicalQubitPtr()->getQubitAddr();
}

bool QGateParse::executeAction()
{
    bool bDagger = m_pNode->isDagger() ^ m_isDagger;
    QuantumGate * pGate = m_pNode->getQGate();
    vector<Qubit * > qubitVector;
    m_pNode->getQuBitVector(qubitVector);
    if (qubitVector.size() <= 0)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
        
    vector<Qubit *> controlvector;
    m_pNode->getControlVector(controlvector);
    for (auto aiter : controlvector)
    {
        m_controlQubitVector.push_back(aiter);
    }
    if (m_controlQubitVector.size() > 0)
    {
        sort(m_controlQubitVector.begin(), m_controlQubitVector.end(), compareQubit);
        m_controlQubitVector.erase(unique(m_controlQubitVector.begin(), m_controlQubitVector.end(), Qubitequal), m_controlQubitVector.end());
    }

    for (auto aQIter : qubitVector)
    {
        for (auto aCIter : controlvector)
        {
            if (Qubitequal(aQIter, aCIter))
            {
                QCERR("targitQubit == controlQubit");
                throw invalid_argument("targitQubit == controlQubit");
            }
        }
    }
    auto aiter = QGateParseMap::getFunction(pGate->getOperationNum());
    aiter(m_pNode->getQGate(), qubitVector, m_pGates, bDagger, m_controlQubitVector);
    return true;
}

bool QGateParse::verify()
{
    vector<Qubit *> qubitVector;
    m_pNode->getQuBitVector(qubitVector);
    m_pNode->getControlVector(qubitVector);
    if (qubitVector.size() <= 0)
    {
        return true;
    }
    for (auto aiter : qubitVector)
    {
        if (!aiter->getOccupancy())
        {
            return false;
        }
    }
    return true;
}

void QGateParseSingleBit(QuantumGate * pGate, vector<Qubit * > & qubitVector, QuantumGates* pGates, bool isDagger, vector<Qubit *> & controlQubitVector)
{
    if (nullptr == pGate)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    QStat matrix;
    pGate->getMatrix(matrix);
    Qubit * pQubit = *(qubitVector.begin());
    size_t bit = pQubit->getPhysicalQubitPtr()->getQubitAddr();
    if (controlQubitVector.size() == 0)
    {
        pGates->unitarySingleQubitGate(bit, matrix, isDagger, 0);
    }
    else
    {
        size_t sTemp;
        vector<size_t> bitNumVector;
        for (auto aiter : controlQubitVector)
        {
            sTemp = aiter->getPhysicalQubitPtr()->getQubitAddr();
            bitNumVector.push_back(sTemp);
        }
        bitNumVector.push_back(bit);
        pGates->controlunitarySingleQubitGate(bit, bitNumVector, matrix, isDagger, 0);
    }

}

void QGateParseDoubleBit(QuantumGate * pGate, vector<Qubit * > & qubitVector, QuantumGates* pGates, bool isDagger, vector<Qubit *> & controlQubitVector)
{
    QStat matrix;
    pGate->getMatrix(matrix);
    auto aiter = qubitVector.begin();
    Qubit * pQubit = *aiter;
    aiter++;
    Qubit * pQubit2 = *aiter;
    size_t bit = pQubit->getPhysicalQubitPtr()->getQubitAddr();
    size_t bit2 = pQubit2->getPhysicalQubitPtr()->getQubitAddr();

    if (controlQubitVector.size() == 0)
    {
        pGates->unitaryDoubleQubitGate(bit, bit2, matrix, isDagger, 0);
    }
    else
    {
        size_t sTemp;
        vector<size_t> bitNumVector;
        for (auto aiter : controlQubitVector)
        {
            sTemp = aiter->getPhysicalQubitPtr()->getQubitAddr();
            bitNumVector.push_back(sTemp);
        }
        bitNumVector.push_back(bit2);
        bitNumVector.push_back(bit);
        pGates->controlunitaryDoubleQubitGate(bit, bit2, bitNumVector, matrix, isDagger, 0);
    }
}

#define REGISTER_QGATE_PARSE(BitCount,FunctionName) \
class insertQGateMapHelper_##FunctionName \
{ \
public: \
     inline insertQGateMapHelper_##FunctionName(int bitCount,QGATE_FUN pFunction) \
    { \
        QGateParseMap::insertMap(bitCount, pFunction); \
    } \
};\
insertQGateMapHelper_##FunctionName _G_insertQGateHelper##FunctionName(BitCount, FunctionName)

QNodeAgency::QNodeAgency(AbstractQuantumCircuit * pNode, QuantumGateParam * pParam, QuantumGates * pGates, bool isDgger, vector<Qubit*> controlQubitVector)
{
    m_pQNodeParse = new QCirCuitParse(pNode, pParam, pGates, isDgger, controlQubitVector);
}

QNodeAgency::QNodeAgency(AbstractControlFlowNode * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    auto pQNode = dynamic_cast<QNode *>(pNode);
    if (QIF_START_NODE == pQNode->getNodeType())
        m_pQNodeParse = new QIfParse(pNode, pParam, pGates);
    else if (WHILE_START_NODE == pQNode->getNodeType())
        m_pQNodeParse = new QWhileParse(pNode, pParam, pGates);
    else
    {
        QCERR("this node is not controlflow");
        throw invalid_argument("this node is not controlflow");
    }

}

QNodeAgency::QNodeAgency(AbstractQuantumMeasure * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    m_pQNodeParse = new MeasureParse(pNode, pParam, pGates);
}

QNodeAgency::QNodeAgency(AbstractQuantumProgram * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    m_pQNodeParse = new QProgParse(pNode, pParam, pGates);
}

QNodeAgency::QNodeAgency(AbstractClassicalProg * pNode)
{
    m_pQNodeParse = new ClassicalProgParse(pNode);
}

QNodeAgency::QNodeAgency(AbstractQGateNode * pNode, QuantumGates * pGates, bool isDagger, vector<Qubit*>& controlQubitVector)
{
    m_pQNodeParse = new QGateParse(pNode, pGates, isDagger, controlQubitVector);
}

QNodeAgency::~QNodeAgency()
{
    if (nullptr != m_pQNodeParse)
    {
        delete m_pQNodeParse;
        m_pQNodeParse = nullptr;
    }
}

bool QNodeAgency::executeAction()
{
    if (nullptr != m_pQNodeParse)
    {
        return m_pQNodeParse->executeAction();
    }
    return false;
}

bool QNodeAgency::verify()
{
    if (nullptr != m_pQNodeParse)
    {
        return m_pQNodeParse->verify();
    }
    return false;
}

QIfParse::QIfParse(AbstractControlFlowNode * pNode, QuantumGateParam * pParam, QuantumGates * pGates) :m_pNode(pNode), m_pGates(pGates), m_pParam(pParam)
{
}

QNodeAgency * QIfParse::getAgency( QNode * pNode)
{
    QNodeAgency * childAgency = nullptr;
    auto iNodeType = pNode->getNodeType();
    if (GATE_NODE == iNodeType)
    {
        vector<Qubit *> controlVector;
        childAgency = new QNodeAgency(dynamic_cast<AbstractQGateNode*>(pNode), m_pGates, false, controlVector);
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        vector<Qubit *> controlQubitVector;
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumCircuit*>(pNode), m_pParam, m_pGates, false, controlQubitVector);
    }
    else if (PROG_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumProgram*>(pNode), m_pParam, m_pGates);
    }
    else if (MEASURE_GATE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumMeasure*>(pNode), m_pParam, m_pGates);
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractControlFlowNode*>(pNode), m_pParam, m_pGates);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractControlFlowNode*>(pNode), m_pParam, m_pGates);
    }
    else
    {
        QCERR("node type error");
        throw invalid_argument("node type error");
    }

    return childAgency;
}

bool QIfParse::executeAction()
{
    AbstractControlFlowNode * pQIfNode = dynamic_cast<AbstractControlFlowNode*>(m_pNode);
    auto aCExpr = pQIfNode->getCExpr();
    QNode * pQNode;
    if (aCExpr->eval())
    {
        pQNode = pQIfNode->getTrueBranch();
        if (nullptr == pQNode)
        {
            return true;
        }
    }
    else
    {
        pQNode = pQIfNode->getFalseBranch();
        if (nullptr == pQNode)
        {
            return true;
        }
    }
    QNodeAgency * pTempAgency = getAgency(pQNode);


    if (!pTempAgency->executeAction())
    {
        delete pTempAgency;
        return false;
    }

    delete pTempAgency;
    return true;
}

bool QIfParse::verify()
{
    AbstractControlFlowNode * pQIfNode = dynamic_cast<AbstractControlFlowNode*>(m_pNode);
    auto aCExpr = pQIfNode->getCExpr();
    QNode * pQNode;

    if ((pQNode = pQIfNode->getTrueBranch()) != nullptr)
    {
        QNodeAgency * pTempAgency = getAgency(pQNode);

        if (!pTempAgency->verify())
        {
            delete pTempAgency;
            return false;
        }
        delete pTempAgency;
    }

    if ((pQNode = pQIfNode->getFalseBranch()) != nullptr)
    {
        QNodeAgency * pTempAgency = getAgency(pQNode);
        
        if (!pTempAgency->verify())
        {
            delete pTempAgency;
            return false;
        }
        delete pTempAgency;
    }

    return true;
}

QWhileParse::QWhileParse(AbstractControlFlowNode * pNode, QuantumGateParam * pParam, QuantumGates * pGates) :m_pNode(pNode), m_pGates(pGates), m_pParam(pParam)
{
}

QNodeAgency * QWhileParse::getAgency(QNode * pNode)
{
    int iNodeType = pNode->getNodeType();
    QNodeAgency *childAgency = nullptr;
    if (GATE_NODE == iNodeType)
    {
        vector<Qubit *> controlVector;
        childAgency = new QNodeAgency(dynamic_cast<AbstractQGateNode*>(pNode), m_pGates, false, controlVector);
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        vector<Qubit *> controlQubitVector;
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumCircuit*>(pNode), m_pParam, m_pGates, false, controlQubitVector);
    }
    else if (PROG_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumProgram*>(pNode), m_pParam, m_pGates);
    }
    else if (MEASURE_GATE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumMeasure*>(pNode), m_pParam, m_pGates);
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractControlFlowNode*>(pNode), m_pParam, m_pGates);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractControlFlowNode*>(pNode), m_pParam, m_pGates);
    }
    else
    {
        QCERR("node type error");
        throw invalid_argument("node type error");
    }

    return childAgency;
}
bool QWhileParse::executeAction()
{
    AbstractControlFlowNode * pQWhileNode = dynamic_cast<AbstractControlFlowNode*>(m_pNode);
    auto aCExpr = pQWhileNode->getCExpr();
    QNode * pQNode;
    while (aCExpr->eval())
    {
        pQNode = pQWhileNode->getTrueBranch();
        if (pQNode == nullptr)
        {
            return true;
        }
        QNodeAgency * pTempAgency = getAgency(pQNode);
        
        if (!pTempAgency->executeAction())
        {
            delete pTempAgency;
            return false;
        }

        delete pTempAgency;
    }
    return true;
}

bool QWhileParse::verify()
{
    AbstractControlFlowNode * pQWhileNode = dynamic_cast<AbstractControlFlowNode*>(m_pNode);
    auto aCExpr = pQWhileNode->getCExpr();
    QNode * pQNode;
    pQNode = pQWhileNode->getTrueBranch();
    if (pQNode == nullptr)
    {
        return true;
    }

    int iNodeType = pQNode->getNodeType();
    QNodeAgency * pTempAgency = getAgency(pQNode);
    if (!pTempAgency->verify())
    {
        delete pTempAgency;
        return false;
    }

    delete pTempAgency;

    return true;
}

MeasureParse::MeasureParse(AbstractQuantumMeasure * pNode, QuantumGateParam * pParam, QuantumGates * pGates) :m_pNode(pNode), m_pGates(pGates), m_pParam(pParam)
{

}

bool MeasureParse::executeAction()
{
    AbstractQuantumMeasure * pMeasureParse = dynamic_cast<AbstractQuantumMeasure *>(m_pNode);
    int iResult = m_pGates->qubitMeasure(pMeasureParse->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
    if (iResult < 0)
    {
        return false;
    }
    CBit * pCExpr = pMeasureParse->getCBit();
    pCExpr->setValue(iResult);
    if (nullptr == pCExpr)
    {
        QCERR("unknow error");
        throw runtime_error("unknow error");
    }
    string sName = pCExpr->getName();
    auto aiter = m_pParam->mReturnValue.find(sName);
    if (aiter != m_pParam->mReturnValue.end())
    {
        aiter->second = (bool)iResult;
    }
    else
    {
        m_pParam->mReturnValue.insert(pair<string, bool>(sName, (bool)iResult));
    }

    return true;
}

bool MeasureParse::verify()
{
    AbstractQuantumMeasure * pMeasureParse = dynamic_cast<AbstractQuantumMeasure *>(m_pNode);
    if (!pMeasureParse->getQuBit()->getPhysicalQubitPtr()->getOccupancy())
    {
        return false;
    }
    return pMeasureParse->getCBit()->getOccupancy();
}

QProgParse::QProgParse(AbstractQuantumProgram * pNode,
                       QuantumGateParam * pParam,
                       QuantumGates * pGates) :
                       m_pNode(pNode),
                       m_pGates(pGates),
                       m_pParam(pParam)
{

}

QNodeAgency * QProgParse::getAgency(QNode * pNode)
{
    int iNodeType = pNode->getNodeType();
    QNodeAgency * childAgency = nullptr;
    if (GATE_NODE == iNodeType)
    {
        vector<Qubit *> controlVector;
        childAgency = new QNodeAgency(dynamic_cast<AbstractQGateNode*>(pNode), m_pGates, false, controlVector);
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        vector<Qubit *> controlQubitVector;
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumCircuit*>(pNode), m_pParam, m_pGates, false, controlQubitVector);
    }
    else if (PROG_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumProgram*>(pNode), m_pParam, m_pGates);
    }
    else if (MEASURE_GATE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractQuantumMeasure*>(pNode), m_pParam, m_pGates);
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractControlFlowNode*>(pNode), m_pParam, m_pGates);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractControlFlowNode*>(pNode), m_pParam, m_pGates);
    }
    else if (CLASS_COND_NODE == iNodeType)
    {
        childAgency = new QNodeAgency(dynamic_cast<AbstractClassicalProg*>(pNode));
    }
    else
    {
        QCERR("node type error");
        throw runtime_error("node type error");
    }

    return childAgency;
}

bool QProgParse::executeAction()
{
    auto aiter = m_pNode->getFirstNodeIter();
    if (nullptr == *aiter)
    {
        return true;
    }

    for (; aiter != m_pNode->getEndNodeIter(); ++aiter)
    {
        QNode * pQNode = *aiter;
        if (nullptr == pQNode)
        {
            return false;
        }

        QNodeAgency * pTempAgency = getAgency(pQNode);

        if (!pTempAgency->executeAction())
        {
            delete pTempAgency;
            return false;
        }

        delete pTempAgency;
    }
    return true;
}

bool QProgParse::verify()
{
    auto aiter = m_pNode->getFirstNodeIter();
    if (nullptr == *aiter)
    {
        return true;
    }

    for (; aiter != m_pNode->getEndNodeIter(); ++aiter)
    {
        QNode * pQNode = *aiter;
        if (nullptr == pQNode)
        {
            return true;
        }
        QNodeAgency * pTempAgency = getAgency(pQNode);

        if (!pTempAgency->verify())
        {
            delete pTempAgency;
            return false;
        }

        delete pTempAgency;
    }
    return true;
}

REGISTER_QGATE_PARSE(1, QGateParseSingleBit);
REGISTER_QGATE_PARSE(2, QGateParseDoubleBit);

ClassicalProgParse::ClassicalProgParse(AbstractClassicalProg * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    m_pNode = pNode;
}

bool ClassicalProgParse::executeAction()
{
    if (nullptr == m_pNode)
    {
        QCERR("ClassicalProgParse m_pnode is null");
        throw runtime_error("ClassicalProgParse m_pnode is null");
    }

    auto result = m_pNode->eval();
    return true;
}

bool ClassicalProgParse::verify()
{
    return true;
}

QNodeAgency * ClassicalProgParse::getAgency(QNode * pNode)
{
    return nullptr;
}
