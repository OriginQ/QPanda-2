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
#include "../QuantumCircuit/QGlobalVariable.h"

QGATE_FUN_MAP QGateParseMap::m_QGateFunctionMap = {};



void QGateParse::singleGateAngletoNum(double alpha, double beta, double gamma, double delta,QStat &matrix)
{
    matrix[0] = COMPLEX(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2));
    matrix[1] = COMPLEX(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2));
    matrix[2] = COMPLEX(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2));
    matrix[3] = COMPLEX(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2));
}

void QGateParse::DoubleGateAngletoNum(double alpha, double beta, double gamma, double delta, QStat & matrix)
{
    matrix[10] = COMPLEX(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2));
    matrix[11] = COMPLEX(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2));
    matrix[14] = COMPLEX(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2));
    matrix[15] = COMPLEX(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2));
    matrix[0] = 1;
    matrix[5] = 1;
}



QCirCuitParse::QCirCuitParse(QCircuit * pNode,QuantumGateParam * pParam, QuantumGates* pGates):m_pNode(pNode), m_pGates(pGates),m_pParam(pParam)
{

}

bool QCirCuitParse::executeAction()
{
    AbstractQuantumCircuit *temp =(dynamic_cast<AbstractQuantumCircuit *>(m_pNode));
    if (temp == nullptr)
    {
        return false;
    }
    if (temp->isDagger())
    {
        auto aiter = temp->getLastNodeIter();
        if (nullptr == *aiter)
        {
            return true;
        }
        for (; aiter != temp->getHeadNodeIter(); --aiter)
        {
            QNode * pNode = *aiter;
            vector<Qubit *> controlQubitVector;
            temp->getControlVector(controlQubitVector);
            QNodeAgency * pQNodeAgency =  nullptr;
            if (pNode->getNodeType() == GATE_NODE)
            {
                pQNodeAgency = new QNodeAgency((QGate*)pNode, m_pGates, true, controlQubitVector);
            }
            else
            {
                pQNodeAgency = new QNodeAgency((QMeasure*)pNode,m_pParam, m_pGates);
            }
            
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
            QNodeAgency * pQNodeAgency = nullptr;
            if (pNode->getNodeType() == GATE_NODE)
            {
                pQNodeAgency = new QNodeAgency((QGate*)pNode, m_pGates, false, controlQubitVector);
            }
            else
            {
                pQNodeAgency = new QNodeAgency((QMeasure*)pNode, m_pParam, m_pGates);
            }

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
    if (temp == nullptr)
    {
        return false;
    }
    if (temp->isDagger())
    {
        auto aiter = temp->getLastNodeIter();
        if (nullptr == *aiter)
        {
            return true;
        }
        for (; aiter != temp->getHeadNodeIter(); --aiter)
        {
            QNode * pNode = *aiter;
            vector<Qubit *> controlQubitVector;
            temp->getControlVector(controlQubitVector);
            QNodeAgency * pQNodeAgency = nullptr;
            if (pNode->getNodeType() == GATE_NODE)
            {
                pQNodeAgency = new QNodeAgency((QGate*)pNode, m_pGates, true, controlQubitVector);
            }
            else
            {
                pQNodeAgency = new QNodeAgency((QMeasure*)pNode, m_pParam, m_pGates);
            }

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
            vector<Qubit *> controlQubitVector;
            temp->getControlVector(controlQubitVector);
            QNodeAgency * pQNodeAgency = nullptr;
            if (pNode->getNodeType() == GATE_NODE)
            {
                pQNodeAgency = new QNodeAgency((QGate*)pNode, m_pGates, false, controlQubitVector);
            }
            else
            {
                pQNodeAgency = new QNodeAgency((QMeasure*)pNode, m_pParam, m_pGates);
            }

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


QGateParse::QGateParse(QNode * pNode, QuantumGates * pGates, bool isDagger, vector<Qubit*>& controlQubitVector) : m_isDagger(isDagger)
{
    //pNode;
    if (nullptr == pNode)
        throw exception();
    else
        m_pNode = pNode;

    m_pGates = pGates;

    for (auto aiter : controlQubitVector)
    {
        m_controlQubitVector.push_back(aiter);
    }
}

bool QGateParse::executeAction()
{
    QGateNode * temp = dynamic_cast<QGateNode *>(m_pNode);
    bool bDagger = temp->isDagger() ^ m_isDagger;
    QuantumGate * pGate = temp->getQGate();
    vector<Qubit * > qubitVector;
    temp->getQuBitVector(qubitVector);
    vector<Qubit *> controlvector;
    temp->getControlVector(controlvector);
    for (auto aiter : controlvector)
    {
        m_controlQubitVector.push_back(aiter);
    }
    auto aiter = QGateParseMap::getFunction(pGate->getOpNum());
    aiter(temp->getQGate(),qubitVector, m_pGates, bDagger, m_controlQubitVector);
    return true;
}

bool QGateParse::verify()
{
    QGate * temp = dynamic_cast<QGate *>(m_pNode);
    vector<Qubit *> qubitVector;
    temp->getQuBitVector(qubitVector);
    if (qubitVector.size() <= 0)
    {
        return true;
    }
    for (auto aiter : qubitVector)
    {
        if (!aiter->getPhysicalQubitPtr()->getOccupancy())
        {
            return false;
        }
    }

    return true;
}

void QGateParseOneBit(QuantumGate * pGate,vector<Qubit * > & qubitVector, QuantumGates* pGates, bool isDagger, vector<Qubit *> & controlQubitVector)
{
    if (nullptr == pGate)
        throw exception();
    //if (nullptr == pGates)
      //  throw exception();
    QStat matrix(4, 0);
    QGateParse::singleGateAngletoNum(pGate->getAlpha(),pGate->getBeta(), pGate->getGamma(), pGate->getDelta(), matrix);
    Qubit * pQubit = *(qubitVector.begin());
    size_t bit = pQubit->getPhysicalQubitPtr()->getQubitAddr();
    if (isDagger)
    {
        if (controlQubitVector.size() == 0)
        {
            pGates->unitarySingleQubitGateDagger(bit, matrix, 0);
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
            pGates->controlunitarySingleQubitGateDagger(bitNumVector, matrix, 0);
        }
    }
    else
    {
        if (controlQubitVector.size() == 0)
        {
            pGates->unitarySingleQubitGate(bit, matrix, 0);
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
            pGates->controlunitarySingleQubitGate(bitNumVector, matrix, 0);
        }
    }
}


void QGateParseTwoBit(QuantumGate * pGate, vector<Qubit * > & qubitVector, QuantumGates* pGates, bool isDagger, vector<Qubit *> & controlQubitVector)
{
    QStat matrix(16, 0);
    QGateParse::DoubleGateAngletoNum(pGate->getAlpha(), pGate->getBeta(), pGate->getGamma(), pGate->getDelta(), matrix);
    auto aiter = qubitVector.begin();
    Qubit * pQubit = *aiter;
    aiter++;
    Qubit * pQubit2 = *aiter;
    size_t bit = pQubit->getPhysicalQubitPtr()->getQubitAddr();
    size_t bit2 = pQubit2->getPhysicalQubitPtr()->getQubitAddr();
    if (isDagger)
    {
        if (controlQubitVector.size() == 0)
        {
            pGates->unitaryDoubleQubitGateDagger(bit, bit2, matrix, 0);
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
            pGates->controlunitaryDoubleQubitGateDagger(bitNumVector, matrix, 0);
        }
    }
    else
    {
        if (controlQubitVector.size() == 0)
        {
            pGates->unitaryDoubleQubitGate(bit, bit2, matrix, 0);
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
            pGates->controlunitaryDoubleQubitGate(bitNumVector, matrix, 0);
        }
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




QNodeAgency::QNodeAgency(QCircuit * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    m_pQNodeParse = new QCirCuitParse(pNode, pParam, pGates);
}

QNodeAgency::QNodeAgency(QIfProg * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    m_pQNodeParse = new QIfParse(pNode, pParam, pGates);
}

QNodeAgency::QNodeAgency(QWhileProg * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    m_pQNodeParse = new QWhileParse(pNode, pParam, pGates);
}


QNodeAgency::QNodeAgency(QMeasure * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    m_pQNodeParse = new MeasureParse(pNode, pParam, pGates);
}

QNodeAgency::QNodeAgency(QProg * pNode, QuantumGateParam * pParam, QuantumGates * pGates)
{
    m_pQNodeParse = new QProgParse(pNode, pParam, pGates);
}

QNodeAgency::QNodeAgency(QGate * pNode, QuantumGates * pGates, bool isDagger, vector<Qubit*>& controlQubitVector)
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

QIfParse::QIfParse(QIfProg * pNode, QuantumGateParam * pParam, QuantumGates * pGates) :m_pNode(pNode), m_pGates(pGates), m_pParam(pParam)
{
}

bool QIfParse::executeAction()
{
    AbstractControlFlowNode * pQIfNode = dynamic_cast<AbstractControlFlowNode*>(m_pNode);
    auto aCExpr = pQIfNode->getCExpr();
    QNode * pQNode;
    if (aCExpr->eval(m_pParam->mReturnValue))
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
    int iNodeType = pQNode->getNodeType();
    QNodeAgency * pTempAgency = nullptr;
    if (GATE_NODE == iNodeType)
    {
        vector<Qubit *> controlVector;
        pTempAgency = new QNodeAgency((QGate *)pQNode, m_pGates, false, controlVector);
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QCircuit *)pQNode, m_pParam, m_pGates);
    }
    else if (PROG_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QProg *)pQNode, m_pParam, m_pGates);
    }
    else if (MEASURE_GATE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QMeasure *)pQNode, m_pParam, m_pGates);
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QWhileProg *)pQNode, m_pParam, m_pGates);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QIfProg *)pQNode, m_pParam, m_pGates);
    }
    else
    {
        throw exception();
    }

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
        int iNodeType = pQNode->getNodeType();
        QNodeAgency * pTempAgency = nullptr;
        if (GATE_NODE == iNodeType)
        {
            vector<Qubit *> controlVector;
            pTempAgency = new QNodeAgency((QGate *)pQNode, m_pGates, false, controlVector);
        }
        else if (CIRCUIT_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QCircuit *)pQNode, m_pParam, m_pGates);
        }
        else if (PROG_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QProg *)pQNode, m_pParam, m_pGates);
        }
        else if (MEASURE_GATE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QMeasure *)pQNode, m_pParam, m_pGates);
        }
        else if (WHILE_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QWhileProg *)pQNode, m_pParam, m_pGates);
        }
        else if (QIF_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QIfProg *)pQNode, m_pParam, m_pGates);
        }
        else
        {
            throw exception();
        }

        if (!pTempAgency->verify())
        {
            delete pTempAgency;
            return false;
        }
        delete pTempAgency;
    }

    if ((pQNode = pQIfNode->getFalseBranch()) != nullptr)
    {
        int iNodeType = pQNode->getNodeType();
        QNodeAgency * pTempAgency = nullptr;
        if (GATE_NODE == iNodeType)
        {
            vector<Qubit *> controlVector;
            pTempAgency = new QNodeAgency((QGate *)pQNode, m_pGates, false, controlVector);
        }
        else if (CIRCUIT_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QCircuit *)pQNode, m_pParam, m_pGates);
        }
        else if (PROG_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QProg *)pQNode, m_pParam, m_pGates);
        }
        else if (MEASURE_GATE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QMeasure *)pQNode, m_pParam, m_pGates);
        }
        else if (WHILE_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QWhileProg *)pQNode, m_pParam, m_pGates);
        }
        else if (QIF_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QIfProg *)pQNode, m_pParam, m_pGates);
        }
        else
        {
            throw exception();
        }

        if (!pTempAgency->verify())
        {
            delete pTempAgency;
            return false;
        }
        delete pTempAgency;
    }

    return true;
}

QWhileParse::QWhileParse(QWhileProg * pNode, QuantumGateParam * pParam, QuantumGates * pGates) :m_pNode(pNode), m_pGates(pGates), m_pParam(pParam)
{
}

bool QWhileParse::executeAction()
{
    AbstractControlFlowNode * pQWhileNode = dynamic_cast<AbstractControlFlowNode*>(m_pNode);
    auto aCExpr = pQWhileNode->getCExpr();
    QNode * pQNode;
    while (aCExpr->eval(m_pParam->mReturnValue))
    {
        pQNode = pQWhileNode->getTrueBranch();
        if (pQNode == nullptr)
        {
            return true;
        }

        int iNodeType = pQNode->getNodeType();
        QNodeAgency * pTempAgency = nullptr;
        if (GATE_NODE == iNodeType)
        {
            vector<Qubit *> controlVector;
            pTempAgency = new QNodeAgency((QGate *)pQNode, m_pGates, false, controlVector);
        }
        else if (CIRCUIT_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QCircuit *)pQNode, m_pParam, m_pGates);
        }
        else if (PROG_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QProg *)pQNode, m_pParam, m_pGates);
        }
        else if (MEASURE_GATE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QMeasure *)pQNode, m_pParam, m_pGates);
        }
        else if (WHILE_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QWhileProg *)pQNode, m_pParam, m_pGates);
        }
        else if (QIF_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QIfProg *)pQNode, m_pParam, m_pGates);
        }
        else
        {
            throw exception();
        }
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
    QNodeAgency * pTempAgency = nullptr;
    if (GATE_NODE == iNodeType)
    {
        vector<Qubit *> controlVector;
        pTempAgency = new QNodeAgency((QGate *)pQNode, m_pGates, false, controlVector);
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QCircuit *)pQNode, m_pParam, m_pGates);
    }
    else if (PROG_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QProg *)pQNode, m_pParam, m_pGates);
    }
    else if (MEASURE_GATE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QMeasure *)pQNode, m_pParam, m_pGates);
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QWhileProg *)pQNode, m_pParam, m_pGates);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        pTempAgency = new QNodeAgency((QIfProg *)pQNode, m_pParam, m_pGates);
    }
    else
    {
        throw exception();
    }
    if (!pTempAgency->verify())
    {
        delete pTempAgency;
        return false;
    }

    delete pTempAgency;

    return true;
}

MeasureParse::MeasureParse(QMeasure * pNode, QuantumGateParam * pParam, QuantumGates * pGates) :m_pNode(pNode), m_pGates(pGates), m_pParam(pParam)
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
    CBit * pCExpr =  pMeasureParse->getCBit();
    string sName = pCExpr->getName();
    auto aiter = m_pParam->mReturnValue.find(sName);
    if (aiter != m_pParam->mReturnValue.end())
    {
        aiter->second = (bool)iResult;
    }
    else
    {
        m_pParam->mReturnValue.insert(pair<string , bool>(sName, (bool)iResult));
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

QProgParse::QProgParse(QProg * pNode, QuantumGateParam * pParam, QuantumGates * pGates) :m_pNode(pNode), m_pGates(pGates), m_pParam(pParam)
{

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
        QNode * pNode = *aiter;
        if (nullptr == pNode)
        {
            return false;
        }

        int iNodeType = pNode->getNodeType();
        QNodeAgency * pTempAgency = nullptr;
        if (GATE_NODE == iNodeType)
        {
            vector<Qubit *> controlVector;
            pTempAgency = new QNodeAgency((QGate *)pNode, m_pGates, false, controlVector);
        }
        else if (CIRCUIT_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QCircuit *)pNode, m_pParam, m_pGates);
        }
        else if(PROG_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QProg *)pNode, m_pParam, m_pGates);
        }
        else if (MEASURE_GATE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QMeasure *)pNode, m_pParam, m_pGates);
        }
        else if (WHILE_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QWhileProg *)pNode, m_pParam, m_pGates);
        }
        else if (QIF_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QIfProg *)pNode, m_pParam, m_pGates);
        }
        else
        {
            throw exception();
        }

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
        QNode * pNode = *aiter;
        if (nullptr == pNode)
        {
            return true;
        }

        int iNodeType = pNode->getNodeType();
         
        QNodeAgency * pTempAgency = nullptr;
        if (GATE_NODE == iNodeType)
        {
            vector<Qubit *> controlVector;
            pTempAgency = new QNodeAgency((QGate *)pNode, m_pGates, false, controlVector);
        }
        else if (CIRCUIT_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QCircuit *)pNode, m_pParam, m_pGates);
        }
        else if (PROG_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QProg *)pNode, m_pParam, m_pGates);
        }
        else if (MEASURE_GATE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QMeasure *)pNode, m_pParam, m_pGates);
        }
        else if (WHILE_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QWhileProg *)pNode, m_pParam, m_pGates);
        }
        else if (QIF_START_NODE == iNodeType)
        {
            pTempAgency = new QNodeAgency((QIfProg *)pNode, m_pParam, m_pGates);
        }
        else
        {
            throw exception();
        }

        if (!pTempAgency->verify())
        {
            delete pTempAgency;
            return false;
        }

        delete pTempAgency;
    }
    return true;
}


REGISTER_QGATE_PARSE(1, QGateParseOneBit);
REGISTER_QGATE_PARSE(2, QGateParseTwoBit);



