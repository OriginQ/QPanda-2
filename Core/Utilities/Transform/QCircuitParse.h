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

#ifndef _QCIRCUIT_PARSE_H
#define _QCIRCUIT_PARSE_H
#include <map>
#include <functional>
#include "VirtualQuantumProcessor/QuantumGates.h"
#include "QuantumCircuit/QProgram.h"
#include "QuantumCircuit/ClassicalProgam.h"
QPANDA_BEGIN


class QNodeAgency;
class AbstractQNodeParse
{
public:
    virtual bool executeAction() = 0;
    virtual bool verify() = 0;
    virtual ~AbstractQNodeParse() {};
private:
    virtual QNodeAgency * getAgency(QNode * pNode) = 0;
};

class ClassicalProgParse :public AbstractQNodeParse
{
private:
    AbstractClassicalProg * m_pNode;
public:
    ClassicalProgParse(AbstractClassicalProg * pNode);
    ~ClassicalProgParse() {};
    bool executeAction();
    bool verify();
private:
    ClassicalProgParse();
    QNodeAgency * getAgency(QNode * pNode);
};

class QCirCuitParse :public AbstractQNodeParse
{
private:
    AbstractQuantumCircuit * m_pNode;
    QuantumGateParam * m_pParam;
    QuantumGates * m_pGates;
    bool   m_bDagger;
    std::vector<Qubit *> m_controlQubitVector;
public:
    QCirCuitParse(AbstractQuantumCircuit * pNode,
                  QuantumGateParam * pParam, 
                  QuantumGates* pGates,
                  bool isDagger,
                  std::vector<Qubit *> controlQbitVector);
    ~QCirCuitParse() {};
    bool executeAction();
    bool verify();
private:
    QCirCuitParse();
    QNodeAgency * getAgency(QNode * pNode);
};

class QIfParse :public AbstractQNodeParse
{
private:
    AbstractControlFlowNode * m_pNode;
    QuantumGateParam * m_pParam;
    QuantumGates* m_pGates;
public:
    QIfParse(AbstractControlFlowNode * pNode, 
             QuantumGateParam * pParam,
             QuantumGates* pGates);
    ~QIfParse() {};
    bool executeAction();
    bool verify();
private:
    QNodeAgency * getAgency(QNode * pNode);
    QIfParse();
};
class QWhileProg;
class QWhileParse :public AbstractQNodeParse
{
private:
    AbstractControlFlowNode * m_pNode;
    QuantumGateParam * m_pParam;
    QuantumGates* m_pGates;
public:
    QWhileParse(AbstractControlFlowNode * pNode, 
                QuantumGateParam * pParam,
                QuantumGates* pGates);
    ~QWhileParse() {};
    bool executeAction();
    bool verify();
private:
    QNodeAgency * getAgency(QNode * pNode);
    QWhileParse();
};

class MeasureParse :public AbstractQNodeParse
{
private:
    AbstractQuantumMeasure * m_pNode;
    QuantumGateParam * m_pParam;
    QuantumGates* m_pGates;
public:
    MeasureParse(AbstractQuantumMeasure * pNode, QuantumGateParam * pParam, QuantumGates* pGates);
    ~MeasureParse() {};
    bool executeAction();
    bool verify();
private:
    QNodeAgency * getAgency(QNode * pNode) { return nullptr; };
    MeasureParse();
};


class QProgParse :public AbstractQNodeParse
{
private:
    AbstractQuantumProgram * m_pNode;
    QuantumGateParam * m_pParam;
    QuantumGates* m_pGates;
public:
    QProgParse(AbstractQuantumProgram * pNode, QuantumGateParam * pParam, QuantumGates* pGates);
    ~QProgParse() {};
    bool executeAction();
    bool verify();
    
private:
    QNodeAgency * getAgency(QNode * pNode);
    QProgParse();
};

class QGateParse :public AbstractQNodeParse
{
private:
    AbstractQGateNode * m_pNode;
    QuantumGates* m_pGates;
    bool m_isDagger;
    std::vector<Qubit *> m_controlQubitVector;
public:
    QGateParse(AbstractQGateNode * pNode, 
               QuantumGates* pGates,
               bool isDagger,
               std::vector<Qubit *> & controlQubitVector);
    ~QGateParse() {};
    bool executeAction();
    bool verify();
private:
    virtual QNodeAgency * getAgency(QNode * pNode) { return nullptr; };
    QGateParse();

};

class QNodeAgency :public AbstractQNodeParse
{
public:
    QNodeAgency(AbstractQuantumCircuit * pNode,
                QuantumGateParam * pParam, 
                QuantumGates * pGates,
                bool isDgger,
                std::vector<Qubit*> controlQubitVector);
    QNodeAgency(AbstractControlFlowNode * pNode,
                QuantumGateParam * pParam,
                QuantumGates* pGates);
    QNodeAgency(AbstractQuantumMeasure * pNode, 
                QuantumGateParam * pParam, 
                QuantumGates* pGates);
    QNodeAgency(AbstractQuantumProgram * pNode, 
                QuantumGateParam * pParam, 
                QuantumGates* pGates);
    QNodeAgency(AbstractClassicalProg * pNode);
    QNodeAgency(AbstractQGateNode * pNode, 
                QuantumGates* pGates, 
                bool isDagger, 
                std::vector<Qubit *> & controlQubitVector);

    ~QNodeAgency();
    bool executeAction();
    bool verify();
private:
    QNodeAgency();
    virtual QNodeAgency * getAgency(QNode * pNode) { return nullptr; };
    AbstractQNodeParse * m_pQNodeParse;

};


typedef void(*QGATE_FUN)(QuantumGate *, 
                         std::vector<Qubit * > &,
                         QuantumGates*,
                         bool,
                         std::vector<Qubit *> &);
typedef std::map<int, QGATE_FUN> QGATE_FUN_MAP;
class QGateParseMap
{

    static QGATE_FUN_MAP m_QGateFunctionMap;
public:

    static void insertMap(int iOpNum, QGATE_FUN pFunction)
    {
        m_QGateFunctionMap.insert(std::pair<int, QGATE_FUN>(iOpNum, pFunction));
    }

    static QGATE_FUN getFunction(int iOpNum)
    {
        auto aiter = m_QGateFunctionMap.find(iOpNum);
        if (aiter == m_QGateFunctionMap.end())
        {
            return nullptr;
        }

        return aiter->second;
    }


};



QPANDA_END

#endif // !_QCIRCUIT_PARSE_H
