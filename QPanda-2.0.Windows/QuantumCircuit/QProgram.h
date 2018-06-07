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

#ifndef _QPROGRAM_H_
#define _QPROGRAM_H_

#include <complex>
#include <initializer_list>
#include <vector>
#include <iterator>
#include <map>

#include "QuantumMachine/OriginClassicalExpression.h"
#include "QNode.h"
#include "QGate.h"
#include "ReadWriteLock.h"
#include "QuantumMeasure.h"
#include "ControlFlow.h"

typedef complex<double> QComplex;
using namespace std;
class QuantumDriver;

/*
*  Quantum gates node:QSingleGateNode and QDoubleGateNode
*
*
*/
class QNodeVector;
extern  QNodeVector _G_QNodeVector;
class NodeIter;

class QGateNode 
{
public:
    virtual size_t getQuBitVector( vector<Qubit *> & ) const = 0;
    virtual size_t getQuBitNum() const = 0;
    virtual QGate * getQGate() const = 0;
    virtual ~QGateNode() {}
};
 
/*
*  Quantum single gate node: RX,RY,RZ,H,S,      CAN ADD OTHER GATES
*  gate:  gate type
*  opQuBit: qubit number
*  
*/
class QGateNodeFactory;

class QuantumGate : public QNode, public QGateNode
{
private:
	vector<Qubit *> m_QuBitVector;
	QGate *m_pGate;
    NodeType m_iNodeType;
    GateType m_iGateType;
    bool m_bIsDagger;
    vector<Qubit *> m_controlQuBitVector;
    QuantumGate();
    QuantumGate(QuantumGate&);

public:
    int iPosition;

    QuantumGate(Qubit*, QGate *);
    QuantumGate(Qubit*, Qubit *, QGate *);
    NodeType getNodeType() const;
    size_t getQuBitVector(vector<Qubit *> &) const;
    size_t getQuBitNum() const;
    QGate * getQGate() const;
    GateType getQGateType() const;
    int getPosition() const;
    bool setDagger(bool);
    bool setControl(vector < Qubit *> &);
    bool isDagger() const;
    size_t getControlVector(vector<Qubit *> &) const;

};

class QCircuit
{
public:
    virtual NodeIter  getFirstNodeIter()  = 0;
    virtual NodeIter  getLastNodeIter()  = 0;
    virtual NodeIter  getEndNodeIter() = 0;
    virtual NodeIter getHeadNodeIter() = 0;
};

class  Item 
{
public:
    virtual Item * getNext()const =0;
    virtual Item * getPre() const=0;
    virtual QNode * getNode() const= 0;
    virtual void setNext(Item *) = 0;
    virtual void setPre(Item *) = 0;
    virtual void setNode(QNode *) = 0;
    virtual ~Item() {};
};

class  OriginItem : public Item
{
private:
    Item * m_pNext;
    Item * m_pPre;
    int    m_iNodeNum;
public:

    Item * getNext()const;
    Item * getPre()const;
    QNode * getNode() const;
    void setNext(Item * pItem);
    void setPre(Item * pItem);
    void setNode(QNode * pNode);
};

class QuantumCircuit : public QCircuit, public QNode
{
private:
    Item * m_pHead;
    Item * m_pEnd;
    SharedMutex m_sm;
    NodeType m_iNodeType;
    bool m_bIsDagger;
    int iPosition;
    vector<Qubit *> m_controlQuBitVector;
    QuantumCircuit(QuantumCircuit &);
    QuantumCircuit() : m_iNodeType(CIRCUIT_NODE), iPosition(-1),m_pHead(nullptr),m_pEnd(nullptr)
    {
    };
public:

    ~QuantumCircuit();
    void pushBackNode(QNode *);
    friend QuantumCircuit & CreateEmptyCircuit();
    QuantumCircuit & operator << (const QuantumGate &);
    QuantumCircuit & operator << (const QuantumMeasure &);
    QuantumCircuit & dagger();
    QuantumCircuit & control(vector<Qubit *> &);
    NodeType getNodeType() const;
    bool isDagger() const;
    bool getControlVector(vector<Qubit *> &);
    NodeIter  getFirstNodeIter() ;
    NodeIter  getLastNodeIter() ;
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    int getPosition() const;
};

/*
*  QuantumProgram:  quantum program,can construct quantum circuit,data struct is linked list
*  QListHeadNode:  QuantumProgram's head pointer.
*  QListLastNode:  QuantumProgram's last pointer.
*  QuantumProgram & operator<<(const T &)：
*    if T is QSingleGateNode/QDoubleGateNode/QIfEndNode,
*    deep copy T and insert it into left QuantumProgram;
*    if T is QuantumIf/QuantumWhile/QuantumProgram,deepcopy 
*    IF/WHILE/QuantumProgram circuit and insert it into left QuantumProgram;
*/
class QuantumProgram : public QNode ,public QCircuit
{
private:
    Item * m_pHead;
    Item * m_pEnd;
    SharedMutex m_sm;
    NodeType m_iNodeType;
    int iPosition;
    QuantumProgram(): m_iNodeType(PROG_NODE), iPosition(-1), m_pHead(nullptr), m_pEnd(nullptr)
    {
    }
    QuantumProgram(QuantumProgram&);
public:
    ~QuantumProgram();
    void pushBackNode(QNode *);
    friend QuantumProgram & CreateEmptyQProg();
    QuantumProgram & operator << (const QuantumIf &);
    QuantumProgram & operator << (const QuantumWhile &);
    QuantumProgram & operator << (const QuantumMeasure &);
    QuantumProgram & operator << (const QuantumProgram &);
    QuantumProgram & operator << (const QuantumGate &);
    QuantumProgram & operator << (const QuantumCircuit &);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeType getNodeType() const;
    void clear();
    int getPosition() const;
};




class NodeIter
{
private:
    Item * m_pCur ;
public:
    NodeIter(Item * pItem)
    {
        m_pCur = pItem;
    }

    NodeIter()
    {
        m_pCur = nullptr;
    }

    Item * getItem() const
    {
        return m_pCur;  
    }
    NodeIter & operator ++();
    QNode * operator *();
    NodeIter & operator --();
    bool operator != (NodeIter );
    bool operator  == (NodeIter );
 };

 extern QuantumProgram & CreateEmptyQProg();


 extern QuantumCircuit & CreateEmptyCircuit();


 class QGateNodeFactory
 {
 public:
     static QGateNodeFactory * getInstance()
     {
         static QGateNodeFactory s_gateNodeFactory;
         return &s_gateNodeFactory;
     }

     QuantumGate & getGateNode(string & name,Qubit *);
     QuantumGate & getGateNode(string & name, Qubit *,double);
     QuantumGate & getGateNode(string & name, Qubit *, Qubit*);
     QuantumGate & getGateNode(double alpha, double beta, double gamma, double delta, Qubit *);
     QuantumGate & getGateNode(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);

 private:
     QGateNodeFactory()
     {
         m_pGateFact = QGateFactory::getInstance();
     }
     QGateFactory * m_pGateFact;
     
 };


#endif
