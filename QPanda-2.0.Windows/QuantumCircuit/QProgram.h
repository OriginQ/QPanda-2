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
//#include "../QuantumMachine/OriginQuantumMachine.h"
#include "../QuantumCircuit/QGlobalVariable.h"
#include "QGate.h"
#include "ReadWriteLock.h"
#include "../QuantumMachine/ClassicalConditionInterface.h"
#include "../QuantumMachine/OriginClassicalExpression.h"
typedef complex<double> QComplex;
using namespace std;
class QuantumDriver;

 /*
class OriginNode
{
private:
    int m_iReference;
    QNode * m_pQNode;
public:
    inline OriginNode() : m_iReference(0), m_pQNode(nullptr)
    {
        
    }

    inline OriginNode()
    {
        
    }
    template<typename T>
    bool creatNode()
    {
        this->m_iReference++;
        m_pQNode = new 
    }

    ~OriginNode()
    {
        if (0 != this->m_iReference)
        {
            this->m_iReference--;
        }
        else
        {
            delete m_pQNode;
        }
    }

    inline OriginNode(OriginNode * old)
    {
        this->m_pQNode = old->getQNode();
    }

    inline NodeType getNodeType() const
    {
        return this->m_pQNode->getNodeType();
    }

    void operator ++ ()
    {
        this->m_iReference++;
    }

    inline QNode * getQNode()
    {
        return this->m_pQNode;
    }

    inline bool setQNode(QNode *)
    {
    }

};
*/



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
};

class QNode
{
public:
    virtual NodeType getNodeType() const = 0;
    virtual int getPosition() const = 0;
    virtual ~QNode() {};
};



class QMeasureNode : public QNode
{
private:
    Qubit  *targetQuBit;
    CBit   *targetCbit;
    NodeType m_iNodeType;  
    GateType m_iGateType;
    int iPosition;
    QMeasureNode();
    QMeasureNode(QMeasureNode &);
    QMeasureNode(Qubit *, CBit *);
public:
    
    //QMeasureNode(QMeasureNode*);
    friend QMeasureNode& Measure(Qubit * targetQuBit, CBit * targetCbit);
    NodeType getNodeType() const;
    Qubit * getQuBit() const;
    CBit * getCBit()const;
    int getQuBitNum() const;

    int getPosition() const;
};
 
/*
*  Quantum single gate node: RX,RY,RZ,H,S,      CAN ADD OTHER GATES
*  gate:  gate type
*  opQuBit: qubit number
*  
*/
class QGateNodeFactory;
class OriginQGateNode : public QNode, public QGateNode
{
private:
	vector<Qubit *> m_QuBitVector;
	QGate *m_pGate;
    NodeType m_iNodeType;
    GateType m_iGateType;
    bool m_bIsDagger;
    vector<Qubit *> m_controlQuBitVector;
    OriginQGateNode();
    OriginQGateNode(OriginQGateNode&);

public:
    int iPosition;

    OriginQGateNode(Qubit*, QGate *);
    OriginQGateNode(Qubit*, Qubit *, QGate *);
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

class OriginQCircuit : public QCircuit, public QNode
{
private:
    Item * m_pHead;
    Item * m_pEnd;
    SharedMutex m_sm;
    NodeType m_iNodeType;
    bool m_bIsDagger;
    int iPosition;
    vector<Qubit *> m_controlQuBitVector;
    OriginQCircuit(OriginQCircuit &);
    OriginQCircuit() : m_iNodeType(CIRCUIT_NODE), iPosition(-1),m_pHead(nullptr),m_pEnd(nullptr)
    {
    };
public:

    ~OriginQCircuit();
    void pushBackNode(QNode *);
    friend OriginQCircuit & CreateEmptyCircuit();
    OriginQCircuit & operator << (const OriginQGateNode &);
    OriginQCircuit & operator << (const QMeasureNode &);
    OriginQCircuit & dagger();
    OriginQCircuit & control(vector<Qubit *> &);
    NodeType getNodeType() const;
    bool isDagger() const;
    bool getControlVector(vector<Qubit *> &);
    NodeIter  getFirstNodeIter() ;
    NodeIter  getLastNodeIter() ;
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    int getPosition() const;
};
class QControlFlowNode
{
public:
    virtual QNode * getTrueBranch() const = 0;
    virtual QNode * getFalseBranch() const = 0;
    virtual ClassicalCondition * getCExpr() const= 0;
};

/*
*  QIfNode:  the start node of the IF circuit
*  ccCondition:  judgement
*  ptTrue:  the head pointer of the true circuit
*  ptFalse:  the head pointer of the false circuit
*  ifEnd:  the last pointer of the IF circuit
*
*/
class QIfNode : public QNode, public QControlFlowNode
{
private:
    ClassicalCondition * ccCondition;
    int iTrueNum;
    int iFalseNum;
    int iPosition;
    NodeType m_iNodeType;
    QIfNode();
    QIfNode(QIfNode &);
    QIfNode(
        ClassicalCondition * ccCon,
        QNode* pTrueNode,
        QNode * pFalseNode) : m_iNodeType(QIF_START_NODE)
    {
        this->ccCondition = ccCon;
        this->iTrueNum = pTrueNode->getPosition();
        this->iFalseNum = pFalseNode->getPosition();
    }
    QIfNode(
        ClassicalCondition * ccCon,
        QNode *node
        ) : m_iNodeType(QIF_START_NODE)
    {
        this->ccCondition = ccCon;
        this->iTrueNum = node->getPosition();
        this->iFalseNum = -1;
    }
public:

    /*
    *  CreateIfProg:  create IF circuit
    *  trueProg:  true circuit of the IF circuit.
    *  falseProg is nullptr
    */
    friend QIfNode &CreateIfProg(
        ClassicalCondition *,
        QNode  *trueNode);
    /*
    *  CreateIfProg:  create IF circuit
    *  trueProg:  true circuit of the IF circuit.
    *  falseProg: flase circuit of the IF circuit.
    */

    friend QIfNode &CreateIfProg(
        ClassicalCondition *,
        QNode *trueNode,
        QNode *falseNode);

    NodeType getNodeType() const;
    QNode * getTrueBranch() const;
    QNode * getFalseBranch() const;
    int getPosition() const;
    ClassicalCondition * getCExpr()const;
    
};


/*
*  QWhileNode:  the start node of the WHILE circuit
*  ccCondition:  judgement
*  whileTrue:  the head pointer of the true circuit
*  whileEnd:  the last pointer of the true circuit,
*             whileEnd->next = QWhileNode *,in overall circuit,WHILE circuit
*             is like a point.
*/

class QWhileNode : public QNode, public QControlFlowNode
{
private:
    NodeType m_iNodeType;
    ClassicalCondition * ccCondition;
    int iTrueNum;
    int iPosition;
    QWhileNode() {};
    QWhileNode(QWhileNode &);
    QWhileNode(ClassicalCondition * ccCon, QNode * node) : m_iNodeType(WHILE_START_NODE), iPosition(-1)
    {
        this->ccCondition = ccCon;
        this->iTrueNum = node->getPosition();
    };
public:
    /*
     *  CreateWhileProg:  create  WHILE circuit
     *  trueProg:  true circuit of the WHILE circuit.
     */
    friend QWhileNode &CreateWhileProg(
        ClassicalCondition *,
        QNode* trueNode);
    NodeType getNodeType() const;
    QNode * getTrueBranch() const;
    QNode * getFalseBranch() const ;
    ClassicalCondition * getCExpr()const;
    int getPosition() const;
};

/*
*  QProg:  quantum program,can construct quantum circuit,data struct is linked list
*  QListHeadNode:  QProg's head pointer.
*  QListLastNode:  QProg's last pointer.
*  QProg & operator<<(const T &)：
*    if T is QSingleGateNode/QDoubleGateNode/QIfEndNode,
*    deep copy T and insert it into left QProg;
*    if T is QIfNode/QWhileNode/QProg,deepcopy 
*    IF/WHILE/QProg circuit and insert it into left QProg;
*/



class QProg : public QNode ,public QCircuit
{
private:
    Item * m_pHead;
    Item * m_pEnd;
    SharedMutex m_sm;
    NodeType m_iNodeType;
    int iPosition;
    QProg(): m_iNodeType(PROG_NODE), iPosition(-1), m_pHead(nullptr), m_pEnd(nullptr)
    {
    }
    QProg(QProg&);
public:
    ~QProg();
    void pushBackNode(QNode *);
    friend QProg & CreateEmptyQProg();
    QProg & operator << (const QIfNode &);
    QProg & operator << (const QWhileNode &);
    QProg & operator << (const QMeasureNode &);
    QProg & operator << (const QProg &);
    QProg & operator << (const OriginQGateNode &);
    QProg & operator << (const OriginQCircuit &);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeType getNodeType() const;
    void clear();
    int getPosition() const;
};

typedef void *RawData;


class QNodeVector
{
private:
    SharedMutex m_sm;
    vector<QNode*> m_pQNodeVector;
    vector<QNode*>::iterator m_currentIter;
public:
    QNodeVector();
    ~QNodeVector();

    bool pushBackNode(QNode *);
    size_t getLastNode();
    bool setHeadNode(QProg &);
    
    vector <QNode *>::iterator getNode(int);
    vector <QNode *>::iterator getEnd();
};

extern QNodeVector _G_QNodeVector;


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

 extern QProg & CreateEmptyQProg();
 extern QWhileNode &CreateWhileProg(
     ClassicalCondition *,
     QNode * trueNode);

 extern QIfNode &CreateIfProg(
     ClassicalCondition *,
     QNode *trueNode);

 extern QIfNode &CreateIfProg(
     ClassicalCondition *,
     QNode *trueNode,
     QNode *falseNode);

 extern OriginQCircuit & CreateEmptyCircuit();

 extern QMeasureNode& Measure(Qubit * targetQuBit, CBit * targetCbit);
 class QGateNodeFactory
 {
 public:
     static QGateNodeFactory * getInstance()
     {
         static QGateNodeFactory s_gateNodeFactory;
         return &s_gateNodeFactory;
     }

     OriginQGateNode & getGateNode(string & name,Qubit *);
     OriginQGateNode & getGateNode(string & name, Qubit *,double);
     OriginQGateNode & getGateNode(string & name, Qubit *, Qubit*);
     OriginQGateNode & getGateNode(double alpha, double beta, double gamma, double delta, Qubit *);
     OriginQGateNode & getGateNode(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);

 private:
     QGateNodeFactory()
     {
         m_pGateFact = QGateFactory::getInstance();
     }
     QGateFactory * m_pGateFact;
     
 };


#endif
