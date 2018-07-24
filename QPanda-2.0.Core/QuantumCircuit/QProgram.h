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
#include "QPanda/QPandaException.h"

typedef complex<double> QComplex;
using namespace std;
using QGATE_SPACE::QuantumGate;
using QGATE_SPACE::QGateFactory;
class QuantumDriver;

/*
*  Quantum gates node:QSingleGateNode and QDoubleGateNode
*
*
*/
class QNodeMap;
extern  QNodeMap _G_QNodeMap;


class AbstractQGateNode
{
public:
    virtual size_t getQuBitVector(vector<Qubit *> &) const = 0;
    virtual Qubit * popBackQuBit() = 0;
    virtual void PushBackQuBit(Qubit *) = 0;
    virtual size_t getQuBitNum() const = 0;
    virtual QuantumGate * getQGate() const = 0;
    virtual void setQGate(QuantumGate *) = 0;
    virtual bool isDagger() const = 0;
    virtual size_t getControlVector(vector<Qubit *> &) const = 0;
    virtual bool setDagger(bool) = 0;
    virtual bool setControl(vector<Qubit *> &) = 0;
    virtual ~AbstractQGateNode() {}
};

/*
*  Quantum single gate node: RX_GATE,RY_GATE,RZ_GATE,H,S_GATE,      CAN ADD OTHER GATES
*  gate:  gate type
*  opQuBit: qubit number
*
*/
class QGateNodeFactory;

class QGate : public QNode, public AbstractQGateNode
{
private:

    AbstractQGateNode * m_pQGateNode;
    QMAP_SIZE m_stPosition;
public:
    ~QGate();
    QGate(const QGate&);
    QGate(Qubit*, QuantumGate *);
    QGate(Qubit*, Qubit *, QuantumGate *);
    NodeType getNodeType() const;
    size_t getQuBitVector(vector<Qubit *> &) const;
    size_t getQuBitNum() const;
    QuantumGate * getQGate() const;
    QMAP_SIZE getPosition() const;
    bool setDagger(bool);
    bool setControl(vector < Qubit *> &);

    QGate dagger();
    QGate control(vector<Qubit *> &);

    bool isDagger() const;
    size_t getControlVector(vector<Qubit *> &) const;
private:
    void setPosition(QMAP_SIZE) {};
    Qubit * popBackQuBit() { return nullptr; };
    void setQGate(QuantumGate *) {};
    void PushBackQuBit(Qubit *) {};

};


class OriginQGate : public QNode, public AbstractQGateNode
{
private:
    vector<Qubit *> m_QuBitVector;
    QuantumGate *m_pQGate;
    NodeType m_iNodeType;
    bool m_bIsDagger;
    vector<Qubit *> m_controlQuBitVector;
    QMAP_SIZE m_stPosition;
public:
    ~OriginQGate();
    OriginQGate(Qubit*, QuantumGate *);
    OriginQGate(Qubit*, Qubit *, QuantumGate *);
    NodeType getNodeType() const;
    size_t getQuBitVector(vector<Qubit *> &) const;
    size_t getQuBitNum() const;
    Qubit * popBackQuBit();
    QuantumGate * getQGate() const;
    QMAP_SIZE getPosition() const;
    void setQGate(QuantumGate *);
    void setPosition(QMAP_SIZE stPosition);
    bool setDagger(bool);
    bool setControl(vector < Qubit *> &);
    bool isDagger() const;
    size_t getControlVector(vector<Qubit *> &) const;
    void PushBackQuBit(Qubit *);

};



class NodeIter
{
private:
    Item * m_pCur;
public:
    NodeIter(Item * pItem)
    {
        m_pCur = pItem;
    }

    NodeIter(const NodeIter & oldIter)
    {
        this->m_pCur = oldIter.getPCur();
    }

    Item * getPCur() const
    {
        return this->m_pCur;
    }

    void setPCur(Item * pItem)
    {
        m_pCur = pItem;
    }
    NodeIter()
    {
        m_pCur = nullptr;
    }
    /*
    Item * getItem() const
    {
    return m_pCur;
    }
    */
    NodeIter & operator ++();
    NodeIter  operator ++(int);
    QNode * operator *();
    NodeIter & operator --();
    NodeIter  operator --(int);
    NodeIter getNextIter();
    bool operator != (NodeIter);
    bool operator  == (NodeIter);
};

class AbstractQuantumProgram
{
public:
    virtual NodeIter  getFirstNodeIter() = 0;
    virtual NodeIter  getLastNodeIter() = 0;
    virtual NodeIter  getEndNodeIter() = 0;
    virtual NodeIter  getHeadNodeIter() = 0;
    virtual NodeIter  insertQNode(NodeIter &, QNode *) = 0;
    virtual NodeIter  deleteQNode(NodeIter &) = 0;
    virtual void pushBackNode(QNode *) = 0;
    virtual ~AbstractQuantumProgram() {};
    virtual void clear() = 0;
};

//class QCircuit;
class AbstractQuantumCircuit
{
public:
    virtual NodeIter  getFirstNodeIter() = 0;
    virtual NodeIter  getLastNodeIter() = 0;
    virtual NodeIter  getEndNodeIter() = 0;
    virtual NodeIter  getHeadNodeIter() = 0;
    virtual NodeIter  insertQNode(NodeIter &, QNode *) = 0;
    virtual NodeIter  deleteQNode(NodeIter &) = 0;
    virtual void pushBackNode(QNode *) = 0;
    virtual ~AbstractQuantumCircuit() {};
    virtual bool isDagger() const = 0;
    virtual bool getControlVector(vector<Qubit *> &) = 0;
    virtual void  setDagger(bool isDagger) = 0;
    virtual void  setControl(vector<Qubit *> &) = 0;
    virtual void clearControl() = 0;
};



class QCircuit : public QNode, public AbstractQuantumCircuit
{
protected:
    AbstractQuantumCircuit * m_pQuantumCircuit;
    QMAP_SIZE m_stPosition;
public:
    QCircuit();
    QCircuit(const QCircuit &);
    ~QCircuit();
    void pushBackNode(QNode *);

    /*
    QCircuit & operator << (QGate);
    QCircuit & operator << (QCircuit);
    QCircuit & operator << ( QMeasure );
    */

    template<typename T>
    QCircuit & operator <<(T);
    virtual QCircuit  dagger();
    virtual QCircuit  control(vector<Qubit *> &);
    NodeType getNodeType() const;
    bool isDagger() const;
    bool getControlVector(vector<Qubit *> &);
    NodeIter  getFirstNodeIter();
    NodeIter  getLastNodeIter();
    NodeIter  getEndNodeIter();
    NodeIter  getHeadNodeIter();

    NodeIter  insertQNode(NodeIter & iter, QNode * pNode);
    NodeIter  deleteQNode(NodeIter & iter);

    virtual void  setDagger(bool isDagger);
    virtual void  setControl(vector<Qubit *> &);
    QMAP_SIZE getPosition() const;
private:
    void setPosition(QMAP_SIZE stPosition);
    void clearControl() {};
};


class HadamardQCircuit :public QCircuit
{
public:
    HadamardQCircuit(vector<Qubit * > & pQubitVector);
    ~HadamardQCircuit() {};
private:
    HadamardQCircuit();

};


typedef AbstractQuantumCircuit * (*CreateQCircuit)();
class QuantumCircuitFactory
{
public:

    void registClass(string name, CreateQCircuit method);
    AbstractQuantumCircuit * getQuantumCircuit(std::string &);

    static QuantumCircuitFactory & getInstance()
    {
        static QuantumCircuitFactory  s_Instance;
        return s_Instance;
    }
private:
    map<string, CreateQCircuit> m_QCirciutMap;
    QuantumCircuitFactory() {};

};

class QuantumCircuitRegisterAction {
public:
    QuantumCircuitRegisterAction(string className, CreateQCircuit ptrCreateFn) {
        QuantumCircuitFactory::getInstance().registClass(className, ptrCreateFn);
    }

};

#define REGISTER_QCIRCUIT(className)                                             \
    AbstractQuantumCircuit* QWhileCreator##className(){      \
        return new className();                    \
    }                                                                   \
    QuantumCircuitRegisterAction g_qWhileCreatorDoubleRegister##className(                        \
        #className,(CreateQCircuit)QWhileCreator##className)


class OriginCircuit : public QNode, public AbstractQuantumCircuit
{
private:
    Item * m_pHead;
    Item * m_pEnd;
    SharedMutex m_sm;
    NodeType m_iNodeType;
    bool m_bIsDagger;
    QMAP_SIZE m_stPosition;
    vector<Qubit *> m_controlQuBitVector;
    OriginCircuit(QCircuit &);

public:
    OriginCircuit() : m_iNodeType(CIRCUIT_NODE), m_pHead(nullptr), m_pEnd(nullptr), m_bIsDagger(false), m_stPosition(999999)
    {
        m_controlQuBitVector.resize(0);
    };
    ~OriginCircuit();
    void pushBackNode(QNode *);

    void setDagger(bool);
    void setControl(vector<Qubit *> &);
    NodeType getNodeType() const;
    bool isDagger() const;
    bool getControlVector(vector<Qubit *> &);
    NodeIter  getFirstNodeIter();
    NodeIter  getLastNodeIter();
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeIter  insertQNode(NodeIter &, QNode *);
    NodeIter  deleteQNode(NodeIter &);
    QMAP_SIZE getPosition() const;
    void setPosition(QMAP_SIZE);
    void clearControl();

};


/*
*  QProg:  quantum program,can construct quantum circuit,data struct is linked list
*  QListHeadNode:  QProg's head pointer.
*  QListLastNode:  QProg's last pointer.
*  QProg & operator<<(const T_GATE &)：
*    if T_GATE is QSingleGateNode/QDoubleGateNode/QIfEndNode,
*    deep copy T_GATE and insert it into left QProg;
*    if T_GATE is QIfProg/QWhileProg/QProg,deepcopy
*    IF/WHILE/QProg circuit and insert it into left QProg;
*/
class QProg : public QNode, public AbstractQuantumProgram
{
private:
    AbstractQuantumProgram * m_pQuantumProgram;
    QMAP_SIZE m_stPosition;
public:
    QProg();
    QProg(const QProg&);
    ~QProg();
    void pushBackNode(QNode *);
    /*
    QProg & operator << ( QIfProg );
    QProg & operator << ( QWhileProg );
    QProg & operator << (QMeasure );
    QProg & operator << ( QProg );
    QProg & operator << ( QGate );
    QProg & operator << ( QCircuit );

    */
    template<typename T>
    QProg & operator <<(T);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeIter  insertQNode(NodeIter & iter, QNode * pNode);
    NodeIter  deleteQNode(NodeIter & iter);
    NodeType getNodeType() const;
    void clear();
    QMAP_SIZE getPosition() const;
private:
    void setPosition(QMAP_SIZE) {};
};

typedef AbstractQuantumProgram * (*CreateQProgram)();
class QuantumProgramFactory
{
public:

    void registClass(string name, CreateQProgram method);
    AbstractQuantumProgram * getQuantumCircuit(std::string &);

    static QuantumProgramFactory & getInstance()
    {
        static QuantumProgramFactory  s_Instance;
        return s_Instance;
    }
private:
    map<string, CreateQProgram> m_QProgMap;
    QuantumProgramFactory() {};

};

class QuantumProgramRegisterAction {
public:
    QuantumProgramRegisterAction(string className, CreateQProgram ptrCreateFn) {
        QuantumProgramFactory::getInstance().registClass(className, ptrCreateFn);
    }

};

#define REGISTER_QPROGRAM(className)                                             \
    AbstractQuantumProgram* QProgCreator##className(){      \
        return new className();                    \
    }                                                                   \
    QuantumProgramRegisterAction g_qProgCreatorDoubleRegister##className(                        \
        #className,(CreateQProgram)QProgCreator##className)




class OriginProgram :public QNode, public AbstractQuantumProgram
{
private:
    Item * m_pHead;
    Item * m_pEnd;
    SharedMutex m_sm;
    NodeType m_iNodeType;
    OriginProgram(OriginProgram&);
    QMAP_SIZE m_stPosition;
public:
    ~OriginProgram();
    OriginProgram();
    void pushBackNode(QNode *);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeIter  insertQNode(NodeIter &, QNode *);
    NodeIter  deleteQNode(NodeIter &);
    NodeType getNodeType() const;
    void clear();
    QMAP_SIZE getPosition() const;
    void setPosition(QMAP_SIZE);
};



extern QProg  CreateEmptyQProg();


extern QCircuit  CreateEmptyCircuit();


class QGateNodeFactory
{
public:
    static QGateNodeFactory * getInstance()
    {
        static QGateNodeFactory s_gateNodeFactory;
        return &s_gateNodeFactory;
    }

    QGate  getGateNode(string & name, Qubit *);
    QGate  getGateNode(string & name, Qubit *, double);
    QGate  getGateNode(string & name, Qubit *, Qubit*);
    QGate  getGateNode(string & name, Qubit * controlQBit, Qubit * targetQBit, double theta);
    QGate  getGateNode(double alpha, double beta, double gamma, double delta, Qubit *);
    QGate  getGateNode(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);
    QGate  getGateNode(string &name, QStat matrix, Qubit *, Qubit *);
    QGate  getGateNode(string &name, QStat matrix, Qubit *);
private:
    QGateNodeFactory()
    {
        m_pGateFact = QGateFactory::getInstance();
    }
    QGateFactory * m_pGateFact;

};

template<typename T>
inline QProg & QProg::operator<<(T node)
{
    auto temp = dynamic_cast<QNode *>(&node);
    if (nullptr == temp)
    {
        throw param_error_exception("param is not QNode", false);
    }
    if (nullptr == this->m_pQuantumProgram)
    {
        throw exception();
    }
    int iNodeType = temp->getNodeType();
    m_pQuantumProgram->pushBackNode(dynamic_cast<QNode*>(&node));
    return *this;
}


template<typename T>
inline QCircuit & QCircuit::operator<<(T node)
{
    auto temp = dynamic_cast<QNode *>(&node);
    if (nullptr == temp)
    {
        throw param_error_exception("param is not QNode", false);
    }
    if (nullptr == this->m_pQuantumCircuit)
    {
        throw exception();
    }
    int iNodeType = temp->getNodeType();

    switch (iNodeType)
    {
    case GATE_NODE:
    case CIRCUIT_NODE:
    case MEASURE_GATE:
        m_pQuantumCircuit->pushBackNode(dynamic_cast<QNode*>(&node));
        break;
    default:
        throw param_error_exception("param node type error", false);
    }
    return *this;
}
#endif
