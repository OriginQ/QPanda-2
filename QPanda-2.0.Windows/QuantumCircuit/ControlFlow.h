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

#ifndef  _CONTROL_FLOW_H
#define  _CONTROL_FLOW_H
#include "QuantumMachine/ClassicalConditionInterface.h"
#include "QNode.h"


#endif // ! _CONTROL_FLOW_H

class AbstractControlFlowNode
{
public:
    virtual QNode * getTrueBranch() const = 0;
    virtual QNode * getFalseBranch() const = 0;
    virtual ClassicalCondition * getCExpr()  = 0;
    virtual ~AbstractControlFlowNode() {}
};

/*
*  QIfProg:  the start node of the IF circuit
*  ccCondition:  judgement
*  ptTrue:  the head pointer of the true circuit
*  ptFalse:  the head pointer of the false circuit
*  ifEnd:  the last pointer of the IF circuit
*
*/
class QIfProg : public QNode, public AbstractControlFlowNode
{
private:
    AbstractControlFlowNode * m_pControlFlow;
    int m_iPosition;
    QIfProg();
public:
    QIfProg(const QIfProg &);
    QIfProg(ClassicalCondition & ccCon, QNode* pTrueNode, QNode * pFalseNode);
    QIfProg(ClassicalCondition & ccCon, QNode *node);
    NodeType getNodeType() const;
    QNode * getTrueBranch() const;
    QNode * getFalseBranch() const;
    int getPosition() const;
    ClassicalCondition * getCExpr();
};


typedef AbstractControlFlowNode * (*CreateIfDoubleB)(ClassicalCondition & ccCon, QNode* pTrueNode, QNode * pFalseNode);
typedef AbstractControlFlowNode * (*CreateIfSingleB)(ClassicalCondition & ccCon, QNode* pTrueNode);
class QuantunIfFactory
{
public:
    void registClass(string name, CreateIfDoubleB method);
    void registClass(string name, CreateIfSingleB method);
    AbstractControlFlowNode * getQuantumIf(std::string &, ClassicalCondition & ccCon, QNode* pTrueNode, QNode * pFalseNode);
    AbstractControlFlowNode * getQuantumIf(std::string &, ClassicalCondition & ccCon, QNode* pTrueNode);

    static QuantunIfFactory & getInstance()
    {
        static QuantunIfFactory  s_Instance;
        return s_Instance;
    }
private:
    map<string, CreateIfSingleB> m_QIfSingleMap;
    map<string, CreateIfDoubleB> m_QIfDoubleMap;
    QuantunIfFactory() {};

};

class QuantumIfRegisterAction {
public:
    QuantumIfRegisterAction(string className, CreateIfDoubleB ptrCreateFn) {
        QuantunIfFactory::getInstance().registClass(className, ptrCreateFn);
    }

    QuantumIfRegisterAction(string className, CreateIfSingleB ptrCreateFn) {
        QuantunIfFactory::getInstance().registClass(className, ptrCreateFn);
    }

};

#define REGISTER_QIF(className)                                             \
    AbstractControlFlowNode* QifSingleCreator##className( ClassicalCondition & ccCon, QNode* pTrueNode){      \
        return new className(ccCon,pTrueNode);                    \
    }                                                                   \
    AbstractControlFlowNode* QifDoubleCreator##className( ClassicalCondition & ccCon, QNode* pTrueNode, QNode * pFalseNode){      \
        return new className(ccCon,pTrueNode,pFalseNode);                    \
    }                                                                   \
    QuantumIfRegisterAction g_qifCreatorDoubleRegister##className(                        \
        #className,(CreateIfDoubleB)QifDoubleCreator##className);   \
    QuantumIfRegisterAction g_qifCreatorSingleRegister##className(                        \
        #className,(CreateIfSingleB)QifSingleCreator##className)



class OriginIf : public QNode, public AbstractControlFlowNode
{
private:
    ClassicalCondition  m_CCondition;
    int iTrueNum;
    int iFalseNum;
    NodeType m_iNodeType;
public:
    OriginIf(ClassicalCondition &ccCon, QNode* pTrueNode, QNode * pFalseNode);
    OriginIf(ClassicalCondition & ccCon, QNode *node);
    NodeType getNodeType() const;
    QNode * getTrueBranch() const;
    QNode * getFalseBranch() const;
    int getPosition() const;
    ClassicalCondition * getCExpr();
};





/*
*  CreateIfProg:  create IF circuit
*  trueProg:  true circuit of the IF circuit.
*  falseProg is nullptr
*/
extern QIfProg CreateIfProg(
    ClassicalCondition &,
    QNode  *trueNode);
/*
*  CreateIfProg:  create IF circuit
*  trueProg:  true circuit of the IF circuit.
*  falseProg: flase circuit of the IF circuit.
*/

extern QIfProg CreateIfProg(
    ClassicalCondition &,
    QNode *trueNode,
    QNode *falseNode);


/*
*  QWhileProg:  the start node of the WHILE circuit
*  ccCondition:  judgement
*  whileTrue:  the head pointer of the true circuit
*  whileEnd:  the last pointer of the true circuit,
*             whileEnd->next = QWhileProg *,in overall circuit,WHILE circuit
*             is like a point.
*/

class QWhileProg : public QNode, public AbstractControlFlowNode
{
private:
    AbstractControlFlowNode * m_pControlFlow;
    int m_iPosition;

    QWhileProg();
public:

    QWhileProg(const QWhileProg &);
    QWhileProg(ClassicalCondition & ccCon, QNode * node);

    /*
    *  CreateWhileProg:  create  WHILE circuit
    *  trueProg:  true circuit of the WHILE circuit.
    */

    NodeType getNodeType() const;
    QNode * getTrueBranch() const;
    QNode * getFalseBranch() const;
    ClassicalCondition * getCExpr();
    int getPosition() const;
};

class OriginWhile :public QNode, public AbstractControlFlowNode
{
private :
    NodeType m_iNodeType;
    ClassicalCondition  m_CCondition;
    int iTrueNum;
    OriginWhile();
public :
    OriginWhile(ClassicalCondition & ccCon, QNode * node);
    NodeType getNodeType() const;
    QNode * getTrueBranch() const;
    QNode * getFalseBranch() const;
    ClassicalCondition * getCExpr();
    int getPosition() const;
};



typedef AbstractControlFlowNode * (*CreateWhile)(ClassicalCondition & ccCon, QNode* pTrueNode);
class QuantunWhileFactory
{
public:

    void registClass(string name, CreateWhile method);
    AbstractControlFlowNode * getQuantumWhile(std::string &, ClassicalCondition & ccCon, QNode* pTrueNode);

    static QuantunWhileFactory & getInstance()
    {
        static QuantunWhileFactory  s_Instance;
        return s_Instance;
    }
private:
    map<string, CreateWhile> m_QWhileMap;
    QuantunWhileFactory() {};

};

class QuantumWhileRegisterAction {
public:
    QuantumWhileRegisterAction(string className, CreateWhile ptrCreateFn) {
        QuantunWhileFactory::getInstance().registClass(className, ptrCreateFn);
    }

};

#define REGISTER_QWHILE(className)                                             \
    AbstractControlFlowNode* QWhileCreator##className( ClassicalCondition & ccCon, QNode* pTrueNode){      \
        return new className(ccCon,pTrueNode);                    \
    }                                                                   \
    QuantumWhileRegisterAction g_qWhileCreatorDoubleRegister##className(                        \
        #className,(CreateWhile)QWhileCreator##className)





extern QWhileProg CreateWhileProg(
    ClassicalCondition &,
    QNode* trueNode);
