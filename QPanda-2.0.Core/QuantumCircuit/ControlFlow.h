/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ControlFlow.h 
Author: Menghan.Dou
Created in 2018-6-30

Classes for ControlFlow 

Update@2018-8-30
  Update by code specification
*/


#ifndef  _CONTROL_FLOW_H
#define  _CONTROL_FLOW_H
#include "QuantumMachine/ClassicalConditionInterface.h"
#include "QNode.h"

/*
* Superclass for QIfProg/QWhileProg
*/
class AbstractControlFlowNode
{
public:
    /*
    Get true branch
    param :
    return : branch node
    Note:
    */
    virtual QNode * getTrueBranch() const = 0;

    /*
    Get false branch
    param :
    return : branch node
    Note:
    */
    virtual QNode * getFalseBranch() const = 0;

    /*
    Get true branch
    param : 
    node : true branch node
    return : 
    Note:
    */
    virtual void setTrueBranch(QNode * node) = 0;

    /*
    set false branch
    param : 
    node : false branch node
    return : 
    Note:
    */
    virtual void setFalseBranch(QNode * node) = 0;

    /*
    Get classical condition
    param :
    return : classical condition
    Note:
    */
    virtual ClassicalCondition * getCExpr() = 0;
    virtual ~AbstractControlFlowNode() {}
};

/*
*  QIfProg:  Proxy class of quantum if program
*/
class QIfProg : public QNode, public AbstractControlFlowNode
{
private:
    AbstractControlFlowNode * m_control_flow;
    qmap_size_t m_position;
    QIfProg();
public:
    ~QIfProg();
    QIfProg(const QIfProg &);
    /*
    QIfProg constructor
    param :
    classical_condition : this QIfProg classical condition
    true_node : true branch node
    false_node : false branch node
    return : 
    Note:
    */
    QIfProg(ClassicalCondition& classical_condition, QNode *true_node, QNode *false_node);

    /*
    QIfProg constructor
    param :
    classical_condition : this QIfProg classical condition
    node : true branch node
    return :
    Note:
    */
    QIfProg(ClassicalCondition &classical_condition, QNode * node);

    /*
    Get the current node type
    param :
    return : node type
    Note:
    */
    virtual NodeType getNodeType() const;

    /*
    Get true branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getTrueBranch() const;

    /*
    Get false branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getFalseBranch() const;

    /*
    Get this node position in global QNode map
    param :
    return : position
    Note:
    */
    virtual qmap_size_t getPosition() const;

    /*
    Get classical condition
    param :
    return : classical condition
    Note:
    */
    virtual ClassicalCondition *getCExpr() ;

private:
    virtual void setPosition(qmap_size_t){};
    virtual void setTrueBranch(QNode*){};
    virtual void setFalseBranch(QNode*){};
};

typedef AbstractControlFlowNode * (*CreateQIfTrueFalse_cb)(ClassicalCondition &, QNode *, QNode *);
typedef AbstractControlFlowNode * (*CreateQIfTrueOnly_cb)(ClassicalCondition &, QNode *);

/*
* QIfFactory: Factory for generating QIf.
*/
class QIfFactory
{
public:
    /*
    Regist QIf class
    param :
    name   : class name
    method : create qif true false callback
    return :
    Note:
    */
    void registClass(string name, CreateQIfTrueFalse_cb method);

    /*
    Regist QIf class
    param :
    name   : class name
    method : create qif true only callback
    return :
    Note:
    */
    void registClass(string name, CreateQIfTrueOnly_cb method);

    /*
    get QIf class
    param :
    class_name   : class name
    classical_condition : classical condition
    true_node : true branch node
    false_node : false branch node
    return :
    Note:
    */
    AbstractControlFlowNode* getQIf(std::string &class_name,
                                    ClassicalCondition &classical_condition, 
                                    QNode *true_node,
                                    QNode *false_node);

   /*
    Regist QIf class
    param :
    name   : class name
    method : create qif true only callback
    return :
    Note:
    */
    AbstractControlFlowNode * getQIf(std::string &, ClassicalCondition &, QNode *);

    /*
    Get qif factory instance
    param :
    return : qifs factory
    Note:
    */
    static QIfFactory & getInstance()
    {
        static QIfFactory  instance;
        return instance;
    }
private:
    map<string, CreateQIfTrueOnly_cb> m_qif_true_only_map;
    map<string, CreateQIfTrueFalse_cb> m_qif_true_false_map;
    QIfFactory() {};

};

class QIfRegisterAction {
public:
    QIfRegisterAction(string class_name, CreateQIfTrueFalse_cb create_callback) {
        QIfFactory::getInstance().registClass(class_name, create_callback);
    }

    QIfRegisterAction(string class_name, CreateQIfTrueOnly_cb create_callback) {
        QIfFactory::getInstance().registClass(class_name, create_callback);
    }

};

#define QIF_REGISTER(className)                                             \
    AbstractControlFlowNode* QifSingleCreator##className(ClassicalCondition& classical_condition, QNode* true_node)       \
    {      \
        return new className(classical_condition, true_node);                    \
    }                                                                   \
    AbstractControlFlowNode* QifDoubleCreator##className(ClassicalCondition& classical_condition, QNode* true_node, QNode* false_node) \
    {      \
        return new className(classical_condition, true_node, false_node);                    \
    }                                                                   \
    QIfRegisterAction _G_qif_creator_double_register##className(                        \
        #className,(CreateQIfTrueFalse_cb)QifDoubleCreator##className);   \
    QIfRegisterAction _G_qif_creator_single_register##className(                        \
        #className,(CreateQIfTrueOnly_cb)QifSingleCreator##className)


/*
* OriginQIf: Qif implementation class
*/
class OriginQIf : public QNode, public AbstractControlFlowNode
{
private:
    ClassicalCondition  m_classical_condition;
    Item * m_true_item;
    Item * m_false_item;
    NodeType m_node_type;
    qmap_size_t m_position;
public:
    ~OriginQIf();

    /*
    OriginQIf constructor
    param :
    classical_condition : OriginQIf classical condition
    true_node : true branch node
    false_node : false branch node
    return : 
    Note:
    */
    OriginQIf(ClassicalCondition & classical_condition, QNode *true_node, QNode *false_node);

    /*
    OriginQIf constructor
    param :
    classical_condition : OriginQIf classical condition
    node : true branch node
    return :
    Note:
    */
    OriginQIf(ClassicalCondition & classical_condition, QNode * node);

    /*
    Get the current node type
    param :
    return : node type
    Note:
    */
    virtual NodeType getNodeType() const;

    /*
    Get true branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getTrueBranch() const;

    /*
    Get false branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getFalseBranch() const;

    /*
    Get this node position in global QNode map
    param :
    return : position
    Note:
    */
    virtual qmap_size_t getPosition() const;

    /*
    Set true branch
    param : 
    node : true branch node
    return : 
    Note:
    */
    virtual void setTrueBranch(QNode * node);

    /*
    Set false branch
    param : 
    node : false branch node
    return :
    Note:
    */
    virtual void setFalseBranch(QNode * node);

    /*
    Set position 
    param : 
    position : this node position in global QNode map
    return : 
    Note:
    */
    virtual void setPosition(qmap_size_t position);

    /*
    Get classical condition
    param :
    return : classical condition
    Note:
    */
    virtual ClassicalCondition *getCExpr() ;
};

/*
Create qif prog 
param :
classical_condition : OriginQIf classical condition
true_node : true branch node
return : QIfProg
Note:
*/
extern QIfProg CreateIfProg(
    ClassicalCondition &classical_condition,
    QNode *true_node);

/*
Create qif prog
param :
classical_condition : OriginQIf classical condition
true_node : true branch node
false_node : false branch node
return : QIfProg
Note:
*/
extern QIfProg CreateIfProg(
    ClassicalCondition &classical_condition,
    QNode *true_node,
    QNode *false_node);

/*
*  QWhileProg:  Proxy class of quantum while program
*/
class QWhileProg : public QNode, public AbstractControlFlowNode
{
private:
    AbstractControlFlowNode * m_control_flow;
    qmap_size_t m_position;

    QWhileProg();
public:
    ~QWhileProg();
    QWhileProg(const QWhileProg &);
    QWhileProg(ClassicalCondition & , QNode *);

    /*
    Get the current node type
    param :
    return : node type
    Note:
    */
    virtual NodeType getNodeType() const;

    /*
    Get true branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getTrueBranch() const;

    /*
    Get false branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getFalseBranch() const;

    /*
    Get this node position in global QNode map
    param :
    return : position
    Note:
    */
    virtual qmap_size_t getPosition() const;

    /*
    Get classical condition
    param :
    return : classical condition
    Note:
    */
    virtual ClassicalCondition *getCExpr();

private:
    virtual void setPosition(qmap_size_t){};
    virtual void setTrueBranch(QNode*){};
    virtual void setFalseBranch(QNode*){};
};

/*
*  OriginQWhile: QWhile implementation class
*/
class OriginQWhile :public QNode, public AbstractControlFlowNode
{
private :
    NodeType m_node_type;
    ClassicalCondition  m_classical_condition;
    Item * m_true_item;
    qmap_size_t m_position;

    OriginQWhile();
public :
    ~OriginQWhile();
    OriginQWhile(ClassicalCondition & ccCon, QNode * node);
    /*
    Get the current node type
    param :
    return : node type
    Note:
    */
    virtual NodeType getNodeType() const;

    /*
    Get true branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getTrueBranch() const;

    /*
    Get false branch
    param :
    return : branch node
    Note:
    */
    virtual QNode* getFalseBranch() const;

    /*
    Get this node position in global QNode map
    param :
    return : position
    Note:
    */
    virtual qmap_size_t getPosition() const;

    /*
    Set true branch
    param : 
    node : true branch node
    return : 
    Note:
    */
    virtual void setTrueBranch(QNode * node);

    /*
    Set false branch
    param : 
    node : false branch node
    return :
    Note:
    */
    virtual void setFalseBranch(QNode * node) {};

    /*
    Set position 
    param : 
    position : this node position in global QNode map
    return : 
    Note:
    */
    virtual void setPosition(qmap_size_t position);

    /*
    Get classical condition
    param :
    return : classical condition
    Note:
    */
    virtual ClassicalCondition *getCExpr() ;
};

typedef AbstractControlFlowNode * (*CreateQWhile_cb)(ClassicalCondition &, QNode *);

/*
* QWhileFactory: Factory for generating QWhile.
*/
class QWhileFactory
{
public:
    void registClass(string name, CreateQWhile_cb method);
    AbstractControlFlowNode * getQWhile(std::string &, ClassicalCondition &, QNode *);
    static QWhileFactory & getInstance()
    {
        static QWhileFactory  instance;
        return instance;
    }

private:
    map<string, CreateQWhile_cb> m_qwhile_map;
    QWhileFactory() {};

};

class QWhileRegisterAction {
public:
    QWhileRegisterAction(string class_name, CreateQWhile_cb create_callback) {
        QWhileFactory::getInstance().registClass(class_name, create_callback);
    }
};

#define QWHILE_REGISTER(className)                                             \
    AbstractControlFlowNode* QWhileCreator##className(ClassicalCondition& classical_condition, QNode* true_node) \
    {      \
        return new className(classical_condition, true_node);                    \
    }                                                                   \
    QWhileRegisterAction _G_qwhile_creator_register##className(                        \
        #className,(CreateQWhile_cb)QWhileCreator##className)

/*
Create qwhile prog
param :
classical_condition : OriginQIf classical condition
true_node : true branch node
return : QIfProg
Note:
*/
extern QWhileProg CreateWhileProg(
    ClassicalCondition &,
    QNode* trueNode);

#endif // ! _CONTROL_FLOW_H