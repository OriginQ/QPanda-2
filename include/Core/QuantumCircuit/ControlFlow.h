/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ControlFlow.h 
Author: Menghan.Dou
Created in 2018-6-30

Classes for ControlFlow 

Update@2018-8-30
  Update by code specification
*/
/*! \file ControlFlow.h */

#ifndef  _CONTROL_FLOW_H
#define  _CONTROL_FLOW_H
#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"
#include "Core/QuantumCircuit/QProgram.h"

QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @class AbstractControlFlowNode
* @brief Superclass for QIfProg/QWhileProg
* @ingroup Core
*/
class AbstractControlFlowNode
{
public:    
    /**
     * @brief Get true branch
     * @return QNode* 
     */
    virtual QNode * getTrueBranch() const = 0;

    /**
     * @brief Get false branch
     * @return QNode* 
     */
    virtual QNode * getFalseBranch() const = 0;

    /**
     * @brief Set the True branch 
     * @param Node True branch node
     */
    virtual void setTrueBranch(QProg node) = 0;

    /**
     * @brief Set the False Branch object
     * @param Node False branch node
     */
    virtual void setFalseBranch(QProg node) = 0;

    /**
     * @brief Get classical expr
     * @return ClassicalCondition ptr 
     */
    virtual ClassicalCondition * getCExpr() = 0;
    virtual ~AbstractControlFlowNode() {}
};

/**
* @class QIfProg
* @brief Proxy class of quantum if program
* @ingroup Core
*/
class QIfProg : public QNode, public AbstractControlFlowNode
{
private:
    std::shared_ptr<AbstractControlFlowNode> m_control_flow;
    QIfProg();
public:
    ~QIfProg();
    /**
     * @brief Construct a new QIfProg object
     * @param old Target QIfProg 
     */
    QIfProg(const QIfProg &old);
    
    /**
     * @brief Construct a new QIfProg 
     * @param classical_condition  this QIfProg classical condition
     * @param true_node true branch node
     * @param false_node false branch node
     */
    QIfProg(ClassicalCondition classical_condition, QProg true_node, QProg false_node);

    /**
     * @brief Construct a new QIfProg object
     * @param classical_condition this QIfProg classical condition
     * @param node true branch node
     */   
    QIfProg(ClassicalCondition classical_condition, QProg node);
    
    /**
     * @brief Get the current node type
     * @return NodeType 
     */
    virtual NodeType getNodeType() const;
    
    /**
     * @brief Get the True Branch 
     * @return QNode ptr
     */
    virtual QNode* getTrueBranch() const;

    /**
     * @brief Get the False Branch
     * @return QNode ptr
     */
    virtual QNode* getFalseBranch() const;

    /**
     * @brief Get classical condition
     * @return classical condition ptr 
     */
    virtual ClassicalCondition *getCExpr();

    std::shared_ptr<QNode> getImplementationPtr();

private:
    virtual void setTrueBranch(QProg ) {};
    virtual void setFalseBranch(QProg ) {};
    virtual void execute(QPUImpl *, QuantumGateParam *) {};
};

typedef AbstractControlFlowNode * (*CreateQIfTrueFalse_cb)(ClassicalCondition &, QNode *, QNode *);
typedef AbstractControlFlowNode * (*CreateQIfTrueOnly_cb)(ClassicalCondition &, QNode *);


class QIfFactory
{
public:

    void registClass(std::string name, CreateQIfTrueFalse_cb method);

    void registClass(std::string name, CreateQIfTrueOnly_cb method);

    AbstractControlFlowNode* getQIf(std::string &class_name,
        ClassicalCondition &classical_condition,
        QNode *true_node,
        QNode *false_node);

    AbstractControlFlowNode * getQIf(std::string & name, 
                                     ClassicalCondition & classical_cond,
                                     QNode * node);

    static QIfFactory & getInstance()
    {
        static QIfFactory  instance;
        return instance;
    }
private:

    std::map<std::string, CreateQIfTrueOnly_cb> m_qif_true_only_map;

    std::map<std::string, CreateQIfTrueFalse_cb> m_qif_true_false_map;
    QIfFactory() {};

};

/**
* @class QIfRegisterAction
* @brief QIf program register action
* @note Provide QIfFactory class registration interface for the outside
 */
class QIfRegisterAction {
public:
    /**
     * @brief Construct a new QIfRegisterAction object
     * Call QIfFactory`s registClass interface
     * @param class_name AbstractControlFlowNode Implementation class name
     * @param create_callback The Constructor of Implementation class for AbstractControlFlowNode 
     *                        which have true and false branch
     */
    inline QIfRegisterAction(std::string class_name, CreateQIfTrueFalse_cb create_callback) {
        QIfFactory::getInstance().registClass(class_name, create_callback);
    }

    /**
     * @brief Construct a new QIfRegisterAction object
     * Call QIfFactory`s registClass interface
     * @param class_name AbstractControlFlowNode Implementation class name
     * @param create_callback The Constructor of Implementation class for AbstractControlFlowNode 
     *                        which only have branch
     */
    inline QIfRegisterAction(std::string class_name, CreateQIfTrueOnly_cb create_callback) {
        QIfFactory::getInstance().registClass(class_name, create_callback);
    }
};

#define QIF_REGISTER(className)                                             \
AbstractControlFlowNode* QifSingleCreator##className(ClassicalCondition classical_condition, QProg true_node)       \
{      \
    return new className(classical_condition, true_node);                    \
}                                                                   \
AbstractControlFlowNode* QifDoubleCreator##className(ClassicalCondition classical_condition, QProg true_node, QProg false_node) \
{      \
    return new className(classical_condition, true_node, false_node);                    \
}                                                                   \
QIfRegisterAction _G_qif_creator_double_register##className(                        \
    #className,(CreateQIfTrueFalse_cb)QifDoubleCreator##className);   \
QIfRegisterAction _G_qif_creator_single_register##className(                        \
    #className,(CreateQIfTrueOnly_cb)QifSingleCreator##className)



class OriginQIf : public QNode, public AbstractControlFlowNode
{
private:
    ClassicalCondition  m_classical_condition;
    Item * m_true_item {nullptr};
    Item * m_false_item {nullptr};
    NodeType m_node_type {QIF_START_NODE};
    std::shared_ptr<QNode> getImplementationPtr()
    {
        QCERR("Can't use this function");
        throw std::runtime_error("Can't use this function");
    };
public:
    ~OriginQIf();
    
    OriginQIf(ClassicalCondition classical_condition, QProg true_node, QProg false_node);

    OriginQIf(ClassicalCondition classical_condition, QProg node);

    virtual NodeType getNodeType() const;

    virtual QNode* getTrueBranch() const;

    virtual QNode* getFalseBranch() const;

    virtual void setTrueBranch(QProg node);

    virtual void setFalseBranch(QProg node);

    virtual ClassicalCondition *getCExpr();

    virtual void execute(QPUImpl *, QuantumGateParam *);
};

/**
* @brief  QPanda2 basic interface for creating a QIf program
* @ingroup  Core
* @param[in]  ClassicalCondition  Cbit
* @param[in]  QNode* QIf true node
* @return     QPanda::QIfProg  QIf program
*/
QIfProg CreateIfProg(
    ClassicalCondition classical_condition,
    QProg true_node);

/**
* @brief  QPanda2 basic interface for creating a QIf program
* @ingroup  Core
* @param[in]  ClassicalCondition  Cbit
* @param[in]  QNode* QIf true node
* @param[in]  QNode* QIf false node
* @return     QPanda::QIfProg  QIf program
*/
QIfProg CreateIfProg(
    ClassicalCondition classical_condition,
    QProg true_node,
    QProg false_node);

/**
* @class QWhileProg
* @brief Proxy class of quantum while program
* @ingroup Core
*/
class QWhileProg : public QNode, public AbstractControlFlowNode
{
private:
    std::shared_ptr<AbstractControlFlowNode> m_control_flow;

    QWhileProg();
public:
    ~QWhileProg();
    QWhileProg(const QWhileProg &);
    QWhileProg(ClassicalCondition , QProg);

    std::shared_ptr<QNode> getImplementationPtr();
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
    Get classical condition
    param :
    return : classical condition
    Note:
    */
    virtual ClassicalCondition *getCExpr();

private:
    virtual void setTrueBranch(QProg ) {};
    virtual void setFalseBranch(QProg ) {};
    virtual void execute(QPUImpl *, QuantumGateParam *) {};
};


class OriginQWhile :public QNode, public AbstractControlFlowNode
{
private:
    NodeType m_node_type {WHILE_START_NODE};
    ClassicalCondition  m_classical_condition;
    Item * m_true_item {nullptr};

    OriginQWhile();
    std::shared_ptr<QNode> getImplementationPtr()
    {
        QCERR("Can't use this function");
        throw std::runtime_error("Can't use this function");
    };
public:
    ~OriginQWhile();
    OriginQWhile(ClassicalCondition ccCon, QProg node);

    virtual NodeType getNodeType() const;

    virtual QNode* getTrueBranch() const;

    virtual QNode* getFalseBranch() const;

    virtual void setTrueBranch(QProg node);

    virtual void setFalseBranch(QProg node) {};

    virtual ClassicalCondition *getCExpr();

    virtual void execute(QPUImpl *, QuantumGateParam *);
};

typedef AbstractControlFlowNode * (*CreateQWhile_cb)(ClassicalCondition &, QNode *);

class QWhileFactory
{
public:
    void registClass(std::string name, CreateQWhile_cb method);
    AbstractControlFlowNode * getQWhile(std::string &, ClassicalCondition &, QNode *);
    static QWhileFactory & getInstance()
    {
        static QWhileFactory  instance;
        return instance;
    }

private:
    std::map<std::string, CreateQWhile_cb> m_qwhile_map;
    QWhileFactory() {};

};

class QWhileRegisterAction {
public:
    QWhileRegisterAction(std::string class_name, CreateQWhile_cb create_callback) {
        QWhileFactory::getInstance().registClass(class_name, create_callback);
    }
};

#define QWHILE_REGISTER(className)                                             \
AbstractControlFlowNode* QWhileCreator##className(ClassicalCondition &classical_condition, QProg true_node) \
{      \
    return new className(classical_condition, true_node);                    \
}                                                                   \
QWhileRegisterAction _G_qwhile_creator_register##className(                        \
    #className,(CreateQWhile_cb)QWhileCreator##className)

/**
* @brief  QPanda2 basic interface for creating a QWhile program
* @ingroup  Core
* @param[in]  ClassicalCondition  Cbit
* @param[in]  QNode* QWhile true node
* @return     QPanda::QWhileProg  QWhile program
*/

QWhileProg CreateWhileProg(
    ClassicalCondition ,
    QProg trueNode);
QPANDA_END
#endif // ! _CONTROL_FLOW_H
