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
* @brief Superclass for QIfProg/QWhileProg
* @ingroup QuantumCircuit
*/
class AbstractControlFlowNode
{
public:    
    /**
     * @brief Get true branch
     * @return std::shared_ptr<QNode>
     */
    virtual std::shared_ptr<QNode> getTrueBranch() const = 0;

    /**
     * @brief Get false branch
     * @return std::shared_ptr<QNode>
     */
    virtual std::shared_ptr<QNode> getFalseBranch() const = 0;

    /**
     * @brief Set the True branch 
     * @param[in] Node True branch node
     */
    virtual void setTrueBranch(QProg node) = 0;

    /**
     * @brief Set the False Branch object
     * @param[in] Node False branch node
     */
    virtual void setFalseBranch(QProg node) = 0;

    /**
     * @brief Get classical expr
     * @return ClassicalCondition ptr 
     */
    virtual ClassicalCondition  getCExpr() = 0;
    virtual ~AbstractControlFlowNode() {}
};

/**
* @brief Proxy class of quantum if program
* @ingroup QuantumCircuit
*/
class QIfProg : public AbstractControlFlowNode
{
private:
    std::shared_ptr<AbstractControlFlowNode> m_control_flow;
    QIfProg();
public:
    ~QIfProg();
    /**
     * @brief Construct a new QIfProg object
     * @param[in] old Target QIfProg 
     */
    QIfProg(const QIfProg &old);
    class OriginQIf;
    QIfProg(std::shared_ptr<AbstractControlFlowNode > qif)
    {
        if (qif)
        {
            auto node = std::dynamic_pointer_cast<QNode>(qif);
            if (node->getNodeType() == QWAIT_NODE)
                m_control_flow = qif;
            else
            {
                QCERR("node error");
                throw std::runtime_error("node error");
            }
        }
        else
        {
            QCERR("node null");
            throw std::runtime_error("node null");
        }
    }
    
    /**
     * @brief Construct a new QIfProg 
     * @param[in] classical_condition  this QIfProg classical condition
     * @param[in] true_node true branch node
     * @param[in] false_node false branch node
     */
    QIfProg(ClassicalCondition classical_condition, QProg true_node, QProg false_node);

    /**
     * @brief Construct a new QIfProg object
     * @param[in] classical_condition this QIfProg classical condition
     * @param[in] node true branch node
     */   
    QIfProg(ClassicalCondition classical_condition, QProg node);
    
    /**
     * @brief Get the current node type
     * @return NodeType 
     */
    virtual NodeType getNodeType() const;
    
    /**
     * @brief Get the True Branch 
     * @return std::shared_ptr<QNode>
     */
    virtual std::shared_ptr<QNode> getTrueBranch() const;

    /**
     * @brief Get the False Branch
     * @return std::shared_ptr<QNode>
     */
    virtual std::shared_ptr<QNode> getFalseBranch() const;

    std::shared_ptr<AbstractControlFlowNode> getImplementationPtr();

    /* will delete */
    virtual ClassicalCondition getCExpr();

    /* new interface */
    /**
    * @brief  get a classical condition
    * @return   ClassicalCondition
    */
    virtual ClassicalCondition getClassicalCondition();
private:
    virtual void setTrueBranch(QProg ) {};
    virtual void setFalseBranch(QProg ) {};
};

typedef AbstractControlFlowNode * (*CreateQIfTrueFalse_cb)(ClassicalCondition &, QProg, QProg );
typedef AbstractControlFlowNode * (*CreateQIfTrueOnly_cb)(ClassicalCondition &, QProg );


/**
 * @brief Factory for class AbstractControlFlowNode
 * @ingroup QuantumCircuit
 */
class QIfFactory
{
public:

    void registClass(std::string name, CreateQIfTrueFalse_cb method);

    void registClass(std::string name, CreateQIfTrueOnly_cb method);

    AbstractControlFlowNode* getQIf(std::string &class_name,
        ClassicalCondition &classical_condition,
		QProg true_node,
		QProg false_node);

    AbstractControlFlowNode * getQIf(std::string & name, 
                                     ClassicalCondition & classical_cond,
									QProg node);

	/**
     * @brief Get the static instance of factory 
	 * @return QIfFactory &
     */
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
* @brief QIf program register action
* @ingroup QuantumCircuit
* @note Provide QIfFactory class registration interface for the outside
 */
class QIfRegisterAction {
public:
    /**
     * @brief Construct a new QIfRegisterAction object
     * Call QIfFactory`s registClass interface
     * @param[in] class_name AbstractControlFlowNode Implementation class name
     * @param[in] create_callback The Constructor of Implementation class for AbstractControlFlowNode 
     *                        which have true and false branch
     */
    inline QIfRegisterAction(std::string class_name, CreateQIfTrueFalse_cb create_callback) {
        QIfFactory::getInstance().registClass(class_name, create_callback);
    }

    /**
     * @brief Construct a new QIfRegisterAction object
     * Call QIfFactory`s registClass interface
     * @param[in] class_name AbstractControlFlowNode Implementation class name
     * @param[in] create_callback The Constructor of Implementation class for AbstractControlFlowNode 
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


/**
* @brief Implementation  class of QIfProg  
* @ingroup QuantumCircuit
*/
class OriginQIf : public QNode, public AbstractControlFlowNode
{
private:
    ClassicalCondition  m_classical_condition;
    Item * m_true_item {nullptr};
    Item * m_false_item {nullptr};
    NodeType m_node_type {QIF_START_NODE};
public:
    ~OriginQIf();
    
    OriginQIf(ClassicalCondition classical_condition, QProg true_node, QProg false_node);

    OriginQIf(ClassicalCondition classical_condition, QProg node);

    virtual NodeType getNodeType() const;

    virtual std::shared_ptr<QNode> getTrueBranch() const;

    virtual std::shared_ptr<QNode> getFalseBranch() const;

    virtual void setTrueBranch(QProg node);

    virtual void setFalseBranch(QProg node);

    virtual ClassicalCondition getCExpr();
};


/**
* @brief Proxy class of quantum while program
* @ingroup QuantumCircuit
*/
class QWhileProg : public AbstractControlFlowNode
{
private:
    std::shared_ptr<AbstractControlFlowNode> m_control_flow;

    QWhileProg();
public:
    ~QWhileProg();
    QWhileProg(const QWhileProg &);
    QWhileProg(std::shared_ptr<AbstractControlFlowNode> qwhile)
    {
        if (qwhile)
        {
            auto node = std::dynamic_pointer_cast<QNode>(qwhile);
            if (node->getNodeType() == QWAIT_NODE)
                m_control_flow = qwhile;
            else
            {
                QCERR("node error");
                throw std::runtime_error("node error");
            }
        }
        else
        {
            QCERR("node null");
            throw std::runtime_error("node null");
        }


    }
    QWhileProg(ClassicalCondition , QProg);

    std::shared_ptr<AbstractControlFlowNode> getImplementationPtr();

    virtual NodeType getNodeType() const;

    virtual std::shared_ptr<QNode>  getTrueBranch() const;

    virtual std::shared_ptr<QNode> getFalseBranch() const;

    /* will delete */
    virtual ClassicalCondition getCExpr();

    /* new interface  */

    virtual ClassicalCondition getClassicalCondition();

private:
    virtual void setTrueBranch(QProg ) {};
    virtual void setFalseBranch(QProg ) {};
};

/**
* @brief Implementation  class of QWhileProg
* @ingroup QuantumCircuit
*/
class OriginQWhile :public QNode, public AbstractControlFlowNode
{
private:
    NodeType m_node_type {WHILE_START_NODE};
    ClassicalCondition  m_classical_condition;
    Item * m_true_item {nullptr};

    OriginQWhile();

public:
    ~OriginQWhile();
    OriginQWhile(ClassicalCondition ccCon, QProg node);

    virtual NodeType getNodeType() const;

    virtual std::shared_ptr<QNode>  getTrueBranch() const;

    virtual std::shared_ptr<QNode>  getFalseBranch() const;

    virtual void setTrueBranch(QProg node);

    virtual void setFalseBranch(QProg node) {};

    virtual ClassicalCondition getCExpr();
};

typedef AbstractControlFlowNode * (*CreateQWhile_cb)(ClassicalCondition &, QProg );

/**
 * @brief QWhile factory
 * @ingroup QuantumCircuit
 */
class QWhileFactory
{
public:
    void registClass(std::string name, CreateQWhile_cb method);
    AbstractControlFlowNode * getQWhile(std::string &, ClassicalCondition &, QProg );
    static QWhileFactory & getInstance()
    {
        static QWhileFactory  instance;
        return instance;
    }

private:
    std::map<std::string, CreateQWhile_cb> m_qwhile_map;
    QWhileFactory() {};

};

/**
* @brief QWhile program register action
* @note Provide QWhileFactory class registration interface for the outside
 */
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


/* will delete */
QIfProg CreateIfProg(
    ClassicalCondition classical_condition,
    QProg true_node);
QIfProg CreateIfProg(
    ClassicalCondition classical_condition,
    QProg true_node,
    QProg false_node);
QWhileProg CreateWhileProg(
    ClassicalCondition ,
    QProg trueNode);


/* new interface */
/**
* @brief  QPanda2 basic interface for creating a QIf program
* @ingroup  QuantumCircuit
* @param[in]  ClassicalCondition  Cbit
* @param[in]  QProg QIf true node
* @return     QIfProg  QIf program
*/
QIfProg createIfProg(
    ClassicalCondition cc,
    QProg true_node);

/**
* @brief  QPanda2 basic interface for creating a QIf program
* @ingroup  QuantumCircuit
* @param[in]  ClassicalCondition  Cbit
* @param[in]  QProg QIf true node
* @param[in]  QProg QIf false node
* @return     QIfProg  QIf program
*/
QIfProg createIfProg(
    ClassicalCondition cc,
    QProg true_node,
    QProg false_node);

/**
* @brief  QPanda2 basic interface for creating a QWhile program
* @ingroup  QuantumCircuit
* @param[in]  ClassicalCondition  Cbit
* @param[in]  QProg QWhile true node
* @return     QWhileProg  QWhile program
*/

QWhileProg createWhileProg(
    ClassicalCondition cc,
    QProg true_node);

QPANDA_END
#endif // ! _CONTROL_FLOW_H
