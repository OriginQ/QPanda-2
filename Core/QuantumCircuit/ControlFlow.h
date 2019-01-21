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


#ifndef  _CONTROL_FLOW_H
#define  _CONTROL_FLOW_H
#include "QNode.h"
#include "ClassicalConditionInterface.h"

QPANDA_BEGIN

/**
 * @brief Superclass for QIfProg/QWhileProg
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
     * @param node True branch node
     */
    virtual void setTrueBranch(QNode * node) = 0;

    /**
     * @brief Set the False Branch object
     * @param node False branch node
     */
    virtual void setFalseBranch(QNode * node) = 0;

    /**
     * @brief get classical expr
     * @return ClassicalCondition ptr 
     */
    virtual ClassicalCondition * getCExpr() = 0;
    virtual ~AbstractControlFlowNode() {}
};

/*
* @brief Proxy class of quantum if program
*/
class QIfProg : public QNode, public AbstractControlFlowNode
{
private:
    AbstractControlFlowNode * m_control_flow;
    qmap_size_t m_position;
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
    QIfProg(ClassicalCondition& classical_condition, QNode *true_node, QNode *false_node);

    /**
     * @brief Construct a new QIfProg object
     * @param classical_condition this QIfProg classical condition
     * @param node true branch node
     */   
    QIfProg(ClassicalCondition &classical_condition, QNode * node);
    
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
     * @brief Get this node position in global QNode map
     * @return position 
     */
    virtual qmap_size_t getPosition() const;

    /**
     * @brief Get classical condition
     * @return classical condition ptr 
     */
    virtual ClassicalCondition *getCExpr();

private:
    virtual void setPosition(qmap_size_t) {};
    virtual void setTrueBranch(QNode*) {};
    virtual void setFalseBranch(QNode*) {};
};

typedef AbstractControlFlowNode * (*CreateQIfTrueFalse_cb)(ClassicalCondition &, QNode *, QNode *);
typedef AbstractControlFlowNode * (*CreateQIfTrueOnly_cb)(ClassicalCondition &, QNode *);

/*
* @brief Factory for generating QIf.
*/
class QIfFactory
{
public:
    /**
     * @brief Regist QIf class
     * 
     * @param name class name
     * @param method create qif true false callback
     */

    void registClass(std::string name, CreateQIfTrueFalse_cb method);

   /**
    * @brief Regist QIf class
    * @param name class name
    * @param method create qif true only callback
    */  
    void registClass(std::string name, CreateQIfTrueOnly_cb method);

    /**
     * @brief Get QIf class
     * @param class_name AbstractControlFlowNode Implementation class name
     * @param classical_condition classical condition
     * @param true_node true branch node
     * @param false_node false branch node
     * @return AbstractControlFlowNode ptr
     */
    AbstractControlFlowNode* getQIf(std::string &class_name,
        ClassicalCondition &classical_condition,
        QNode *true_node,
        QNode *false_node);

    /**
     * @brief Get QIf class
     * @param name AbstractControlFlowNode Implementation class name
     * @param classical_cond classical condition
     * @param node Ture branch node ptr
     * @return AbstractControlFlowNode ptr
     */
    AbstractControlFlowNode * getQIf(std::string & name, 
                                     ClassicalCondition & classical_cond,
                                     QNode * node);
    
    /**
     * @brief Get the QIfFactory object
     * @return QIfFactory ref
     */
    static QIfFactory & getInstance()
    {
        static QIfFactory  instance;
        return instance;
    }
private:
    /**
     * @brief A collection of constructors that create only true branch qif 
     */
    std::map<std::string, CreateQIfTrueOnly_cb> m_qif_true_only_map;

    /**
     * @brief A collection of constructors that reate a qif that contains the true and false branches
     */
    std::map<std::string, CreateQIfTrueFalse_cb> m_qif_true_false_map;
    QIfFactory() {};

};

/**
 * @brief QIf program register action
 * Provide QIfFactory class registration interface to the outside
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


/**
 * @brief Qif implementation class
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
    
    /**
     * @brief OriginQIf constructor
     * 
     * @param classical_condition OriginQIf classical condition
     * @param true_node true branch node
     * @param false_node false branch node
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
    virtual ClassicalCondition *getCExpr();
};

/*
Create qif prog
param :
classical_condition : OriginQIf classical condition
true_node : true branch node
return : QIfProg
Note:
*/
QIfProg CreateIfProg(
    ClassicalCondition classical_condition,
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
QIfProg CreateIfProg(
    ClassicalCondition classical_condition,
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
    QWhileProg(ClassicalCondition &, QNode *);

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
    virtual void setPosition(qmap_size_t) {};
    virtual void setTrueBranch(QNode*) {};
    virtual void setFalseBranch(QNode*) {};
};

/*
*  OriginQWhile: QWhile implementation class
*/
class OriginQWhile :public QNode, public AbstractControlFlowNode
{
private:
    NodeType m_node_type;
    ClassicalCondition  m_classical_condition;
    Item * m_true_item;
    qmap_size_t m_position;

    OriginQWhile();
public:
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
    virtual ClassicalCondition *getCExpr();
};

typedef AbstractControlFlowNode * (*CreateQWhile_cb)(ClassicalCondition &, QNode *);

/*
* QWhileFactory: Factory for generating QWhile.
*/
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
QWhileProg CreateWhileProg(
    ClassicalCondition ,
    QNode* trueNode);
QPANDA_END
#endif // ! _CONTROL_FLOW_H