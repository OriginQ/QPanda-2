/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.

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
/*! \file ClassicalProgram.h */
#ifndef  _CLASSICAL_PROGAM_H
#define  _CLASSICAL_PROGAM_H
#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumMachine/CBitFactory.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"
#include <iostream>
QPANDA_BEGIN

/**
* @class AbstractClassicalProg
* @brief Classical program abstract class
* @ingroup Core
*/
class AbstractClassicalProg
{
public:
    virtual ~AbstractClassicalProg() {};
    /**
     * @brief Get classical program value
     * @return cbit_size_t 
     */
    virtual std::shared_ptr<CExpr> getExpr() = 0;
    virtual cbit_size_t eval() = 0;
private:

};

/**
* @class ClassicalProg
* @brief Classical program class
* @ingroup Core
* @note  The proxy class of the AbstractClassicalProg implementation class
*/
class ClassicalProg :public QNode,public AbstractClassicalProg
{
public:
    /**
     * @brief Construct a new Classical Prog object
     * @param classical_cond Target classical condition
     */
    ClassicalProg(ClassicalCondition & classical_cond);

    /**
     * @brief Construct a new Classical Prog object
     * @param old Target classical program
     */
    ClassicalProg(const ClassicalProg & old);
    ~ClassicalProg();

    /**
    * @brief  Get current node type
    * @return  NodeType  current node type
    * @see  NodeType
    */
    NodeType getNodeType() const;

    std::shared_ptr<QNode> getImplementationPtr();

    /**
     * @brief Get classical program value
     * @return cbit_size_t 
     */
    virtual cbit_size_t eval();

    std::shared_ptr<CExpr> getExpr();

private:
    std::shared_ptr<AbstractClassicalProg> m_node;
    virtual void execute(QPUImpl *, QuantumGateParam *) {};
};

/**
* @class OriginClassicalProg
 * @brief Origin classical program class
 * @note Implementation class of AbstractClassicalProg and QNode
 * This class type can hold classical expr and insert into QNodeMap
 */
class OriginClassicalProg :public QNode, public AbstractClassicalProg
{
public:
    /**
     * @brief Construct a new Origin Classical Prog object
     * @param classical_cond Target classical condition
     */
    OriginClassicalProg(ClassicalCondition & );
    
    /**
     * @brief Destroy the Origin Classical Prog object
     */
    ~OriginClassicalProg();

    /**
    * @brief  Get current node type
    * @return  NodeType  current node type
    * @see  NodeType
    */
    NodeType getNodeType() const;
    
    /**
     * @brief Get classical program value
     * @return cbit_size_t 
     */
    virtual cbit_size_t eval();

    virtual void execute(QPUImpl *, QuantumGateParam *);

    inline std::shared_ptr<CExpr> getExpr()
    {
        return m_expr;
    }

private:
    std::shared_ptr<CExpr> m_expr;  ///< classical expr share ptr
    NodeType m_node_type;           ///< current QNode type

    std::shared_ptr<QNode> getImplementationPtr()
    {
        QCERR("Can't use this function");
        throw std::runtime_error("Can't use this function");
    };
};

typedef AbstractClassicalProg * (*CreateClassicalQProgram)(ClassicalCondition &);


class ClassicalProgFactory
{
public:
    void registClass(std::string name, CreateClassicalQProgram method);

    AbstractClassicalProg * getClassicalProgm(std::string & name,ClassicalCondition & classical_cond);

    static ClassicalProgFactory & getInstance()
    {
        static ClassicalProgFactory  s_Instance;
        return s_Instance;
    }
private:
    std::map<std::string, CreateClassicalQProgram> m_ProgMap; 
    ClassicalProgFactory() {};
};

class ClassicalProgRegisterAction {
public:

    inline ClassicalProgRegisterAction(std::string className, CreateClassicalQProgram ptrCreateFn) {
        ClassicalProgFactory::getInstance().registClass(className, ptrCreateFn);
    }
};

#define REGISTER_CLASSICAL_PROGRAM(className)                                        \
    AbstractClassicalProg* ClassicalQProgCreator##className(ClassicalCondition & cc){\
        return new className(cc);                                                   \
    }\
    ClassicalProgRegisterAction g_qClassicalProgCreatorRegister##className(      \
        #className,(CreateClassicalQProgram)ClassicalQProgCreator##className);


QPANDA_END
#endif
