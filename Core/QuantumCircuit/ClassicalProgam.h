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
#ifndef  _CLASSICAL_PROGAM_H
#define  _CLASSICAL_PROGAM_H
#include "QNode.h"
#include "QuantumMachine/CBitFactory.h"
#include "ClassicalConditionInterface.h"
QPANDA_BEGIN

/**
 * @brief Classical program abstract class
 */
class AbstractClassicalProg
{
public:
    virtual ~AbstractClassicalProg() {};
    /**
     * @brief Get classical program value
     * 
     * @return cbit_size_t 
     */
    virtual cbit_size_t eval() = 0;
private:

};

/**
 * @brief classical program class
 * The proxy class of the AbstractClassicalProg implementation class
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
     * @brief Get the Node Type 
     * 
     * @return NodeType 
     */
    NodeType getNodeType() const;

    /**
     * @brief Get Position 
     * Get the current object`s position in the QNodeMap
     * @return qmap_size_t 
     */
    qmap_size_t getPosition() const;

    /**
     * @brief Get classical program value
     * 
     * @return cbit_size_t 
     */
    virtual cbit_size_t eval();

private:
    AbstractClassicalProg * m_node;
    void setPosition(qmap_size_t) {};
};

/**
 * @brief Origin classical program class
 * Implementation class of AbstractClassicalProg and QNode
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
     * 
     */
    ~OriginClassicalProg();

    /**
     * @brief Get the Node Type 
     * 
     * @return NodeType 
     */
    NodeType getNodeType() const;

    /**
     * @brief Get Position 
     * Get the current object`s position in the QNodeMap
     * @return qmap_size_t 
     */
    qmap_size_t getPosition() const;

    /**
     * @brief Set Position 
     * Set the current object`s position in the QNodeMap
     * @Param position The current object`s position in the QNodeMap
     */
    void setPosition(qmap_size_t position);
    
    /**
     * @brief Get classical program value
     * 
     * @return cbit_size_t 
     */
    virtual cbit_size_t eval();

private:
    std::shared_ptr<CExpr> m_expr;  ///< classical expr share ptr
    NodeType m_node_type;           ///< current QNode type
    qmap_size_t m_stPosition;       ///< current QNode position in the QNodeMap
};

typedef AbstractClassicalProg * (*CreateClassicalQProgram)(ClassicalCondition &);

/**
 * @brief classical program factory
 * Users can register their own implementation of AbstractClassicalProg 
 * through the ClassicalProgFactory class.
 */
class ClassicalProgFactory
{
public:

    /**
     * @brief register AbstractClassicalProg implementation class 
     * 
     * @param name Subclass name 
     * @param method Construction method for AbstractClassicalProg implementation class
     */
    void registClass(std::string name, CreateClassicalQProgram method);
    /**
     * @brief Get the AbstractClassicalProg implementation class ptr
     * @param name AbstractClassicalProg implementation class`s name
     * @param classical_cond classcial condition
     * @return AbstractClassicalProg* 
     */
    AbstractClassicalProg * getClassicalProgm(std::string & name,ClassicalCondition & classical_cond);

    /**
     * @brief Get the ClassicalProgFactory object
     * 
     * @return ClassicalProgFactory& 
     */
    static ClassicalProgFactory & getInstance()
    {
        static ClassicalProgFactory  s_Instance;
        return s_Instance;
    }
private:
    /**
     * @brief AbstractClassicalProg implementation class name and Construction method
     */
    std::map<std::string, CreateClassicalQProgram> m_ProgMap; 
    ClassicalProgFactory() {};
};

/**
 * @brief Classical program register action
 * Provide ClassicalProgFactory class registration interface to the outside
 */
class ClassicalProgRegisterAction {
public:
    /**
     * @brief Construct a new Classical Prog Register Action object
     * @param className AbstractClassicalProg implementation class`s name
     * @param ptrCreateFn Construction method for AbstractClassicalProg implementation class
     */
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