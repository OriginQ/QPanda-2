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
/*! \file QProgram.h */
#ifndef _QPROGRAM_H_
#define _QPROGRAM_H_

#include <complex>
#include <initializer_list>
#include <vector>
#include <iterator>
#include <map>

#include "Core/QuantumCircuit/QNode.h"
#include "Core/Utilities/ReadWriteLock.h"
#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/QPandaException.h"
QPANDA_BEGIN

/**
* @defgroup Core
* @brief QPanda2 core component
*/

/**
* @class  AbstractQuantumProgram
* @brief   Quantum program basic abstract class
* @ingroup Core
*/

class AbstractQuantumProgram
{
public:
    virtual NodeIter  getFirstNodeIter() = 0;
    virtual NodeIter  getLastNodeIter() = 0;
    virtual NodeIter  getEndNodeIter() = 0;
    virtual NodeIter  getHeadNodeIter() = 0;
    virtual NodeIter  insertQNode(NodeIter &, QNode *) = 0;
    virtual NodeIter  deleteQNode(NodeIter &) =0;

    virtual void pushBackNode(QNode *) = 0;
    virtual void pushBackNode(std::shared_ptr<QNode>) = 0;
    virtual ~AbstractQuantumProgram() {};
    virtual void clear() = 0;
};

/**
* @class QProg
* @brief    Quantum program,can construct quantum circuit,data struct is linked list
* @ingroup  Core
*/
class QProg : public QNode,public AbstractQuantumProgram
{
private:
    std::shared_ptr<AbstractQuantumProgram> m_quantum_program;
public:
    QProg();
    QProg(const QProg&);

    template<typename Ty>
    QProg(Ty &node);

    QProg(QNode *);
    QProg(std::shared_ptr<QNode>);
    QProg(ClassicalCondition &node);

    ~QProg();
    std::shared_ptr<QNode> getImplementationPtr();
    void pushBackNode(QNode *);
    void pushBackNode(std::shared_ptr<QNode>);
    template<typename T>
    QProg & operator <<(T);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeIter insertQNode(NodeIter & iter, QNode * pNode);
    NodeIter deleteQNode(NodeIter & iter);
    NodeType getNodeType() const;
    void clear();
private:
    void execute(QPUImpl *, QuantumGateParam *) {};
};

typedef AbstractQuantumProgram * (*CreateQProgram)();
class QuantumProgramFactory
{
public:

    void registClass(std::string name, CreateQProgram method);
    AbstractQuantumProgram * getQuantumQProg(std::string &);

    static QuantumProgramFactory & getInstance()
    {
        static QuantumProgramFactory  instance;
        return instance;
    }
private:
    std::map<std::string, CreateQProgram> m_qprog_map;
    QuantumProgramFactory() {};
};

class QuantumProgramRegisterAction {
public:
    QuantumProgramRegisterAction(std::string className, CreateQProgram ptrCreateFn) {
        QuantumProgramFactory::getInstance().registClass(className, ptrCreateFn);
    }
};

#define REGISTER_QPROGRAM(className)                                           \
    AbstractQuantumProgram* QProgCreator##className(){                           \
        return new className();                                                   \
    }                                                                          \
    QuantumProgramRegisterAction g_qProgCreatorDoubleRegister##className(      \
        #className,(CreateQProgram)QProgCreator##className)

class OriginProgram :public QNode, public AbstractQuantumProgram
{
private:
    Item *m_head {nullptr};
    Item *m_end {nullptr};
    SharedMutex m_sm;
    NodeType m_node_type {PROG_NODE};
    OriginProgram(OriginProgram&);
    std::shared_ptr<QNode> getImplementationPtr()
    {
        QCERR("Can't use this function");
        throw std::runtime_error("Can't use this function");
    }
public:
    ~OriginProgram();
    OriginProgram();
    /**
    * @brief  Insert new node at the end of current quantum program node
    * @param[in]  QNode*  quantum node
    * @return     void
    * @see  QNode
    */
    void pushBackNode(QNode *);
    void pushBackNode(std::shared_ptr<QNode>);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeIter insertQNode(NodeIter &, QNode *);
    NodeIter deleteQNode(NodeIter &);
    NodeType getNodeType() const;

    /**
    * @brief  Clear all node in current quantum program node
    * @return     void
    */
    void clear();
    void execute(QPUImpl *, QuantumGateParam *);
};

/**
* @brief  QPanda2 basic interface for creating a empty quantum program
* @ingroup  Core
* @return     QPanda::QProg  quantum program
*/
QProg CreateEmptyQProg();

/**
* @brief  Insert new Node at the end of current node
* @param[in]  Node  QGate/QCircuit/QProg/QIf/QWhile
* @return     QPanda::QProg&   quantum program
* @see QNode
* @note
*    if T_GATE is QSingleGateNode/QDoubleGateNode/QIfEndNode,
*    deep copy T_GATE and insert it into left QProg;
*    if T_GATE is QIfProg/QWhileProg/QProg,deepcopy
*    IF/WHILE/QProg circuit and insert it into left QProg
*/
template<typename T>
QProg & QProg::operator<<(T node)
{
    if (!this->m_quantum_program)
    {
        throw std::runtime_error("m_quantum_program is nullptr");
    }

    auto temp = dynamic_cast<QNode *>(&node);
    if (nullptr != temp)
    {
        int iNodeType = temp->getNodeType();
        m_quantum_program->pushBackNode(temp);
        return *this;
    }
    else
    {
        throw qprog_construction_fail("bad node type");
    }
}
template <>
QProg & QProg::operator<<<ClassicalCondition >(ClassicalCondition  node);

template<typename Ty>
QProg::QProg(Ty &node)
    :QProg()
{
    if (!this->m_quantum_program)
    {
        throw std::runtime_error("m_quantum_program is nullptr");
    }
    static_assert(std::is_base_of<QNode, Ty>::value, "bad node type");
    m_quantum_program->pushBackNode(&node);
}

QPANDA_END
#endif
