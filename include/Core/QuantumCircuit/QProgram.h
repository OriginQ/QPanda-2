/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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
#include "Core/Utilities/Tools/ReadWriteLock.h"
#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/QuantumCircuit/QNodeManager.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"

QPANDA_BEGIN

/**
* @class  AbstractQuantumProgram
* @brief   Quantum program basic abstract class
* @ingroup QuantumCircuit
*/
class AbstractQuantumProgram : public AbstractNodeManager
{
public:
    virtual ~AbstractQuantumProgram() {};
	
	/**
    * @brief  Clear all node in current quantum program node
    */
    virtual void clear() = 0;


    /**
    * @brief  Gets the maximum physical address of used qubits 
    * @return size_t maximum physical address
    */
    virtual size_t get_max_qubit_addr() = 0;

    /**
    * @brief  Get the used qubits for current quantum program
    * @param[out]  QVec  used qubits  vector
    * @return  size_t
    */
    virtual size_t get_used_qubits(QVec&) = 0;

    /**
    * @brief  Get the used  classical bits for current quantum program
    * @param[out]  QVec  used qubits  vector
    * @return  size_t
    */
    virtual size_t get_used_cbits(std::vector<ClassicalCondition>&) = 0;

    /**
    * @brief  Get current quantum program qgate number
    * @return  size_t
    */
    virtual size_t get_qgate_num() = 0;

    /**
    * @brief  Measure  operation  in the last position of the program
    * @return  bool
    */
    virtual bool  is_measure_last_pos()= 0;

    /**
    * @brief  Get Measure operation  position of the program
    * @return   std::map<Qubit*, bool>
    */
    virtual std::map<Qubit*, bool> get_measure_pos() = 0;

    /**
    * @brief  Get Measure operation qubits and cbits vector
    * @return   std::vector<std::pair<Qubit*, ClassicalCondition>> 
    */
    virtual std::vector<std::pair<Qubit*, ClassicalCondition>> get_measure_qubits_cbits() = 0;
};

/**
* @class QProg
* @brief    Quantum program,can construct quantum circuit,data struct is linked list
* @ingroup  QuantumCircuit
*/
class QProg : public AbstractQuantumProgram
{
private:
    std::shared_ptr<AbstractQuantumProgram> m_quantum_program;
public:
    QProg();
	QProg(const QProg&);

    template<typename Ty>
    QProg(Ty &node);

    QProg(std::shared_ptr<QNode>);
    QProg(std::shared_ptr<AbstractQuantumProgram>);
    QProg(ClassicalCondition &node);
	QProg(QProg& other);
	
    ~QProg();
    std::shared_ptr<AbstractQuantumProgram> getImplementationPtr();
    void pushBackNode(std::shared_ptr<QNode>);
    template<typename T>
    QProg & operator <<(T);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeIter insertQNode(const NodeIter & iter, std::shared_ptr<QNode>);
    NodeIter deleteQNode(NodeIter & iter);
    NodeType getNodeType() const;
    void clear();
	bool is_empty() { return getFirstNodeIter() == getEndNodeIter(); }
    virtual size_t get_max_qubit_addr();
    virtual size_t get_used_qubits(QVec&);
    virtual size_t get_used_cbits(std::vector<ClassicalCondition>&);
    virtual size_t get_qgate_num();
    virtual bool  is_measure_last_pos();
    virtual std::map<Qubit*, bool> get_measure_pos();
    virtual std::vector<std::pair<Qubit*, ClassicalCondition>> get_measure_qubits_cbits();
};

typedef AbstractQuantumProgram * (*CreateQProgram)();

/**
* @brief   Factory for class AbstractQuantumProgram
* @ingroup QuantumCircuit
*/
class QuantumProgramFactory
{
public:

    void registClass(std::string name, CreateQProgram method);
    AbstractQuantumProgram * getQuantumQProg(std::string &);

	/**
     * @brief Get the static instance of factory 
	 * @return QuantumProgramFactory &
     */
    static QuantumProgramFactory & getInstance()
    {
        static QuantumProgramFactory  instance;
        return instance;
    }
private:
    std::map<std::string, CreateQProgram> m_qprog_map;
    QuantumProgramFactory() {};
};

/**
* @brief Quantum program register action
* @note Provide QuantumProgramFactory class registration interface for the outside
 */
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
    INIT_PRIORITY QuantumProgramRegisterAction g_qProgCreatorDoubleRegister##className(      \
        #className,(CreateQProgram)QProgCreator##className)


/**
* @brief Implementation  class of QProg
* @ingroup QuantumCircuit
*/
class OriginProgram :public QNode, public AbstractQuantumProgram
{
private:
    QNodeManager m_node_manager{ this };
    SharedMutex m_sm;
    NodeType m_node_type{ PROG_NODE };
    OriginProgram(OriginProgram&);
    QVec m_used_qubit_vector;
    size_t m_qgate_num{ 0 };
    std::vector<ClassicalCondition> m_used_cbit_vector;
    std::map<Qubit*, bool> m_last_measure;
    std::vector<std::pair<Qubit*, ClassicalCondition>>  m_mea_qubits_cbits;
public:
    ~OriginProgram();
    OriginProgram();
    /**
    * @brief  Insert new node at the end of current quantum program node
    * @param[in]  QNode*  quantum node
    * @return     void
    * @see  QNode
    */
    void pushBackNode(std::shared_ptr<QNode> node) {
        check_insert_node_type(node);
        m_node_manager.push_back_node(node);
    }
    NodeIter getFirstNodeIter() { return m_node_manager.get_first_node_iter(); }
    NodeIter getLastNodeIter() { return m_node_manager.get_last_node_iter(); }
    NodeIter getEndNodeIter() { return m_node_manager.get_end_node_iter(); }
    NodeIter getHeadNodeIter() { return m_node_manager.get_head_node_iter(); }
	NodeIter insertQNode(const NodeIter& perIter, std::shared_ptr<QNode> node) {
		check_insert_node_type(node);
		return m_node_manager.insert_QNode(perIter, node);
	}
	NodeIter deleteQNode(NodeIter& target_iter) {
		m_qgate_num--;
		return m_node_manager.delete_QNode(target_iter);
	}
    NodeType getNodeType() const;

    /**
    * @brief  Clear all node in current quantum program node
    * @return     void
    */
    void clear() { m_node_manager.clear(); }

    bool check_insert_node_type(std::shared_ptr<QNode> node);

    size_t get_max_qubit_addr();

    size_t get_used_qubits(QVec& qubit_vector);

    size_t get_used_cbits(std::vector<ClassicalCondition>&);

    size_t get_qgate_num();

    std::map<Qubit*, bool> get_measure_pos() { return m_last_measure; }

    bool is_measure_last_pos();

    std::vector<std::pair<Qubit*, ClassicalCondition>> get_measure_qubits_cbits() { return m_mea_qubits_cbits; }

};

/* will delete */
QProg CreateEmptyQProg();

/* new interface */
/**
* @brief  QPanda2 basic interface for creating a empty quantum program
* @ingroup  QuantumCircuit
* @return     QPanda::QProg  quantum program
*/
QProg createEmptyQProg();

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

	m_quantum_program->pushBackNode(std::dynamic_pointer_cast<QNode>(node.getImplementationPtr()));
	return *this;
}
template <>
QProg & QProg::operator<<<ClassicalCondition >(ClassicalCondition node);

template<typename Ty>
QProg::QProg(Ty &node)
    :QProg()
{
    if (!this->m_quantum_program)
    {
        throw std::runtime_error("m_quantum_program is nullptr");
    }
    m_quantum_program->pushBackNode(std::dynamic_pointer_cast<QNode>(node.getImplementationPtr()));
}

QPANDA_END
#endif
