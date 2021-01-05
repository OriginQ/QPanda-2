/*! \file QCircuit.h */
#ifndef _QCIRCUIT_H
#define _QCIRCUIT_H

#include <complex>
#include <initializer_list>
#include <vector>
#include <iterator>
#include <map>

#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/Tools/ReadWriteLock.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/QuantumCircuit/QNodeManager.h"

QPANDA_BEGIN

/**
* @brief Quantum circuit basic abstract class
* @ingroup QuantumCircuit
*/
class AbstractQuantumCircuit : public AbstractNodeManager
{
public:
	/**
    * @brief  Judge current quantum circuit is dagger
    * @return  bool 
    */
    virtual bool isDagger() const = 0;

	/**
    * @brief  Get control vector fron current quantum circuit node
    * @param[in]  QVec& qubits  vector
	* @return  bool
    * @see QVec
    */
    virtual bool getControlVector(QVec &) = 0;

	/**
    * @brief  Set dagger to current quantum circuit
    * @param[in]  bool is dagger
    */
    virtual void  setDagger(bool isDagger) = 0;

	/**
    * @brief  Set control qubits to current quantum circuit
    * @param[in]  QVec  control qubits  vector
    * @see QVec
    */
    virtual void  setControl(QVec ) = 0;

	/**
    * @brief  Clear the control qubits for current quantum circuit
    * @see QVec
    */
    virtual void clearControl() = 0;
    virtual ~AbstractQuantumCircuit() {};
};

/**
* @brief Quantum circuit basic abstract class
* @ingroup QuantumCircuit
*/
class QCircuit : public AbstractQuantumCircuit
{
protected:
    std::shared_ptr<AbstractQuantumCircuit> m_pQuantumCircuit;
public:
    QCircuit();
    QCircuit(const QCircuit &);
    QCircuit(QGate & gate);
    QCircuit(std::shared_ptr<AbstractQuantumCircuit> node);
    ~QCircuit();
    std::shared_ptr<AbstractQuantumCircuit> getImplementationPtr();

    /**
    * @brief  Insert new Node at the end of current quantum circuit node
    * @param[in]  QNode*  quantum node
    * @return     void
    * @see  QNode
    */
    void pushBackNode(std::shared_ptr<QNode>) ;

    /**
    * @brief  Insert new Node at the end of current node
    * @param[in]  node  QGate/QCircuit
    * @return     QPanda::QCircuit&   quantum circuit
    * @see QNode
    */
    template<typename T>
    QCircuit & operator <<(T node);

    /**
    * @brief  Get a dagger circuit  base on current quantum circuit node
    * @return     QPanda::QCircuit  quantum circuit
    */
    virtual QCircuit  dagger();
    /**
    * @brief  Get a control quantumgate  base on current quantum circuit node
    * @param[in]  QVec control qubits  vector
    * @return     QPanda::QCircuit  quantum circuit
    * @see QVec
    */
    virtual QCircuit  control(const QVec );

    /**
    * @brief  Get current node type
    * @return     NodeType  current node type
    * @see  NodeType
    */
    NodeType getNodeType() const;

    /**
    * @brief  Judge current quantum circuit is dagger
	* @return  bool
    */
    bool isDagger() const;

    /**
    * @brief  Get control vector from current quantum circuit node
    * @param[in]  QVec& qubits  vector
	* @return  bool
    * @see QVec
    */
    bool getControlVector(QVec &);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter getEndNodeIter();
    NodeIter getHeadNodeIter();

    NodeIter insertQNode(const NodeIter &iter, std::shared_ptr<QNode> pNode);
    NodeIter deleteQNode(NodeIter &iter);
	bool is_empty() { return getFirstNodeIter() == getEndNodeIter(); }

    /**
    * @brief  Set dagger to current quantum circuit
    * @param[in]  bool is dagger
    */
    virtual void  setDagger(bool isDagger);

    /**
    * @brief  Set control qubits to current quantum circuit
    * @param[in]  QVec  control qubits  vector
    * @see QVec
    */
    virtual void  setControl(const QVec );
private:
    void clearControl() {}
};

/**
* @brief Hadamard quantum circuit program
* @ingroup QuantumCircuit
*/
class HadamardQCircuit :public QCircuit
{
public:
    HadamardQCircuit(QVec & pQubitVector);
    ~HadamardQCircuit() {}
private:
    HadamardQCircuit();
};

/* will delete */
QCircuit CreateEmptyCircuit();
HadamardQCircuit CreateHadamardQCircuit(QVec & pQubitVector);

/* new interface */
/**
* @brief  QPanda2 basic interface for creating a empty circuit
* @ingroup QuantumCircuit
* @return     QPanda::QCircuit  
*/
QCircuit createEmptyCircuit();

/**
* @brief  Create a hadamard qcircuit
* @ingroup QuantumCircuit
* @param[in]  QVec&  qubit vector 
* @return     QPanda::HadamardQCircuit  hadamard qcircuit 
*/
HadamardQCircuit createHadamardQCircuit(QVec & pQubitVector);


/**
* @brief Implementation  class of QCircuit 
* @ingroup QuantumCircuit
*/
class OriginCircuit : public QNode, public AbstractQuantumCircuit
{
private:
	QNodeManager m_node_manager{this};
    SharedMutex m_sm;
    NodeType m_node_type;
    bool m_Is_dagger;
    QVec m_control_qubit_vector;
    OriginCircuit(const OriginCircuit &);

public:
    OriginCircuit():
        m_node_type(CIRCUIT_NODE),
        m_Is_dagger(false)
    {
        m_control_qubit_vector.resize(0);
    }
    ~OriginCircuit();
	void pushBackNode(std::shared_ptr<QNode> node) { 
		if (check_insert_node_type(node))
		{
			m_node_manager.push_back_node(node);
		}
	}
    void setDagger(bool);
    void setControl(QVec );
    NodeType getNodeType() const;
    bool isDagger() const;
    bool getControlVector(QVec &);
    NodeIter  getFirstNodeIter() { return m_node_manager.get_first_node_iter(); }
    NodeIter  getLastNodeIter() { return m_node_manager.get_last_node_iter(); }
    NodeIter  getEndNodeIter() { return m_node_manager.get_end_node_iter(); }
    NodeIter getHeadNodeIter() { return m_node_manager.get_head_node_iter(); }
    NodeIter  insertQNode(const NodeIter &perIter, std::shared_ptr<QNode> node) {
		if (check_insert_node_type(node))
		{
			return m_node_manager.insert_QNode(perIter, node);
		}

		return NodeIter();
	}
    NodeIter  deleteQNode(NodeIter &target_iter) { return m_node_manager.delete_QNode(target_iter); }
	/**
	* @brief  Clear all node in current quantum program node
	* @return     void
	*/
	void clear() { m_node_manager.clear(); }

    void clearControl();

	bool check_insert_node_type(std::shared_ptr<QNode> node) {
		if (nullptr == node.get())
		{
			QCERR("node is null");
			throw std::runtime_error("node is null");
		}

		const NodeType t = node->getNodeType();
		switch (t)
		{
		case GATE_NODE:
		case CIRCUIT_NODE:
		case CLASS_COND_NODE:
			return true;
			break;

		default:
			throw qcircuit_construction_fail("bad node type");
		}

		return false;
	}
};

typedef AbstractQuantumCircuit* (*CreateQCircuit)();

/**
 * @brief Factory for class AbstractQuantumCircuit
 * @ingroup QuantumCircuit
 */
class QuantumCircuitFactory
{
public:

    void registClass(std::string name, CreateQCircuit method);
    AbstractQuantumCircuit * getQuantumCircuit(std::string &);

	/**
     * @brief Get the static instance of factory 
	 * @return QuantumCircuitFactory &
     */
    static QuantumCircuitFactory & getInstance()
    {
        static QuantumCircuitFactory  s_Instance;
        return s_Instance;
    }
private:
    std::map<std::string, CreateQCircuit> m_QCirciutMap;
    QuantumCircuitFactory() {}
};

/**
* @brief QCircuit program register action
* @note Provide QuantumCircuitFactory class registration interface for the outside
 */
class QuantumCircuitRegisterAction {
public:
    QuantumCircuitRegisterAction(std::string className, CreateQCircuit ptrCreateFn) {
        QuantumCircuitFactory::getInstance().registClass(className, ptrCreateFn);
    }
};

template<typename T>
QCircuit & QCircuit::operator<<(T node)
{
    if (nullptr == this->m_pQuantumCircuit)
    {
        throw std::runtime_error("m_pQuantumCircuit is null");
    }

	m_pQuantumCircuit->pushBackNode(std::dynamic_pointer_cast<QNode>(node.getImplementationPtr()));
    return *this;
}


#define REGISTER_QCIRCUIT(className)                                           \
    AbstractQuantumCircuit* QCircuitCreator##className(){                           \
        return new className();                                                   \
    }                                                                           \
    QuantumCircuitRegisterAction g_qCircuitCreatorDoubleRegister##className(     \
        #className,(CreateQCircuit)QCircuitCreator##className)

QPANDA_END
#endif // !_QCIRCUIT_H
