/*! \file QCircuit.h */
#ifndef _QCIRCUIT_H
#define _QCIRCUIT_H

#include <complex>
#include <initializer_list>
#include <vector>
#include <iterator>
#include <map>

#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/ReadWriteLock.h"
#include "Core/Utilities/QPandaException.h"
QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @class AbstractQuantumCircuit
* @brief Quantum circuit basic abstract class
* @ingroup Core
*/
class AbstractQuantumCircuit
{
public:
    virtual NodeIter getFirstNodeIter() = 0;
    virtual NodeIter getLastNodeIter() = 0;
    virtual NodeIter getEndNodeIter() = 0;
    virtual NodeIter getHeadNodeIter() = 0;
    virtual NodeIter insertQNode(NodeIter &, QNode *) = 0;
    virtual NodeIter deleteQNode(NodeIter &) = 0;
    virtual void pushBackNode(QNode *) = 0;
    virtual void pushBackNode(std::shared_ptr<QNode> ) = 0;
    virtual bool isDagger() const = 0;
    virtual bool getControlVector(QVec &) = 0;
    virtual void  setDagger(bool isDagger) = 0;
    virtual void  setControl(QVec ) = 0;
    virtual void clearControl() = 0;
    virtual ~AbstractQuantumCircuit() {};
};

/**
* @class QCircuit
* @brief Quantum circuit basic abstract class
* @ingroup Core
*/
class QCircuit : public QNode, public AbstractQuantumCircuit
{
protected:
    std::shared_ptr<AbstractQuantumCircuit> m_pQuantumCircuit;
public:
    QCircuit();
    QCircuit(const QCircuit &);
    ~QCircuit();
    std::shared_ptr<QNode> getImplementationPtr();

    /**
    * @brief  Insert new Node at the end of current quantum circuit node
    * @param[in]  QNode*  quantum node
    * @return     void
    * @see  QNode
    */
    void pushBackNode(QNode *);
    void pushBackNode(std::shared_ptr<QNode>) ;

    /**
    * @brief  Insert new Node at the end of current node
    * @param[in]  Node  QGate/QCircuit
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
    virtual QCircuit  control(QVec &);

    /**
    * @brief  Get current node type
    * @return     NodeType  current node type
    * @see  NodeType
    */
    NodeType getNodeType() const;

    /**
    * @brief  Judge current quantum circuit is dagger
    * @retval   0  true
    * @retval   1  false
    */
    bool isDagger() const;

    /**
    * @brief  Get control vector fron current quantum circuit node
    * @param[in]  QVec& qubits  vector
    * @retval   0  true
    * @retval   1  false
    * @see QVec
    */
    bool getControlVector(QVec &);
    NodeIter getFirstNodeIter();
    NodeIter getLastNodeIter();
    NodeIter getEndNodeIter();
    NodeIter getHeadNodeIter();

    NodeIter insertQNode(NodeIter &iter, QNode *pNode);
    NodeIter deleteQNode(NodeIter &iter);

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
    virtual void  setControl(QVec );
private:
    void clearControl() {}
    void execute(QPUImpl *, QuantumGateParam *) {}
};

class HadamardQCircuit :public QCircuit
{
public:
    HadamardQCircuit(QVec & pQubitVector);
    ~HadamardQCircuit() {}
private:
    HadamardQCircuit();
};


/**
* @brief  QPanda2 basic interface for creating a empty circuit
* @return     QPanda::QCircuit  
* @ingroup Core
*/
QCircuit CreateEmptyCircuit();

/**
* @brief  Create a hadamard qcircuit
* @param[in]  QVec&  qubit vector 
* @return     QPanda::HadamardQCircuit  hadamard qcircuit 
*/
HadamardQCircuit CreateHadamardQCircuit(QVec & pQubitVector);


class OriginCircuit : public QNode, public AbstractQuantumCircuit
{
private:
    Item * m_head;
    Item * m_end;
    SharedMutex m_sm;
    NodeType m_node_type;
    bool m_Is_dagger;
    QVec m_control_qubit_vector;
    OriginCircuit(const OriginCircuit &);
    std::shared_ptr<QNode> getImplementationPtr()
    {
        QCERR("Can't use this function");
        throw std::runtime_error("Can't use this function");
    }
public:
    OriginCircuit():
        m_node_type(CIRCUIT_NODE),
        m_head(nullptr),
        m_end(nullptr),
        m_Is_dagger(false)
    {
        m_control_qubit_vector.resize(0);
    }
    ~OriginCircuit();
    void pushBackNode(QNode *);
    void pushBackNode(std::shared_ptr<QNode>);
    void setDagger(bool);
    void setControl(QVec );
    NodeType getNodeType() const;
    bool isDagger() const;
    bool getControlVector(QVec &);
    NodeIter  getFirstNodeIter();
    NodeIter  getLastNodeIter();
    NodeIter  getEndNodeIter();
    NodeIter getHeadNodeIter();
    NodeIter  insertQNode(NodeIter &, QNode *);
    NodeIter  deleteQNode(NodeIter &);
    void clearControl();
    void execute(QPUImpl *, QuantumGateParam *);
};

typedef AbstractQuantumCircuit* (*CreateQCircuit)();
class QuantumCircuitFactory
{
public:

    void registClass(std::string name, CreateQCircuit method);
    AbstractQuantumCircuit * getQuantumCircuit(std::string &);

    static QuantumCircuitFactory & getInstance()
    {
        static QuantumCircuitFactory  s_Instance;
        return s_Instance;
    }
private:
    std::map<std::string, CreateQCircuit> m_QCirciutMap;
    QuantumCircuitFactory() {}
};

class QuantumCircuitRegisterAction {
public:
    QuantumCircuitRegisterAction(std::string className, CreateQCircuit ptrCreateFn) {
        QuantumCircuitFactory::getInstance().registClass(className, ptrCreateFn);
    }
};

template<typename T>
QCircuit & QCircuit::operator<<(T node)
{
    auto temp = dynamic_cast<QNode *>(&node);
    if (nullptr == temp)
    {
        throw std::invalid_argument("param is not QNode");
    }
    if (nullptr == this->m_pQuantumCircuit)
    {
        throw std::runtime_error("m_pQuantumCircuit is null");
    }
    int iNodeType = temp->getNodeType();

    switch (iNodeType)
    {
    case GATE_NODE:
    case CIRCUIT_NODE:
    case CLASS_COND_NODE:
        m_pQuantumCircuit->pushBackNode(dynamic_cast<QNode*>(&node));
        break;
    default:
        throw qcircuit_construction_fail("bad node type");
    }
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
