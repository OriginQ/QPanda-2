/*! \file QNode.h */
#ifndef _QNODE_H
#define _QNODE_H

#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/QuantumGateParameter.h"
#include <vector>
#include <map>
#include <memory>

QPANDA_BEGIN

typedef int64_t qmap_size_t;

class QObject
{
    /**
    * @brief  Get current node type
    * @return     NodeType  current node type
    * @see  NodeType
    */
    virtual NodeType getNodeType() const = 0;
};

/**
* @class QNode
* @brief   Quantum node basic abstract class
* @ingroup QuantumCircuit
*/
class QNode : public QObject
{
public:
    virtual NodeType getNodeType() const = 0;
    virtual ~QNode() {}
};

/**
* @brief Item basic abstract class
* @ingroup QuantumCircuit
*/
class Item
{
public:
    virtual Item * getNext()const = 0;
    virtual Item * getPre() const = 0;
    virtual std::shared_ptr<QNode> getNode()const = 0;
    virtual void setNext(Item *) = 0;
    virtual void setPre(Item *) = 0;
    virtual void setNode(std::shared_ptr<QNode> pNode) = 0;
    virtual ~Item() {}
};

/**
* @brief Origin item implementation class
* @ingroup QuantumCircuit
*/
class  OriginItem : public Item
{
private:
    Item *m_pNext;			/**< next item pointer*/
    Item *m_pPre;				/**< previous item pointer*/
    std::shared_ptr<QNode> m_node;  /**< QNode shared pointer*/
public:
    OriginItem();
    ~OriginItem();
    Item * getNext()const;
    Item * getPre()const;
    std::shared_ptr<QNode> getNode() const;
    void setNext(Item * pItem);
    void setPre(Item * pItem);
    void setNode(std::shared_ptr<QNode> pNode);
};


class NodeIter
{
private:
    Item * m_pCur;
public:
    NodeIter(Item * pItem)
    {
        m_pCur = pItem;
    }

    NodeIter(const NodeIter & oldIter)
    {
        this->m_pCur = oldIter.getPCur();
    }

    Item * getPCur() const
    {
        return this->m_pCur;
    }

    void setPCur(Item * pItem)
    {
        m_pCur = pItem;
    }
    NodeIter()
    {
        m_pCur = nullptr;
    }

    NodeIter & operator++();
    NodeIter operator++(int);
    std::shared_ptr<QNode>operator*() const;
    NodeIter &operator--();
    NodeIter operator--(int);
    NodeIter getNextIter();
    bool operator!=(NodeIter) const;
    bool operator==(NodeIter) const;
};

/**
* @class AbstractNodeManager
* @brief   Quantum node manager basic abstract class
* @ingroup QuantumCircuit
*/
class AbstractNodeManager
{
public:
	virtual ~AbstractNodeManager() {}

	/**
	* @brief  Get the first NodeIter
	* @return  NodeIter
	*/
	virtual NodeIter getFirstNodeIter() = 0;

	/**
	* @brief  Get the last NodeIter
	* @return  NodeIter
	*/
	virtual NodeIter getLastNodeIter() = 0;

	/**
	* @brief  Get the end NodeIter
	* @return  NodeIter
	*/
	virtual NodeIter getEndNodeIter() = 0;

	/**
	* @brief  Get the head NodeIter
	* @return  NodeIter
	*/
	virtual NodeIter getHeadNodeIter() = 0;

	/**
	* @brief  Insert a new QNode at the location specified by NodeIter
	* @param[in] NodeIter&  specified  location
	* @param[in] std::shared_ptr<QNode> Inserted QNode
	* @return  NodeIter
	*/
	virtual NodeIter insertQNode(const NodeIter &, std::shared_ptr<QNode>) = 0;

	/**
	* @brief  Delete a QNode at the location specified by NodeIter
	* @param[in] NodeIter&  specified  location
	* @return  NodeIter  Deleted NodeIter
	*/
	virtual NodeIter deleteQNode(NodeIter &) = 0;

	/**
	* @brief  Insert a new Node at the end of current quantum circuit
	* @param[in]  QNode*  quantum node
	* @return     void
	* @see  QNode
	*/
	virtual void pushBackNode(std::shared_ptr<QNode>) = 0;
};

QPANDA_END

#endif // !_QNODE_H

