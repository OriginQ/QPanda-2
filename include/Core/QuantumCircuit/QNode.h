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
/**
* @namespace QPanda
*/
QPANDA_BEGIN
typedef int64_t qmap_size_t;

/**
* @class QNode
* @brief   Quantum node basic abstract class
* @ingroup Core
*/
class QNode
{
public:
    /**
    * @brief  Get current node type
    * @return     NodeType  current node type
    * @see  NodeType
    */
    virtual NodeType getNodeType() const = 0;
    virtual std::shared_ptr<QNode> getImplementationPtr() = 0;
    virtual void execute(QPUImpl *,QuantumGateParam *) = 0;
    virtual ~QNode() {}
};

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

class  OriginItem : public Item
{
private:
    Item *m_pNext;
    Item *m_pPre;
    std::shared_ptr<QNode> m_node;
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
    std::shared_ptr<QNode>operator*();
    NodeIter &operator--();
    NodeIter operator--(int);
    NodeIter getNextIter();
    bool operator!=(NodeIter);
    bool operator==(NodeIter);
};

QPANDA_END

#endif // !_QNODE_H

