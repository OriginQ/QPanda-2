#ifndef _QNODE_H
#define _QNODE_H

#include "Utilities/ReadWriteLock.h"
#include <vector>
#include <map>
#include "QuantumCircuit/QGlobalVariable.h"
#include "QPandaNamespace.h"
QPANDA_BEGIN
typedef int64_t qmap_size_t;
class QNode
{
public:
    virtual NodeType getNodeType() const = 0;
    virtual qmap_size_t getPosition() const = 0;
    virtual void setPosition(qmap_size_t) = 0;
    virtual ~QNode() {};
};

struct MapNode
{
    qmap_size_t m_iReference;
    QNode   * m_pNode;
    MapNode(const MapNode& old)
    {
        m_iReference = old.m_iReference;
        m_pNode = old.m_pNode;
    }
    MapNode(qmap_size_t iCount, QNode * pNode) :m_iReference(iCount), m_pNode(pNode)
    {
    }
};

class QNodeMap
{
protected:
    QNodeMap();
    QNodeMap(const QNodeMap &);
    QNodeMap &operator =(const QNodeMap &);
private:
    SharedMutex m_sm;
    qmap_size_t m_sCount;
    std::map<qmap_size_t, MapNode> m_pQNodeMap;
    std::map<qmap_size_t, MapNode>::iterator m_currentIter;
public:
    static QNodeMap &getInstance();
    ~QNodeMap();

    qmap_size_t pushBackNode(QNode *);
    //bool setHeadNode(QNode *);
    
    QNode * getNode(qmap_size_t);
    bool addNodeRefer(qmap_size_t sNum);
    bool deleteNode(qmap_size_t);
    std::map<qmap_size_t, MapNode>::iterator getEnd();
};

class Item
{
public:
    virtual Item * getNext()const = 0;
    virtual Item * getPre() const = 0;
    virtual QNode * getNode() const = 0;
    virtual void setNext(Item *) = 0;
    virtual void setPre(Item *) = 0;
    virtual void setNode(QNode *) = 0;
    virtual ~Item() {};
};

class  OriginItem : public Item
{
private:
    Item *m_pNext;
    Item *m_pPre;
    qmap_size_t m_iNodeNum;
public:
    OriginItem();
    ~OriginItem();
    Item * getNext()const;
    Item * getPre()const;
    QNode * getNode() const;
    void setNext(Item * pItem);
    void setPre(Item * pItem);
    void setNode(QNode * pNode);
};
QPANDA_END

#endif // !_QNODE_H

