#ifndef _QNODE_H
#define _QNODE_H

#include "ReadWriteLock.h"
#include <vector>
#include <map>
#include "QuantumCircuit/QGlobalVariable.h"
class QNode
{
public:
    virtual NodeType getNodeType() const = 0;
    virtual int getPosition() const = 0;
    virtual ~QNode() {};
};

typedef int QMAP_SIZE;

struct MapNode
{
    QMAP_SIZE m_iReference;
    QNode   * m_pNode;
    MapNode(const MapNode& old)
    {
        m_iReference = old.m_iReference;
        m_pNode = old.m_pNode;
    }
    MapNode(QMAP_SIZE iCount, QNode * pNode) :m_iReference(iCount), m_pNode(pNode)
    {
    }
};


class QNodeMap
{


private:
    SharedMutex m_sm;
    QMAP_SIZE m_sCount;
    map<int, MapNode> m_pQNodeVector;
    map<int, MapNode>::iterator m_currentIter;
public:
    QNodeMap();
    ~QNodeMap();

    QMAP_SIZE pushBackNode(QNode *);
    //bool setHeadNode(QNode *);
    
    QNode * getNode(QMAP_SIZE);
    bool addNodeRefer(QMAP_SIZE sNum);
    bool deleteNode(QMAP_SIZE);
    map<int, MapNode>::iterator getEnd();
};




class  Item
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
    Item * m_pNext;
    Item * m_pPre;
    int    m_iNodeNum;
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


extern QNodeMap _G_QNodeMap;

#endif // !_QNODE_H

