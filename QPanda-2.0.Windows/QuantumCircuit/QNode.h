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
    map<int, MapNode>::iterator getEnd();
};

extern QNodeMap _G_QNodeMap;

#endif // !_QNODE_H

