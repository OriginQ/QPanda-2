#include "QNode.h"
USING_QPANDA
using namespace std;
QNodeMap &QNodeMap::getInstance()
{
    static QNodeMap node_map;
    return node_map;
}

QNodeMap::QNodeMap() :m_sCount(0)
{ }

QNodeMap::~QNodeMap()
{
    auto aiter = m_pQNodeMap.begin();
    while (aiter != m_pQNodeMap.end())
    {
        QNode * pNode = aiter->second.m_pNode;
        qmap_size_t iRef = aiter->second.m_iReference;

        delete (pNode);
        aiter = m_pQNodeMap.erase(aiter);
    }
}

qmap_size_t QNodeMap::pushBackNode(QNode *pNode)
{
    WriteLock wl(m_sm);
    MapNode temp = { 0, pNode };
    m_sCount++;
    auto a =m_pQNodeMap.insert(pair<qmap_size_t, MapNode>(m_sCount,temp));
    return m_sCount;
}


QNode *QNodeMap::getNode(qmap_size_t iNum)
{
    ReadLock rl(m_sm);
    if (iNum == -1)
    {
        return nullptr;
    }
    auto aiter = m_pQNodeMap.find(iNum);
    if (m_pQNodeMap.end() == aiter)
        return nullptr;
    return aiter->second.m_pNode;
}

bool QNodeMap::addNodeRefer(qmap_size_t sNum)
{
    WriteLock wl(m_sm);
    auto aiter = m_pQNodeMap.find(sNum);
    if (m_pQNodeMap.end() == aiter)
        return false;
    aiter->second.m_iReference++;
    return true;
}

bool QNodeMap::deleteNode(qmap_size_t sNum)
{

    ReadLock * rl = new ReadLock(m_sm);
    WriteLock * wl = nullptr;
    auto aiter = m_pQNodeMap.find(sNum);
    if (m_pQNodeMap.end() == aiter)
    {
        delete rl;
        return false;
    }
        
    if (aiter->second.m_iReference > 1)
    {
        delete rl;
        wl = new WriteLock(m_sm);
        aiter->second.m_iReference--;
        delete wl;
    }
    else
    {
        delete rl;
        if (nullptr != aiter->second.m_pNode)
        {
            delete aiter->second.m_pNode;
            aiter->second.m_pNode = nullptr;
        }
        WriteLock wl(m_sm);
        m_pQNodeMap.erase(aiter);
    }
    return true;
}

map<qmap_size_t, MapNode>::iterator QNodeMap::getEnd()
{
    return m_pQNodeMap.end();
}

OriginItem::OriginItem() :m_iNodeNum(-1), m_pNext(nullptr), m_pPre(nullptr)
{ }

OriginItem::~OriginItem()
{
    QNodeMap::getInstance().deleteNode(m_iNodeNum);
}

Item * OriginItem::getNext() const
{
    return m_pNext;
}

Item * OriginItem::getPre() const
{
    return m_pPre;
}

QNode *OriginItem::getNode() const
{
    auto aiter = QNodeMap::getInstance().getNode(m_iNodeNum);
    return aiter;
}

void  OriginItem::setNext(Item *pItem)
{    
    m_pNext = pItem;
}

void OriginItem::setPre(Item *pItem)
{
    m_pPre = pItem;
}

void OriginItem::setNode(QNode *pNode)
{
    if (m_iNodeNum != -1)
    {
        QNodeMap::getInstance().deleteNode(m_iNodeNum);
    }
    m_iNodeNum = pNode->getPosition();
    if (!QNodeMap::getInstance().addNodeRefer(m_iNodeNum))
    {
        QCERR("unknow error");
        throw runtime_error("unknown error");
    }
}
