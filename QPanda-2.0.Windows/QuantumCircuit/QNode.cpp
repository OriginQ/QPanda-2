#include "QNode.h"
#include "QPanda/QPandaException.h"

QNodeMap _G_QNodeMap;
QNodeMap::QNodeMap() :m_sCount(0)
{

}

QNodeMap::~QNodeMap()
{
    for (auto aiter = m_pQNodeVector.begin(); aiter != m_pQNodeVector.end(); aiter++)
    {
        QNode * pNode = aiter->second.m_pNode;
        //std::cout<<"position = " << pNode->getPosition() << endl;
        //cout << "nodetype ="<< pNode->getNodeType() << endl;
        delete (pNode);
    }
}

QMAP_SIZE QNodeMap::pushBackNode(QNode * pNode)
{
    WriteLock wl(m_sm);
    MapNode temp = { 1, pNode };
    m_sCount++;
    auto a =m_pQNodeVector.insert(pair<QMAP_SIZE, MapNode>(m_sCount,temp));
    return m_sCount;
}

/*
bool QNodeMap::setHeadNode(QNode * prog)
{
WriteLock wl(m_sm);
if (prog->getPosition() > m_pQNodeVector.size())
{
return false;
}
m_currentIter = m_pQNodeVector.begin() + (prog->getPosition() - 1);
return true;
}
*/

QNode * QNodeMap::getNode(QMAP_SIZE iNum)
{
    ReadLock rl(m_sm);
    if (iNum > m_pQNodeVector.size()||(iNum == -1))
    {
        return nullptr;
    }
    auto aiter = m_pQNodeVector.find(iNum);
    if (m_pQNodeVector.end() == aiter)
        return nullptr;
    return aiter->second.m_pNode;
}

map<int, MapNode>::iterator QNodeMap::getEnd()
{
    return  m_pQNodeVector.end();
}

