#include "QNode.h"
USING_QPANDA
using namespace std;

OriginItem::OriginItem(): m_pNext(nullptr), m_pPre(nullptr)
{ }

OriginItem::~OriginItem()
{
    m_node.reset();
}

Item * OriginItem::getNext() const
{
    return m_pNext;
}

Item * OriginItem::getPre() const
{
    return m_pPre;
}

shared_ptr<QNode>OriginItem::getNode() const
{
    if (!m_node)
    {
        QCERR("m_node is nullpt");
        throw runtime_error("m_node is nullptr");
    }
    return m_node;
}

void  OriginItem::setNext(Item *pItem)
{    
    m_pNext = pItem;
}

void OriginItem::setPre(Item *pItem)
{
    m_pPre = pItem;
}

void OriginItem::setNode(std::shared_ptr<QNode> pNode)
{
    if (!pNode)
    {
        QCERR("pNode is nullptr");
        throw invalid_argument("pNode is nullptr");
    }
    m_node=pNode;
}

NodeIter& NodeIter::operator++()
{
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getNext();
    }
    return *this;
}

NodeIter NodeIter::operator++(int)
{
    NodeIter temp(*this);
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getNext();
    }
    return temp;
}

shared_ptr<QNode> NodeIter::operator*()
{
    if (m_pCur)
    {
        return m_pCur->getNode();
    }
    else
    {
        QCERR("m_pCur is nullptr");
        throw invalid_argument("m_pCur is nullptr");
    }
}

NodeIter& NodeIter::operator--()
{
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getPre();
    }
    return *this;
}

NodeIter NodeIter::operator--(int i)
{
    NodeIter temp(*this);
    if (nullptr != m_pCur)
    {
        this->m_pCur = m_pCur->getPre();

    }
    return temp;
}

NodeIter NodeIter::getNextIter()
{
    if (nullptr != m_pCur)
    {
        auto pItem = m_pCur->getNext();
        NodeIter temp(pItem);
        return temp;
    }
    else
    {
        NodeIter temp(nullptr);
        return temp;
    }
}

bool NodeIter::operator!=(NodeIter  iter)
{
    return this->m_pCur != iter.m_pCur;
}

bool NodeIter::operator==(NodeIter iter)
{
    return this->m_pCur == iter.m_pCur;
}
