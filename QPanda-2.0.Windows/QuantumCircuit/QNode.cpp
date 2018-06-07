#include "QNode.h"


QNodeVector _G_QNodeVector;
QNodeVector::QNodeVector()
{

}

QNodeVector::~QNodeVector()
{
    for (auto aiter = m_pQNodeVector.begin(); aiter != m_pQNodeVector.end(); aiter++)
    {
        QNode * pNode = *aiter;
        //std::cout<<"position = " << pNode->getPosition() << endl;
        //cout << "nodetype ="<< pNode->getNodeType() << endl;
        delete (pNode);
    }
}

bool QNodeVector::pushBackNode(QNode * pNode)
{
    WriteLock wl(m_sm);
    m_pQNodeVector.push_back(pNode);
    return true;
}

size_t QNodeVector::getLastNode()
{
    ReadLock rl(m_sm);
    return m_pQNodeVector.size();
}

/*
bool QNodeVector::setHeadNode(QNode * prog)
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

vector<QNode*>::iterator QNodeVector::getNode(int iNum)
{
    ReadLock rl(m_sm);
    if (iNum > m_pQNodeVector.size()||(iNum == -1))
    {
        return m_pQNodeVector.end();
    }
    return m_pQNodeVector.begin() + (iNum - 1);
}

vector<QNode*>::iterator QNodeVector::getEnd()
{
    return  m_pQNodeVector.end();
}

