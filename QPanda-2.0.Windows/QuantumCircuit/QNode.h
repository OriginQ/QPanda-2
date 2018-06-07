#ifndef _QNODE_H
#define _QNODE_H

#include "ReadWriteLock.h"
#include <vector>
#include "QuantumCircuit/QGlobalVariable.h"
class QNode
{
public:
    virtual NodeType getNodeType() const = 0;
    virtual int getPosition() const = 0;
    virtual ~QNode() {};
};

class QNodeVector
{
private:
    SharedMutex m_sm;
    vector<QNode*> m_pQNodeVector;
    vector<QNode*>::iterator m_currentIter;
public:
    QNodeVector();
    ~QNodeVector();

    bool pushBackNode(QNode *);
    size_t getLastNode();
    //bool setHeadNode(QNode *);

    vector <QNode *>::iterator getNode(int);
    vector <QNode *>::iterator getEnd();
};

extern QNodeVector _G_QNodeVector;

#endif // !_QNODE_H

