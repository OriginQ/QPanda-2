#include "Core/Utilities/QProgInfo/QGateCounter.h"
using namespace std;
USING_QPANDA

QGateCounter::QGateCounter() :
    m_count(0)
{

}

QGateCounter::~QGateCounter()
{

}

void QGateCounter::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
    if(cur_node)
        m_count++;
}

size_t QGateCounter::count()
{
    return m_count;
}

