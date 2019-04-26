#include "QGateCounter.h"
using namespace std;
USING_QPANDA

QGateCounter::QGateCounter() :
    m_count(0)
{

}

QGateCounter::~QGateCounter()
{

}

void QGateCounter::execute(AbstractQGateNode *cur_node, QNode *parent_node)
{
    m_count++;
}

void QGateCounter::execute(AbstractQuantumMeasure *cur_node, QNode *parent_node)
{
    m_count++;
}

size_t QGateCounter::count()
{
    return m_count;
}

