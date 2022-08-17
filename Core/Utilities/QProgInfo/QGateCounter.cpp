#include "Core/Utilities/QProgInfo/QGateCounter.h"
using namespace std;
USING_QPANDA

QGateCounter::QGateCounter() :
    m_count(0),
    m_qgate_num_map()
{

}

QGateCounter::~QGateCounter()
{

}

void QGateCounter::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
    if (cur_node) {
        m_count++;
        GateType gatetype = static_cast<GateType>(cur_node->getQGate()->getGateType());
        m_qgate_num_map[gatetype]++;
    }
}

size_t QGateCounter::count()
{
    return m_count;
}

const std::map<GateType, size_t> QGateCounter::getGateMap() {
    return m_qgate_num_map;
}